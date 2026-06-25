"""Search schematic components and highlight matches on page images."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from db import PageRecord, SchematicRepository
from storage import StorageBackend

MatchMode = Literal["exact", "contains", "regex"]


@dataclass
class ComponentMatch:
    name: str
    kind: str
    page_number: int
    bbox: dict[str, int]
    image_link: str
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "image_link": self.image_link,
            "score": self.score,
        }


@dataclass
class HighlightResult:
    schematic_id: str
    page_number: int
    page_id: int | str
    components: list[str]
    match_count: int
    highlight_image_link: str
    matches: list[ComponentMatch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schematic_id": self.schematic_id,
            "page_number": self.page_number,
            "page_id": self.page_id,
            "components": self.components,
            "match_count": self.match_count,
            "highlight_image_link": self.highlight_image_link,
            "matches": [m.to_dict() for m in self.matches],
        }


class SchematicRetriever:
    """Retrieve page rows and highlight requested components."""

    def __init__(
        self,
        repository: SchematicRepository,
        storage: StorageBackend,
    ) -> None:
        self.repository = repository
        self.storage = storage

    def find_pages_for_components(
        self,
        schematic_id: str,
        component_names: list[str],
        *,
        match_all: bool = False,
    ) -> list[PageRecord]:
        cleaned = [name.strip() for name in component_names if name.strip()]
        return self.repository.find_pages_by_components(
            schematic_id,
            cleaned,
            match_all=match_all,
        )

    def resolve_component_bboxes(
        self,
        page: PageRecord,
        component_names: list[str],
        *,
        mode: MatchMode = "exact",
        case_sensitive: bool = False,
    ) -> list[ComponentMatch]:
        """Resolve bboxes for requested components on a single page row."""
        matches: list[ComponentMatch] = []
        image_link = page.smetadata.get("image_link", "")
        page_number = int(page.smetadata.get("page_number", 0))

        for query in component_names:
            for name, entries in page.words_bboxes.items():
                if not self._is_match(query, name, mode=mode, case_sensitive=case_sensitive):
                    continue
                score = self._match_score(query, name, mode=mode, case_sensitive=case_sensitive)
                for entry in entries:
                    matches.append(
                        ComponentMatch(
                            name=name,
                            kind=entry.get("kind", "text"),
                            page_number=page_number,
                            bbox={k: int(entry[k]) for k in ("x0", "y0", "x1", "y1")},
                            image_link=image_link,
                            score=score,
                        )
                    )

        matches.sort(key=lambda m: (-m.score, m.name))
        return matches

    def highlight_components(
        self,
        schematic_id: str,
        component_names: list[str],
        *,
        match_all: bool = False,
        mode: MatchMode = "exact",
        case_sensitive: bool = False,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        padding: int = 4,
        show_label: bool = True,
    ) -> list[HighlightResult]:
        """
        Find pages containing the requested components, highlight them,
        upload highlight images, and return result metadata.
        """
        pages = self.find_pages_for_components(
            schematic_id,
            component_names,
            match_all=match_all,
        )
        if not pages:
            return []

        results: list[HighlightResult] = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        comp_slug = self._safe_filename("_".join(component_names[:5]))

        for page in pages:
            page_matches = self.resolve_component_bboxes(
                page,
                component_names,
                mode=mode,
                case_sensitive=case_sensitive,
            )
            if not page_matches:
                continue

            image_link = page.smetadata["image_link"]
            image_bytes = self.storage.download_bytes(image_link)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to decode page image: {image_link}")

            for match in page_matches:
                draw_bbox(
                    image,
                    match.bbox,
                    label=match.name if show_label else None,
                    color=color,
                    thickness=thickness,
                    padding=padding,
                )

            page_number = int(page.smetadata.get("page_number", 0))
            highlight_key = (
                f"{schematic_id}/highlights/{stamp}_{comp_slug}_page_{page_number:03d}.png"
            )
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError("Failed to encode highlighted image")

            highlight_link = self.storage.upload_bytes(
                highlight_key,
                encoded.tobytes(),
                content_type="image/png",
            )

            results.append(
                HighlightResult(
                    schematic_id=schematic_id,
                    page_number=page_number,
                    page_id=page.id,
                    components=list(component_names),
                    match_count=len(page_matches),
                    highlight_image_link=highlight_link,
                    matches=page_matches,
                )
            )

        return results

    @staticmethod
    def _normalize(value: str, case_sensitive: bool) -> str:
        return value if case_sensitive else value.lower()

    def _is_match(
        self,
        query: str,
        name: str,
        *,
        mode: MatchMode,
        case_sensitive: bool,
    ) -> bool:
        q = self._normalize(query.strip(), case_sensitive)
        n = self._normalize(name.strip(), case_sensitive)

        if mode == "exact":
            return q == n
        if mode == "contains":
            return q in n
        if mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.search(query, name, flags) is not None
        return False

    @staticmethod
    def _match_score(
        query: str,
        name: str,
        *,
        mode: MatchMode,
        case_sensitive: bool,
    ) -> float:
        q = query.strip()
        n = name.strip()
        q_cmp = q if case_sensitive else q.lower()
        n_cmp = n if case_sensitive else n.lower()

        if mode == "exact":
            return 1.0 if q_cmp == n_cmp else 0.0
        if mode == "contains":
            if q_cmp == n_cmp:
                return 1.0
            if n_cmp.startswith(q_cmp):
                return 0.9
            return 0.7
        return 0.5

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = re.sub(r"[^\w\-]+", "_", value.strip())
        return cleaned[:80] or "components"


def draw_bbox(
    image: np.ndarray,
    bbox: dict[str, int],
    *,
    label: str | None,
    color: tuple[int, int, int],
    thickness: int,
    padding: int,
) -> None:
    x0 = max(0, bbox["x0"] - padding)
    y0 = max(0, bbox["y0"] - padding)
    x1 = min(image.shape[1] - 1, bbox["x1"] + padding)
    y1 = min(image.shape[0] - 1, bbox["y1"] + padding)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        label_y0 = max(0, y0 - th - baseline - 4)
        cv2.rectangle(
            image,
            (x0, label_y0),
            (x0 + tw + 4, label_y0 + th + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x0 + 2, label_y0 + th + 2),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Legacy local-file retriever (backward compatible)
# ---------------------------------------------------------------------------

class LocalSchematicRetriever:
    """Load processed schematic data from components.json on disk."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.json_path = self.data_dir / "components.json"
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"components.json not found in {self.data_dir}. Run schematic_processor first."
            )
        with self.json_path.open(encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)
        self.pages: list[dict[str, Any]] = self.manifest.get("pages", [])

    def search(
        self,
        query: str,
        *,
        mode: MatchMode = "contains",
        kind: str | None = None,
        page_number: int | None = None,
        case_sensitive: bool = False,
    ) -> list[ComponentMatch]:
        if not query.strip():
            return []

        retriever = SchematicRetriever.__new__(SchematicRetriever)
        matches: list[ComponentMatch] = []

        for page in self.pages:
            if page_number is not None and page["page_number"] != page_number:
                continue

            image_path = str(self.data_dir / page["image_path"])
            words_bboxes = page.get("words_bboxes")
            if not words_bboxes:
                # Backward compat with older components list format
                words_bboxes = {}
                for comp in page.get("components", []):
                    words_bboxes.setdefault(comp["name"], []).append(
                        {**comp["bbox"], "kind": comp.get("kind", "text")}
                    )

            for name, entries in words_bboxes.items():
                for entry in entries:
                    if kind is not None and entry.get("kind") != kind:
                        continue
                    if not retriever._is_match(
                        query, name, mode=mode, case_sensitive=case_sensitive
                    ):
                        continue
                    score = retriever._match_score(
                        query, name, mode=mode, case_sensitive=case_sensitive
                    )
                    matches.append(
                        ComponentMatch(
                            name=name,
                            kind=entry.get("kind", "text"),
                            page_number=page["page_number"],
                            bbox={k: int(entry[k]) for k in ("x0", "y0", "x1", "y1")},
                            image_link=image_path,
                            score=score,
                        )
                    )

        matches.sort(key=lambda m: (-m.score, m.page_number, m.name))
        return matches

    def highlight(
        self,
        query: str,
        *,
        output_dir: str | Path | None = None,
        mode: MatchMode = "contains",
        kind: str | None = None,
        page_number: int | None = None,
        case_sensitive: bool = False,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        padding: int = 4,
        show_label: bool = True,
        display: bool = False,
    ) -> list[dict[str, Any]]:
        matches = self.search(
            query,
            mode=mode,
            kind=kind,
            page_number=page_number,
            case_sensitive=case_sensitive,
        )
        if not matches:
            return []

        if output_dir is None:
            output_dir = self.data_dir / "highlighted"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        by_page: dict[int, list[ComponentMatch]] = {}
        for match in matches:
            by_page.setdefault(match.page_number, []).append(match)

        results: list[dict[str, Any]] = []
        for page_num, page_matches in sorted(by_page.items()):
            image_path = Path(page_matches[0].image_link)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            for match in page_matches:
                draw_bbox(
                    image,
                    match.bbox,
                    label=match.name if show_label else None,
                    color=color,
                    thickness=thickness,
                    padding=padding,
                )

            out_name = f"highlight_{SchematicRetriever._safe_filename(query)}_page_{page_num:03d}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), image)

            results.append(
                {
                    "query": query,
                    "page_number": page_num,
                    "match_count": len(page_matches),
                    "matches": [m.to_dict() for m in page_matches],
                    "output_path": str(out_path),
                }
            )

            if display:
                cv2.imshow(f"Matches: {query} (page {page_num})", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Search schematic components and highlight them on page images."
    )
    parser.add_argument("query", help="Component name or substring to search for")
    parser.add_argument("-d", "--data-dir", default="output")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--mode", choices=["exact", "contains", "regex"], default="contains")
    parser.add_argument("--kind", choices=["reference_designator", "text"], default=None)
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    retriever = LocalSchematicRetriever(args.data_dir)
    matches = retriever.search(
        args.query,
        mode=args.mode,
        kind=args.kind,
        page_number=args.page,
        case_sensitive=args.case_sensitive,
    )
    if not matches:
        print(f"No matches found for '{args.query}'")
        return

    print(f"Found {len(matches)} match(es) for '{args.query}':")
    for match in matches:
        print(
            f"  page {match.page_number}: {match.name} "
            f"bbox={match.bbox} kind={match.kind}"
        )

    results = retriever.highlight(
        args.query,
        output_dir=args.output_dir,
        mode=args.mode,
        kind=args.kind,
        page_number=args.page,
        case_sensitive=args.case_sensitive,
        display=args.display,
    )
    for result in results:
        print(f"Saved highlighted image: {result['output_path']}")


if __name__ == "__main__":
    main()
