"""Extract schematic component names and bounding boxes from PDF pages."""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz  # PyMuPDF
import numpy as np
import pdfplumber

from db import PageRecord, SchematicRepository
from storage import StorageBackend

# Maximum total pixels (width * height) for rendered page images.
MAX_PIXELS: int = 4_000_000

REF_DES_PATTERN = re.compile(
    r"^(?:"
    r"R|C|L|U|D|Q|J|Y|F|K|E|M|T|P|"
    r"SW|TP|FB|JP|JS|LED|XTAL|"
    r"CON|JMP|HDR|BT|X|VR|"
    r"QFP|SHLD|BOOT|DIR|ISP|ADC|GND"
    r")(?:\d+[A-Z]?|\d*)$",
    re.IGNORECASE,
)


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def scale(self, factor: float) -> "BBox":
        return BBox(
            x0=round(self.x0 * factor, 2),
            y0=round(self.y0 * factor, 2),
            x1=round(self.x1 * factor, 2),
            y1=round(self.y1 * factor, 2),
        )

    def to_int_dict(self) -> dict[str, int]:
        return {
            "x0": int(round(self.x0)),
            "y0": int(round(self.y0)),
            "x1": int(round(self.x1)),
            "y1": int(round(self.y1)),
        }


@dataclass
class Component:
    name: str
    bbox: BBox
    bbox_pdf: BBox
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "bbox": self.bbox.to_int_dict(),
            "bbox_pdf": asdict(self.bbox_pdf),
        }


@dataclass
class PageProcessResult:
    page_number: int
    pdf_width: float
    pdf_height: float
    image_width: int
    image_height: int
    render_zoom: float
    downscale_factor: float
    total_scale: float
    image_png: bytes
    words_bboxes: dict[str, list[dict[str, Any]]]
    component_count: int


@dataclass
class ProcessResult:
    schematic_id: str
    source_pdf: str
    file_link: str
    page_count: int
    pages: list[PageRecord]


def compute_render_zoom(pdf_width: float, pdf_height: float, max_pixels: int) -> float:
    page_pixels = pdf_width * pdf_height
    if page_pixels <= max_pixels:
        return 1.0
    return (max_pixels / page_pixels) ** 0.5


def compute_downscale(image_width: int, image_height: int, max_pixels: int) -> float:
    total = image_width * image_height
    if total <= max_pixels:
        return 1.0
    return (max_pixels / total) ** 0.5


def classify_text(text: str) -> str:
    if REF_DES_PATTERN.match(text.strip()):
        return "reference_designator"
    return "text"


def word_to_bbox_pdf(word: dict[str, Any]) -> BBox:
    return BBox(
        x0=float(word["x0"]),
        y0=float(word["top"]),
        x1=float(word["x1"]),
        y1=float(word["bottom"]),
    )


def components_to_words_bboxes(components: list[Component]) -> dict[str, list[dict[str, Any]]]:
    """Group components into words_bboxes JSONB shape: name -> [bbox entries]."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comp in components:
        entry = {
            **comp.bbox.to_int_dict(),
            "kind": comp.kind,
        }
        grouped.setdefault(comp.name, []).append(entry)
    return grouped


def render_page_image(pdf_path: str, page_index: int, zoom: float) -> tuple[np.ndarray, int, int]:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n == 4:
            image = image[:, :, :3]
        return image, pixmap.width, pixmap.height
    finally:
        doc.close()


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode page image as PNG")
    return buffer.tobytes()


def _process_single_page(args: tuple[Any, ...]) -> PageProcessResult:
    """
    Worker entry point for parallel page processing.

    Must remain a top-level function for ProcessPoolExecutor on Windows.
    """
    (
        pdf_path,
        page_index,
        max_pixels,
        extract_all_text,
        components_only,
    ) = args

    page_number = page_index + 1

    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index]
        pdf_width = float(page.width)
        pdf_height = float(page.height)

        render_zoom = compute_render_zoom(pdf_width, pdf_height, max_pixels)
        image, img_w, img_h = render_page_image(str(pdf_path), page_index, render_zoom)

        downscale = compute_downscale(img_w, img_h, max_pixels)
        total_scale = render_zoom * downscale

        if downscale < 1.0:
            image = resize_image(image, downscale)
            img_w, img_h = image.shape[1], image.shape[0]

        components: list[Component] = []
        for word in page.extract_words(use_text_flow=True, keep_blank_chars=False):
            name = word.get("text", "").strip()
            if not name:
                continue

            kind = classify_text(name)
            if components_only and kind != "reference_designator":
                continue
            if not extract_all_text and kind != "reference_designator":
                continue

            bbox_pdf = word_to_bbox_pdf(word)
            bbox_image = bbox_pdf.scale(total_scale)
            components.append(
                Component(name=name, bbox=bbox_image, bbox_pdf=bbox_pdf, kind=kind)
            )

    return PageProcessResult(
        page_number=page_number,
        pdf_width=pdf_width,
        pdf_height=pdf_height,
        image_width=img_w,
        image_height=img_h,
        render_zoom=round(render_zoom, 6),
        downscale_factor=round(downscale, 6),
        total_scale=round(total_scale, 6),
        image_png=encode_png(image),
        words_bboxes=components_to_words_bboxes(components),
        component_count=len(components),
    )


class SchematicProcessor:
    """Process schematic PDFs, upload page images, and persist page metadata."""

    def __init__(
        self,
        storage: StorageBackend,
        repository: SchematicRepository,
        *,
        max_pixels: int = MAX_PIXELS,
        max_workers: int | None = None,
        extract_all_text: bool = True,
        components_only: bool = False,
    ) -> None:
        self.storage = storage
        self.repository = repository
        self.max_pixels = max_pixels
        self.max_workers = max_workers or min(os.cpu_count() or 1, 8)
        self.extract_all_text = extract_all_text
        self.components_only = components_only

    def process(
        self,
        pdf_path: str | Path,
        schematic_id: str,
        *,
        replace_existing: bool = True,
    ) -> ProcessResult:
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if replace_existing:
            self.repository.delete_schematic(schematic_id)

        file_key = f"{schematic_id}/source/{pdf_path.name}"
        file_link = self.storage.upload_file(
            file_key,
            pdf_path,
            content_type="application/pdf",
        )

        with pdfplumber.open(str(pdf_path)) as pdf:
            page_count = len(pdf.pages)

        worker_args = [
            (
                str(pdf_path),
                page_index,
                self.max_pixels,
                self.extract_all_text,
                self.components_only,
            )
            for page_index in range(page_count)
        ]

        page_results: list[PageProcessResult] = []
        workers = min(self.max_workers, page_count) if page_count else 1

        if workers <= 1:
            page_results = [_process_single_page(args) for args in worker_args]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_process_single_page, args) for args in worker_args]
                for future in as_completed(futures):
                    page_results.append(future.result())

        page_results.sort(key=lambda p: p.page_number)

        records: list[PageRecord] = []
        for page in page_results:
            image_key = f"{schematic_id}/pages/page_{page.page_number:03d}.png"
            image_link = self.storage.upload_bytes(
                image_key,
                page.image_png,
                content_type="image/png",
            )

            smetadata = {
                "schematic_id": schematic_id,
                "source_pdf": str(pdf_path),
                "file_link": file_link,
                "page_number": page.page_number,
                "image_link": image_link,
                "pdf_width": page.pdf_width,
                "pdf_height": page.pdf_height,
                "image_width": page.image_width,
                "image_height": page.image_height,
                "render_zoom": page.render_zoom,
                "downscale_factor": page.downscale_factor,
                "total_scale": page.total_scale,
                "component_count": page.component_count,
                "max_pixels": self.max_pixels,
            }

            row_id = self.repository.insert_page(
                schematic_id,
                smetadata,
                page.words_bboxes,
            )
            records.append(
                PageRecord(
                    id=row_id,
                    schematic_id=schematic_id,
                    smetadata=smetadata,
                    words_bboxes=page.words_bboxes,
                )
            )

        return ProcessResult(
            schematic_id=schematic_id,
            source_pdf=str(pdf_path),
            file_link=file_link,
            page_count=page_count,
            pages=records,
        )


# ---------------------------------------------------------------------------
# Legacy local-file helpers (backward compatible CLI / dev usage)
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    max_pixels: int = MAX_PIXELS,
    extract_all_text: bool = True,
    components_only: bool = False,
) -> Path:
    """Legacy helper: write components.json + local page images."""
    import json

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_pdf": str(pdf_path.resolve()),
        "max_pixels": max_pixels,
        "pages": [],
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)

    args_list = [
        (str(pdf_path), i, max_pixels, extract_all_text, components_only)
        for i in range(page_count)
    ]
    results = [_process_single_page(args) for args in args_list]

    for page in results:
        image_name = f"page_{page.page_number:03d}.png"
        image_path = pages_dir / image_name
        image_path.write_bytes(page.image_png)

        components = [
            {
                "name": name,
                "kind": entries[0]["kind"],
                "bbox": {k: entries[0][k] for k in ("x0", "y0", "x1", "y1")},
            }
            for name, entries in page.words_bboxes.items()
        ]
        # Expand duplicate names into separate component entries
        expanded_components: list[dict[str, Any]] = []
        for name, entries in page.words_bboxes.items():
            for entry in entries:
                expanded_components.append(
                    {
                        "name": name,
                        "kind": entry["kind"],
                        "bbox": {k: entry[k] for k in ("x0", "y0", "x1", "y1")},
                    }
                )

        manifest["pages"].append(
            {
                "page_number": page.page_number,
                "pdf_width": page.pdf_width,
                "pdf_height": page.pdf_height,
                "image_path": str(Path("pages") / image_name),
                "image_width": page.image_width,
                "image_height": page.image_height,
                "render_zoom": page.render_zoom,
                "downscale_factor": page.downscale_factor,
                "total_scale": page.total_scale,
                "component_count": page.component_count,
                "components": expanded_components,
                "words_bboxes": page.words_bboxes,
            }
        )

    json_path = output_dir / "components.json"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return json_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract schematic components and page images from a PDF."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="LPCX5411x-Schematic_A.pdf",
        help="Input schematic PDF",
    )
    parser.add_argument("-o", "--output-dir", default="output")
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--components-only", action="store_true")
    args = parser.parse_args()

    json_path = process_pdf(
        args.pdf_path,
        args.output_dir,
        max_pixels=args.max_pixels,
        components_only=args.components_only,
    )
    print(f"Saved component data to {json_path}")


if __name__ == "__main__":
    main()
