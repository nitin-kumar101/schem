"""
Production entry point for schematic processing and component highlighting.

Examples
--------
# Process PDF -> storage + DB (local dev defaults)
python main.py process LPCX5411x-Schematic_A.pdf --schematic-id lpcx5411x

# Highlight multiple components
python main.py highlight lpcx5411x C58 U12 LPC4322JET100

# Legacy local-only workflow (components.json + local images)
python main.py local-process LPCX5411x-Schematic_A.pdf -o output
python main.py local-highlight C58 -d output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import get_settings
from db import build_repository
from schematic_processor import SchematicProcessor, process_pdf
from schematic_retriever import LocalSchematicRetriever, SchematicRetriever
from storage import build_storage


def cmd_process(args: argparse.Namespace) -> None:
    settings = get_settings()
    storage = build_storage(settings)
    repository = build_repository(settings)

    processor = SchematicProcessor(
        storage,
        repository,
        max_pixels=args.max_pixels or settings.max_pixels,
        max_workers=args.workers or settings.max_workers,
        components_only=args.components_only or settings.components_only,
    )

    result = processor.process(
        args.pdf_path,
        args.schematic_id,
        replace_existing=not args.keep_existing,
    )

    print(json.dumps(
        {
            "schematic_id": result.schematic_id,
            "source_pdf": result.source_pdf,
            "file_link": result.file_link,
            "page_count": result.page_count,
            "pages": [
                {
                    "id": page.id,
                    "page_number": page.smetadata["page_number"],
                    "image_link": page.smetadata["image_link"],
                    "component_count": page.smetadata["component_count"],
                }
                for page in result.pages
            ],
        },
        indent=2,
    ))


def cmd_highlight(args: argparse.Namespace) -> None:
    settings = get_settings()
    storage = build_storage(settings)
    repository = build_repository(settings)

    retriever = SchematicRetriever(repository, storage)
    component_names = args.components

    if args.find_only:
        pages = retriever.find_pages_for_components(
            args.schematic_id,
            component_names,
            match_all=args.match_all,
        )
        if not pages:
            print(f"No pages found for components: {', '.join(component_names)}")
            sys.exit(1)
        for page in pages:
            print(
                f"page {page.smetadata.get('page_number')}: "
                f"id={page.id} image={page.smetadata.get('image_link')}"
            )
        return

    results = retriever.highlight_components(
        args.schematic_id,
        component_names,
        match_all=args.match_all,
        mode=args.mode,
        case_sensitive=args.case_sensitive,
    )

    if not results:
        print(f"No matches found for: {', '.join(component_names)}")
        sys.exit(1)

    print(json.dumps([r.to_dict() for r in results], indent=2))


def cmd_local_process(args: argparse.Namespace) -> None:
    settings = get_settings()
    json_path = process_pdf(
        args.pdf_path,
        args.output_dir,
        max_pixels=args.max_pixels or settings.max_pixels,
        components_only=args.components_only,
    )
    print(f"Saved local component data to {json_path}")


def cmd_local_highlight(args: argparse.Namespace) -> None:
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
        sys.exit(1)

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


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Schematic PDF processing and component highlighting pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- process (production) ---
    p_process = sub.add_parser("process", help="Process PDF, upload images, save to DB")
    p_process.add_argument("pdf_path", help="Path to schematic PDF")
    p_process.add_argument(
        "--schematic-id",
        required=True,
        help="Unique id for this schematic (used in storage keys and DB rows)",
    )
    p_process.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help=f"Max width*height per page (default: {settings.max_pixels})",
    )
    p_process.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel page workers (default: min(cpu_count, 8))",
    )
    p_process.add_argument(
        "--components-only",
        action="store_true",
        help="Store only reference-designator-like names",
    )
    p_process.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing rows for this schematic_id before insert",
    )
    p_process.set_defaults(func=cmd_process)

    # --- highlight (production) ---
    p_highlight = sub.add_parser(
        "highlight",
        help="Find pages by component list, highlight, upload result",
    )
    p_highlight.add_argument("schematic_id", help="Schematic id used during processing")
    p_highlight.add_argument(
        "components",
        nargs="+",
        help="Component names to highlight (e.g. C58 U12 LPC4322JET100)",
    )
    p_highlight.add_argument(
        "--match-all",
        action="store_true",
        help="Only return pages that contain ALL listed components",
    )
    p_highlight.add_argument(
        "--mode",
        choices=["exact", "contains", "regex"],
        default="exact",
        help="Component name match mode (default: exact)",
    )
    p_highlight.add_argument("--case-sensitive", action="store_true")
    p_highlight.add_argument(
        "--find-only",
        action="store_true",
        help="Only print matching page rows, do not highlight",
    )
    p_highlight.set_defaults(func=cmd_highlight)

    # --- local-process (dev) ---
    p_local = sub.add_parser(
        "local-process",
        help="Legacy local processing to output/components.json",
    )
    p_local.add_argument("pdf_path", help="Path to schematic PDF")
    p_local.add_argument("-o", "--output-dir", default="output")
    p_local.add_argument("--max-pixels", type=int, default=None)
    p_local.add_argument("--components-only", action="store_true")
    p_local.set_defaults(func=cmd_local_process)

    # --- local-highlight (dev) ---
    p_local_h = sub.add_parser(
        "local-highlight",
        help="Legacy local highlight from output/components.json",
    )
    p_local_h.add_argument("query", help="Component name to search")
    p_local_h.add_argument("-d", "--data-dir", default="output")
    p_local_h.add_argument("-o", "--output-dir", default=None)
    p_local_h.add_argument(
        "--mode",
        choices=["exact", "contains", "regex"],
        default="contains",
    )
    p_local_h.add_argument("--kind", choices=["reference_designator", "text"], default=None)
    p_local_h.add_argument("--page", type=int, default=None)
    p_local_h.add_argument("--case-sensitive", action="store_true")
    p_local_h.add_argument("--display", action="store_true")
    p_local_h.set_defaults(func=cmd_local_highlight)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()