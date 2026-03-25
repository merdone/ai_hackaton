from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_tools import create_contact_sheet, resolve_dataset_root, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset and optionally render a review sheet.")
    parser.add_argument("--source", required=True, type=Path, help="Path to the dataset root.")
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="Optional output image path for a contact sheet with ground-truth boxes.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of images to include in the contact sheet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for the contact sheet sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset_root(args.source)
    summary = validate_dataset(dataset).to_dict()
    summary["source_root"] = str(dataset.root)
    summary["dataset_yaml_path"] = str(dataset.dataset_yaml_path) if dataset.dataset_yaml_path else None
    summary["notes_path"] = str(dataset.notes_path) if dataset.notes_path else None

    if args.contact_sheet:
        contact_sheet_path = create_contact_sheet(
            dataset=dataset,
            output_path=args.contact_sheet.expanduser().resolve(),
            sample_size=args.sample_size,
            seed=args.seed,
        )
        summary["contact_sheet"] = str(contact_sheet_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
