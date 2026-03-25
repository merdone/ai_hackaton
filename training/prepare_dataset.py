from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_tools import resolve_dataset_root, split_dataset, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible train/val split for YOLO training.")
    parser.add_argument("--source", required=True, type=Path, help="Path to the source dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "training" / "worker_data",
        help="Destination folder for the prepared dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images to place into the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate the output folder if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset_root(args.source)
    summary = validate_dataset(dataset)
    if summary.missing_labels_count or summary.extra_labels_count or summary.malformed_count:
        raise RuntimeError(
            "Dataset has structural issues. Run inspect_dataset.py first and fix them before splitting."
        )

    result = split_dataset(
        dataset=dataset,
        output_root=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    result["validation_summary"] = summary.to_dict()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
