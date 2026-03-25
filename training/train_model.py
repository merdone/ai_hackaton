from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import default_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector on the prepared worker dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "training" / "worker_data" / "dataset.yaml",
        help="Path to the prepared dataset.yaml file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to the starting YOLO weights.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "detect",
        help="Output folder for training runs.",
    )
    parser.add_argument("--name", default="worker_train", help="Run name inside the project folder.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example 'cpu' or '0'. Defaults to CUDA if available.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Data-loader worker count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    project_path = args.project.expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(project_path),
        name=args.name,
        device=device,
        workers=args.workers,
    )

    summary = {
        "data": str(data_path),
        "model": str(model_path),
        "device": device,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "save_dir": str(Path(results.save_dir).resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
