from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import default_model_path
from training.dataset_tools import read_image_unicode, read_label_lines, resolve_split_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the worker detector on a labeled YOLO dataset.")
    parser.add_argument("--source", required=True, type=Path, help="Path to the dataset root.")
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to the YOLO model weights.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split name to evaluate. Falls back to flat images/labels if the split does not exist.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument(
        "--thresholds",
        default="0.2,0.25,0.3,0.35,0.4",
        help="Comma-separated confidence thresholds to evaluate.",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match.")
    return parser.parse_args()


def yolo_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    center_x, center_y, box_width, box_height = box
    return np.array(
        [
            (center_x - box_width / 2) * width,
            (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width,
            (center_y + box_height / 2) * height,
        ],
        dtype=float,
    )


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {source_root}")
    if not (source_root / "classes.txt").is_file():
        raise FileNotFoundError(f"classes.txt not found in dataset root: {source_root}")

    images_dir, labels_dir = resolve_split_dirs(source_root, args.split)
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    if not thresholds:
        raise RuntimeError("No valid thresholds were provided.")

    model = YOLO(str(args.model.expanduser().resolve()))
    cached_predictions: dict[str, dict[str, object]] = {}
    small_area_limit = 0.01

    for image_path in image_paths:
        image = read_image_unicode(image_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        image_height, image_width = image.shape[:2]
        ground_truth_boxes: list[np.ndarray] = []
        for line in read_label_lines(labels_dir / f"{image_path.stem}.txt"):
            _, center_x, center_y, box_width, box_height = line.split()
            ground_truth_boxes.append(
                yolo_to_xyxy(
                    (
                        float(center_x),
                        float(center_y),
                        float(box_width),
                        float(box_height),
                    ),
                    image_width,
                    image_height,
                )
            )

        result = model.predict(
            image,
            conf=0.05,
            imgsz=args.imgsz,
            verbose=False,
            classes=[0],
        )[0]
        predictions: list[tuple[np.ndarray, float]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                predictions.append((box.astype(float), float(confidence)))

        cached_predictions[image_path.name] = {
            "ground_truth": ground_truth_boxes,
            "predictions": predictions,
            "width": image_width,
            "height": image_height,
        }

    rows = []
    for threshold in thresholds:
        total_ground_truth = 0
        total_predictions = 0
        matched_ground_truth = 0
        matched_predictions = 0
        matched_confidences: list[float] = []
        small_ground_truth = 0
        matched_small_ground_truth = 0

        for item in cached_predictions.values():
            ground_truth = item["ground_truth"]
            predictions = [
                prediction
                for prediction in item["predictions"]
                if prediction[1] >= threshold
            ]
            image_width = int(item["width"])
            image_height = int(item["height"])

            total_ground_truth += len(ground_truth)
            total_predictions += len(predictions)
            used_predictions: set[int] = set()

            for gt_box in ground_truth:
                gt_area_ratio = (
                    max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
                ) / float(image_width * image_height)
                if gt_area_ratio < small_area_limit:
                    small_ground_truth += 1

                best_index = None
                best_iou = 0.0
                best_confidence = 0.0
                for prediction_index, (pred_box, confidence) in enumerate(predictions):
                    if prediction_index in used_predictions:
                        continue
                    overlap = iou(gt_box, pred_box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_index = prediction_index
                        best_confidence = confidence

                if best_index is not None and best_iou >= args.iou:
                    used_predictions.add(best_index)
                    matched_ground_truth += 1
                    matched_predictions += 1
                    matched_confidences.append(best_confidence)
                    if gt_area_ratio < small_area_limit:
                        matched_small_ground_truth += 1

        precision = matched_predictions / total_predictions if total_predictions else 0.0
        recall = matched_ground_truth / total_ground_truth if total_ground_truth else 0.0
        rows.append(
            {
                "threshold": threshold,
                "precision_proxy_iou50": round(precision, 4),
                "recall_proxy_iou50": round(recall, 4),
                "avg_conf_matched_predictions": round(float(np.mean(matched_confidences)), 4)
                if matched_confidences
                else None,
                "median_conf_matched_predictions": round(float(np.median(matched_confidences)), 4)
                if matched_confidences
                else None,
                "pred_boxes": total_predictions,
                "gt_boxes": total_ground_truth,
                "small_gt_recall_proxy": round(matched_small_ground_truth / small_ground_truth, 4)
                if small_ground_truth
                else None,
            }
        )

    output = {
        "source_root": str(source_root),
        "evaluated_images_dir": str(images_dir),
        "evaluated_labels_dir": str(labels_dir),
        "model_path": str(args.model.expanduser().resolve()),
        "imgsz": args.imgsz,
        "iou_threshold": args.iou,
        "rows": rows,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
