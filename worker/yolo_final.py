import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

from settings import WorkerSettings, get_settings


def create_video_writer(cap: cv2.VideoCapture, settings: WorkerSettings) -> tuple[cv2.VideoWriter, int]:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*settings.yolo_fourcc)

    output_path = Path(settings.yolo_preview_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return writer, fps


def extract_features(settings: WorkerSettings) -> list[dict[str, float | int]]:
    model = YOLO(settings.yolo_model_path)
    model.to(settings.yolo_device)

    cap = cv2.VideoCapture(settings.yolo_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {settings.yolo_video_path}")

    out, _ = create_video_writer(cap, settings)
    history: dict[int, dict[str, float]] = {}
    extracted_features: list[dict[str, float | int]] = []

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=settings.yolo_tracker,
                classes=list(settings.yolo_classes),
                imgsz=settings.yolo_img_size,
                conf=settings.yolo_confidence,
            )
            annotated_frame = results[0].plot()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                bboxes = results[0].boxes.xywh.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    cx, cy, w, h = bboxes[i]
                    current_aspect_ratio = w / h if h > 0 else 0.0

                    if track_id in history:
                        prev_cx = history[track_id]["center_x"]
                        prev_cy = history[track_id]["center_y"]
                        prev_ar = history[track_id]["aspect_ratio"]

                        distance = math.hypot(cx - prev_cx, cy - prev_cy)
                        speed_relative = distance / h if h > 0 else 0.0
                        aspect_ratio_change = current_aspect_ratio - prev_ar

                        extracted_features.append(
                            {
                                "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                "track_id": int(track_id),
                                "speed_relative": float(round(speed_relative, 4)),
                                "aspect_ratio": float(round(current_aspect_ratio, 4)),
                                "aspect_ratio_change": float(round(aspect_ratio_change, 4)),
                            }
                        )

                    history[track_id] = {
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "aspect_ratio": float(current_aspect_ratio),
                    }

            out.write(annotated_frame)

            if settings.yolo_show_window:
                resized_frame = cv2.resize(
                    annotated_frame,
                    (settings.yolo_preview_width, settings.yolo_preview_height),
                )
                cv2.imshow(settings.yolo_window_name, resized_frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    return extracted_features


def save_features(features: list[dict[str, float | int]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(features, f, indent=4)


def main() -> None:
    settings = get_settings()
    features = extract_features(settings)
    save_features(features, settings.yolo_features_output_path)


if __name__ == "__main__":
    main()
