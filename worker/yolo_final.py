import json
import math
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO

from settings import WorkerSettings, get_settings


class RenderZone(TypedDict):
    name: str
    polygon: np.ndarray


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


def load_zones_payload(path: str) -> dict | None:
    zones_path = Path(path)
    if not zones_path.exists():
        return None

    with zones_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload.get("zones"), list):
        return None
    return payload


def build_scaled_zones(payload: dict, frame_width: int, frame_height: int) -> list[RenderZone]:
    source_width = int(payload.get("image_width") or frame_width)
    source_height = int(payload.get("image_height") or frame_height)

    if source_width <= 0 or source_height <= 0:
        return []

    x_scale = frame_width / source_width
    y_scale = frame_height / source_height

    scaled: list[RenderZone] = []
    for zone in payload.get("zones", []):
        points = zone.get("points", [])
        if len(points) < 3:
            continue

        polygon = np.array(
            [[int(point[0] * x_scale), int(point[1] * y_scale)] for point in points],
            dtype=np.int32,
        )
        if polygon.shape[0] < 3:
            continue

        scaled.append({"name": str(zone.get("name", "Zone")), "polygon": polygon})

    return scaled


def draw_zones(frame: np.ndarray, zones: list[RenderZone]) -> None:
    for zone in zones:
        polygon = zone["polygon"]
        name = zone["name"]

        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 220, 0), thickness=2)

        moments = cv2.moments(polygon)
        if moments["m00"] != 0:
            label_x = int(moments["m10"] / moments["m00"])
            label_y = int(moments["m01"] / moments["m00"])
        else:
            label_x, label_y = int(polygon[0][0]), int(polygon[0][1])

        cv2.putText(
            frame,
            name,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def detect_zone_for_bbox_xywh(cx: float, cy: float, zones: list[RenderZone]) -> str:
    if not zones:
        return "None"

    point = (float(cx), float(cy))
    for zone in zones:
        polygon = zone["polygon"].astype(np.float32)
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return zone["name"]
    return "None"


def draw_zone_highlight(frame: np.ndarray, cx: float, cy: float, w: float, h: float, zone_name: str) -> None:
    if zone_name == "None":
        return

    x1 = int(cx - (w / 2.0))
    y1 = int(cy - (h / 2.0))
    x2 = int(cx + (w / 2.0))
    y2 = int(cy + (h / 2.0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    label = f"Zone: {zone_name}"
    label_y = max(20, y1 + 30)
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        3,
        cv2.LINE_AA,
    )


def extract_features(settings: WorkerSettings) -> list[dict[str, float | int | str]]:
    model = YOLO(settings.yolo_model_path)
    model.to(settings.yolo_device)

    cap = cv2.VideoCapture(settings.yolo_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {settings.yolo_video_path}")

    out, _ = create_video_writer(cap, settings)
    history: dict[int, dict[str, float]] = {}
    extracted_features: list[dict[str, float | int | str]] = []

    zones_payload = load_zones_payload(settings.yolo_zones_path) if settings.yolo_draw_zones else None
    frame_zones: list[RenderZone] | None = None

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

            if zones_payload is not None:
                if frame_zones is None:
                    frame_height, frame_width = annotated_frame.shape[:2]
                    frame_zones = build_scaled_zones(zones_payload, frame_width, frame_height)
                if frame_zones:
                    draw_zones(annotated_frame, frame_zones)

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                bboxes = results[0].boxes.xywh.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    cx, cy, w, h = bboxes[i]
                    current_aspect_ratio = w / h if h > 0 else 0.0
                    current_zone = detect_zone_for_bbox_xywh(cx, cy, frame_zones or [])
                    draw_zone_highlight(annotated_frame, cx, cy, w, h, current_zone)

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
                                "zone_intersection": current_zone,
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


def save_features(features: list[dict[str, float | int | str]], output_path: str) -> None:
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
