from __future__ import annotations

import time
from datetime import datetime
from typing import Any, TypedDict

import cv2
import streamlit as st
from ultralytics import YOLO

from app.live.ml import predict_action
from app.settings import AppSettings
from video_pipeline.yolo_final import (
    build_scaled_zones,
    detect_zone_for_bbox_xywh,
    draw_zone_highlight,
    draw_zones,
    load_zones_payload,
)


class ActiveEventState(TypedDict):
    start_time: datetime
    confidences: list[float]
    zone: str
    features: dict[str, float]


def draw_action_label(frame, cx: float, cy: float, w: float, h: float, action: str, confidence: float) -> None:
    x1 = int(cx - (w / 2.0))
    y1 = int(cy - (h / 2.0))
    label_y = max(18, y1 - 50)
    label = f"{action} ({confidence:.2f})"
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def process_live_stream(
    video_placeholder,
    worker_settings,
    app_settings: AppSettings,
    rf_model: Any,
    effective_device: str,
    yolo_confidence: float,
    rf_confidence_min: float,
    live_video_path: str,
    live_zones_path: str,
    db,
):
    """Live-потік із CV-логікою, налаштованою прямо з UI (YOLO/RF confidence)."""
    yolo_model = YOLO(worker_settings.yolo_model_path)
    yolo_model.to(effective_device)

    cap = cv2.VideoCapture(live_video_path)
    if not cap.isOpened():
        st.error(f"Не вдалося відкрити відео: {live_video_path}")
        return

    history: dict[int, dict[str, float]] = {}
    active_events: dict[int, ActiveEventState] = {}

    zones_payload = load_zones_payload(live_zones_path) if worker_settings.yolo_draw_zones else None
    frame_zones = None

    try:
        while cap.isOpened() and st.session_state.get("is_running", False):
            success, frame = cap.read()
            if not success:
                st.warning("Відеопотік завершився.")
                break

            results = yolo_model.track(
                frame,
                persist=True,
                tracker=worker_settings.yolo_tracker,
                classes=list(worker_settings.yolo_classes),
                imgsz=worker_settings.yolo_img_size,
                conf=yolo_confidence,
                verbose=False,
            )
            annotated_frame = results[0].plot()

            if zones_payload is not None:
                if frame_zones is None:
                    frame_height, frame_width = annotated_frame.shape[:2]
                    frame_zones = build_scaled_zones(zones_payload, frame_width, frame_height)
                if frame_zones:
                    draw_zones(annotated_frame, frame_zones)

            now = datetime.now()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                bboxes = results[0].boxes.xywh.cpu().numpy()
                confs = results[0].boxes.conf.cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    cx, cy, w, h = bboxes[i]
                    conf = float(confs[i])
                    current_zone = detect_zone_for_bbox_xywh(cx, cy, frame_zones or [])
                    draw_zone_highlight(annotated_frame, cx, cy, w, h, current_zone)

                    current_aspect_ratio = w / h if h > 0 else 0.0
                    speed_relative = 0.0
                    speed_relative_change = 0.0
                    aspect_ratio_change = 0.0
                    if track_id in history:
                        prev_cx = history[track_id]["center_x"]
                        prev_cy = history[track_id]["center_y"]
                        prev_ar = history[track_id]["aspect_ratio"]
                        prev_speed = history[track_id].get("speed_relative", 0.0)

                        distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                        speed_relative = distance / h if h > 0 else 0.0
                        speed_relative_change = speed_relative - prev_speed
                        aspect_ratio_change = current_aspect_ratio - prev_ar

                    current_features = {
                        "speed_relative": round(float(speed_relative), 4),
                        "speed_relative_change": round(float(speed_relative_change), 4),
                        "aspect_ratio": round(float(current_aspect_ratio), 4),
                        "aspect_ratio_change": round(float(aspect_ratio_change), 4),
                    }

                    current_action, current_action_conf = predict_action(
                        model=rf_model,
                        features=current_features,
                        train_features=app_settings.train_features,
                        fallback_confidence=conf,
                    )
                    draw_action_label(annotated_frame, cx, cy, w, h, current_action, current_action_conf)

                    history[track_id] = {
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "aspect_ratio": float(current_aspect_ratio),
                        "speed_relative": float(speed_relative),
                    }

                    if track_id not in active_events:
                        active_events[track_id] = {
                            "start_time": now,
                            "confidences": [conf],
                            "zone": current_zone,
                            "features": current_features,
                        }
                    else:
                        state = active_events[track_id]
                        state["confidences"].append(conf)
                        state["zone"] = current_zone
                        state["features"] = current_features

                        if (now - state["start_time"]).total_seconds() > 3:
                            avg_conf = sum(state["confidences"]) / max(len(state["confidences"]), 1)
                            predicted_class, predicted_conf = predict_action(
                                model=rf_model,
                                features=state["features"],
                                train_features=app_settings.train_features,
                                fallback_confidence=float(avg_conf),
                            )

                            if float(predicted_conf) >= rf_confidence_min:
                                metadata = dict(state["features"])
                                metadata["detector_avg_confidence"] = round(float(avg_conf), 4)
                                metadata["ui_yolo_confidence"] = round(float(yolo_confidence), 3)
                                metadata["ui_rf_confidence_min"] = round(float(rf_confidence_min), 3)

                                db.log_event(
                                    worker_id=f"Worker_{track_id}",
                                    classification=predicted_class,
                                    zone=state["zone"],
                                    start_time=state["start_time"],
                                    end_time=now,
                                    confidence=float(predicted_conf),
                                    metadata=metadata,
                                )

                            active_events[track_id] = {
                                "start_time": now,
                                "confidences": [conf],
                                "zone": current_zone,
                                "features": state["features"],
                            }

            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", width="stretch")
            time.sleep(0.01)
    finally:
        cap.release()
