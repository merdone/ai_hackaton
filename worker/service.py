from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from worker.rules import classify_action, determine_zone, should_write_event
from worker.settings import DEFAULT_WORKER_SETTINGS, WorkerSettings
from worker.storage import ensure_events_table, write_event
from worker.tracking import SimpleTracker


def _build_detections(result) -> list[dict[str, float | int]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    bboxes = boxes.xywh.cpu().numpy()
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(bboxes)

    detections: list[dict[str, float | int]] = []
    for index, bbox in enumerate(bboxes):
        center_x, center_y, width, height = bbox
        if height <= 0:
            continue
        detections.append(
            {
                "result_index": index,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width": float(width),
                "height": float(height),
                "confidence": float(confidences[index]),
            }
        )

    return detections


def _process_video(
    conn: sqlite3.Connection,
    model: YOLO,
    video_path: Path,
    settings: WorkerSettings,
) -> None:
    tracker = SimpleTracker()
    last_written: dict[int, dict[str, float | str]] = {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        result = model(
            frame,
            classes=list(settings.tracked_classes),
            imgsz=settings.image_size,
            conf=settings.track_confidence,
            verbose=False,
        )[0]

        detections = _build_detections(result)
        tracked_detections = tracker.update(detections)
        frame_width = frame.shape[1]
        now_monotonic = time.monotonic()

        for detection in tracked_detections:
            if bool(detection["is_new"]):
                continue

            track_id = int(detection["track_id"])
            speed_relative = math.hypot(
                float(detection["center_x"]) - float(detection["previous_center_x"]),
                float(detection["center_y"]) - float(detection["previous_center_y"]),
            )
            aspect_ratio = float(detection["width"]) / float(detection["height"])
            action = classify_action(aspect_ratio, speed_relative, settings)
            zone = determine_zone(float(detection["center_x"]), frame_width)

            if should_write_event(last_written.get(track_id), action, zone, now_monotonic, settings):
                worker_id = f"track_{track_id}"
                confidence = round(float(detection["confidence"]), 3)
                write_event(conn, worker_id, action, zone, confidence)
                last_written[track_id] = {
                    "action": action,
                    "zone": zone,
                    "written_at": now_monotonic,
                }

    capture.release()


def run_worker(
    model_path: Path,
    video_path: Path,
    db_path: Path,
    settings: WorkerSettings = DEFAULT_WORKER_SETTINGS,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    conn = sqlite3.connect(db_path)
    ensure_events_table(conn)

    print(f"Worker started on device: {device}")
    print(f"Using model: {model_path}")
    print(f"Using video: {video_path}")
    print(f"Writing events to: {db_path}")

    while True:
        _process_video(conn, model, video_path, settings)
        print("Reached end of video. Restarting from the beginning in 1 second.")
        time.sleep(1)
