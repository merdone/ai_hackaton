from __future__ import annotations

from pathlib import Path
import math
import sqlite3
import sys
import time
from datetime import datetime

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import default_model_path, default_video_path, events_db_path
from common.simple_tracker import SimpleTracker

SORTING_ASPECT_RATIO = 0.85
RUNNING_SPEED_THRESHOLD = 15.0
WALKING_SPEED_THRESHOLD = 2.0
WRITE_INTERVAL_SECONDS = 3.0
TRACK_CONFIDENCE = 0.2
TRACKED_CLASSES = [0]


def ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            timestamp TEXT,
            worker_id TEXT,
            action TEXT,
            zone TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()


def classify_action(aspect_ratio: float, speed_relative: float) -> str:
    if aspect_ratio > SORTING_ASPECT_RATIO:
        return "Sorting"
    if speed_relative > RUNNING_SPEED_THRESHOLD:
        return "Running"
    if speed_relative > WALKING_SPEED_THRESHOLD:
        return "Walking"
    return "Idle"


def determine_zone(center_x: float, frame_width: int) -> str:
    return "Zone_A" if center_x < frame_width / 2 else "Zone_B"


def should_write_event(
    last_event: dict[str, float | str] | None,
    action: str,
    zone: str,
    now_monotonic: float,
) -> bool:
    if last_event is None:
        return True
    if last_event["action"] != action or last_event["zone"] != zone:
        return True
    return now_monotonic - float(last_event["written_at"]) >= WRITE_INTERVAL_SECONDS


def write_event(conn: sqlite3.Connection, worker_id: str, action: str, zone: str, confidence: float) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (timestamp, worker_id, action, zone, confidence),
    )
    conn.commit()
    print(f"Stored event: {timestamp} | {worker_id} | {action} | {zone} | conf={confidence:.3f}")


def process_video(conn: sqlite3.Connection, model_path: Path, video_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    tracker = SimpleTracker()
    last_written: dict[int, dict[str, float | str]] = {}

    print(f"Worker started on device: {device}")
    print(f"Using model: {model_path}")
    print(f"Using video: {video_path}")

    while True:
        results = model(
            str(video_path),
            stream=True,
            classes=TRACKED_CLASSES,
            imgsz=640,
            conf=TRACK_CONFIDENCE,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                tracker.update([])
                continue

            bboxes = boxes.xywh.cpu().numpy()
            confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(bboxes)
            frame_width = result.orig_shape[1]

            detections = []
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

            tracked_detections = tracker.update(detections)
            now_monotonic = time.monotonic()

            for detection in tracked_detections:
                track_id = int(detection["track_id"])
                speed_relative = math.hypot(
                    float(detection["center_x"]) - float(detection["previous_center_x"]),
                    float(detection["center_y"]) - float(detection["previous_center_y"]),
                )
                aspect_ratio = float(detection["width"]) / float(detection["height"])
                action = classify_action(aspect_ratio, speed_relative)
                zone = determine_zone(float(detection["center_x"]), frame_width)

                if not bool(detection["is_new"]) and should_write_event(last_written.get(track_id), action, zone, now_monotonic):
                    worker_id = f"track_{track_id}"
                    confidence = round(float(detection["confidence"]), 3)
                    write_event(conn, worker_id, action, zone, confidence)
                    last_written[track_id] = {
                        "action": action,
                        "zone": zone,
                        "written_at": now_monotonic,
                    }

        print("Reached end of video. Restarting from the beginning in 1 second.")
        time.sleep(1)


def main() -> None:
    model_path = default_model_path()
    video_path = default_video_path()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    conn = sqlite3.connect(events_db_path())
    ensure_events_table(conn)
    process_video(conn, model_path, video_path)


if __name__ == "__main__":
    main()
