import cv2
import math
import torch
from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import default_model_path, default_video_path
from worker.tracking import SimpleTracker

MODEL_PATH = default_model_path()
VIDEO_PATH = default_video_path()

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

model = YOLO(str(MODEL_PATH))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Using model: {MODEL_PATH}")
model.to(device)

print(f"Using video: {VIDEO_PATH}")
tracker = SimpleTracker()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    result = model(
        frame,
        classes=[0],
        imgsz=640,
        conf=0.2,
        verbose=False,
    )[0]

    annotated_frame = result.plot()
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xywh.cpu().numpy()
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [1.0] * len(bboxes)

        detections = []
        for index, bbox in enumerate(bboxes):
            cx, cy, w, h = bbox
            if h <= 0:
                continue
            detections.append(
                {
                    "result_index": index,
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "width": float(w),
                    "height": float(h),
                    "confidence": float(confidences[index]),
                }
            )

        tracked_detections = tracker.update(detections)

        for detection in tracked_detections:
            cx = float(detection["center_x"])
            cy = float(detection["center_y"])
            w = float(detection["width"])
            h = float(detection["height"])
            current_aspect_ratio = w / h
            speed_relative = math.hypot(
                cx - float(detection["previous_center_x"]),
                cy - float(detection["previous_center_y"]),
            )

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            track_id = int(detection["track_id"])
            text = f"ID {track_id} | Spd: {speed_relative:.2f} | AR: {current_aspect_ratio:.2f}"
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 255),
                4,
            )

    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Box Metrics Demo", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
