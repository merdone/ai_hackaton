import cv2
import torch
from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import default_pose_model_path, default_video_path
from worker.tracking import SimpleTracker

POSE_MODEL_PATH = default_pose_model_path()
VIDEO_PATH = default_video_path()

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

model = YOLO(str(POSE_MODEL_PATH))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Using pose model: {POSE_MODEL_PATH}")
print(f"Using video: {VIDEO_PATH}")
model.to(device)

tracker = SimpleTracker()
history = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    result = model(
        frame,
        classes=[0],
        verbose=False,
    )[0]
    annotated_frame = result.plot()

    if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
        keypoints = result.keypoints.xy.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()
        bbox_centers = result.boxes.xywh.cpu().numpy()
        confidences = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [1.0] * len(bboxes)

        detections = []
        for index, bbox in enumerate(bbox_centers):
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

        for detection in tracked_detections:
            result_index = int(detection["result_index"])
            kp = keypoints[result_index]
            right_wrist_y = kp[10][1]
            right_ankle_x = kp[16][0]

            if right_wrist_y == 0 or right_ankle_x == 0:
                continue

            track_id = int(detection["track_id"])
            action = "Idle"
            if track_id in history:
                delta_wrist_y = abs(right_wrist_y - history[track_id]["wrist_y"])
                delta_ankle_x = abs(right_ankle_x - history[track_id]["ankle_x"])

                if delta_ankle_x > 5:
                    action = "Moving"
                elif delta_wrist_y > 5 and delta_ankle_x < 2:
                    action = "Sorting"

            history[track_id] = {"wrist_y": right_wrist_y, "ankle_x": right_ankle_x}

            x1, y1 = int(bboxes[result_index][0]), int(bboxes[result_index][1])
            cv2.putText(
                annotated_frame,
                f"ID {track_id} | Action: {action}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                3,
            )

    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Pose Actions Demo", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
