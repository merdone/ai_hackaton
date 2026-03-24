import cv2
import json
import math
from ultralytics import YOLO

model = YOLO('../models/best.pt')

model.to('cuda')

cap = cv2.VideoCapture('../data/video2.mkv')

# --- ДОБАВЛЯЕМ ЭТО ---
# Получаем параметры оригинального видео
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настраиваем кодек и файл для сохранения
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Стандартный кодек для mp4
out = cv2.VideoWriter('../data/preview_with_ids.mp4', fourcc, fps, (width, height))
history = {}

extracted_features = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], imgsz=640, conf=0.2)
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.xywh.cpu().numpy()

        for i, track_id in enumerate(track_ids):
            cx, cy, w, h = bboxes[i]

            current_aspect_ratio = w / h

            if track_id in history:
                prev_cx = history[track_id]['center_x']
                prev_cy = history[track_id]['center_y']
                prev_ar = history[track_id]['aspect_ratio']

                distance = math.hypot(cx - prev_cx, cy - prev_cy)

                if h > 0:
                    speed_relative = distance / h
                else:
                    speed_relative = 0

                aspect_ratio_change = current_aspect_ratio - prev_ar

                extracted_features.append({
                    "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "track_id": int(track_id),
                    "speed_relative": float(round(speed_relative, 4)),
                    "aspect_ratio": float(round(current_aspect_ratio, 4)),
                    "aspect_ratio_change": float(round(aspect_ratio_change, 4))
                })

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)

                # text = f"Spd: {speed_relative:.1f} | AR: {current_aspect_ratio:.2f}"
                #
                # color = (0, 255, 0)
                #
                # cv2.putText(annotated_frame, text, (x1, y1 - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)

            history[track_id] = {
                'center_x': cx,
                'center_y': cy,
                'aspect_ratio': current_aspect_ratio
            }
    out.write(annotated_frame)
    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Features Extractor", resized_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
out.release()

with open('../data/features_temp.json', 'w') as f:
    json.dump(extracted_features, f, indent=4)