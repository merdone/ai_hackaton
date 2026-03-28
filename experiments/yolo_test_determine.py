import cv2
import math
from ultralytics import YOLO

# Робоча версія: біжить/йде/стоїть

model = YOLO('../models/yolov8m.pt')
model.to('cuda')

cap = cv2.VideoCapture('../data/run.mp4')

history = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, tracker="bytetrack1.yaml", classes=[0])
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.xywh.cpu().numpy()  # Беремо формат X_center, Y_center, Width, Height

        for i, track_id in enumerate(track_ids):
            cx, cy, w, h = bboxes[i]
            # 1. Рахуємо Aspect Ratio (відношення ширини до висоти)
            # Якщо людина стоїть прямо, h > w (співвідношення < 1).
            # Якщо нахилилася за коробкою, w може стати більшою за h (співвідношення > 1).
            current_aspect_ratio = w / h

            if track_id in history:
                prev_cx = history[track_id]['center_x']
                prev_cy = history[track_id]['center_y']

                # 2. Рахуємо пройдену відстань за теоремою Піфагора
                distance = math.hypot(cx - prev_cx, cy - prev_cy)

                # 3. Нормалізуємо швидкість (speed_relative з вашого ТЗ)
                # Ділимо на висоту рамки (h), щоб швидкість не залежала від того,
                # близько людина до камери чи далеко.
                # speed_relative = distance / h
                speed_relative = distance
                # Виводимо цифри прямо на відео для дебагу (округлюємо для зручності)
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)

                text = f"Spd: {speed_relative:.2f} | AR: {current_aspect_ratio:.2f}"
                cv2.putText(annotated_frame, text, (x1, y1 - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 10)

            # Оновлюємо історію для наступного кадру
            history[track_id] = {
                'center_x': cx,
                'center_y': cy,
                'aspect_ratio': current_aspect_ratio
            }

    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Features Extractor", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()