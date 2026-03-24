import cv2
import math
from ultralytics import YOLO

model = YOLO('../models/best.pt')

model.to('cuda')

cap = cv2.VideoCapture('../data/video3.mkv')

history = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], imgsz=640, conf=0.2)
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.xywh.cpu().numpy()  # Берем формат X_center, Y_center, Width, Height

        for i, track_id in enumerate(track_ids):
            cx, cy, w, h = bboxes[i]

            # 1. Считаем Aspect Ratio
            current_aspect_ratio = w / h

            if track_id in history:
                prev_cx = history[track_id]['center_x']
                prev_cy = history[track_id]['center_y']

                # 2. Считаем пройденное расстояние (в пикселях за 1 кадр)
                distance = math.hypot(cx - prev_cx, cy - prev_cy)

                # Пока оставляем сырую дистанцию, как ты прописал
                speed_relative = distance

                # --- ЛОГИКА ОПРЕДЕЛЕНИЯ СОСТОЯНИЙ ---
                action = "Idle"  # Базовое состояние

                # Если ширина рамки приближается к высоте (человек согнулся/наклонился)
                if current_aspect_ratio > 0.85:
                    action = "Sorting"
                # Если пропорции обычные, смотрим на скорость перемещения центра
                elif speed_relative > 15:  # Больше 15 пикселей за кадр
                    action = "Running"
                elif speed_relative > 2:  # От 2 до 15 пикселей за кадр
                    action = "Walking"

                # 3. Выводим текст и боксы
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)

                # Формируем строку с состоянием и метриками
                text = f"{action} | Spd: {speed_relative:.1f} | AR: {current_aspect_ratio:.2f}"

                # Делаем цвет текста зависимым от действия для наглядности
                color = (0, 255, 0)  # Зеленый для Idle/Walking
                if action == "Running":
                    color = (0, 0, 255)  # Красный
                elif action == "Sorting":
                    color = (255, 165, 0)  # Оранжевый

                cv2.putText(annotated_frame, text, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 10)

            # Обновляем историю
            history[track_id] = {
                'center_x': cx,
                'center_y': cy,
                'aspect_ratio': current_aspect_ratio
            }

    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Features Extractor", resized_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()