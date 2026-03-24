import cv2
from ultralytics import YOLO

# тестирование модели, которая определяет скелет

model = YOLO('../models/yolov8s-pose.pt')
model.to('cuda')

cap = cv2.VideoCapture('../data/testvideo4.mp4')

history = {}

print("Запускаем анализ скелетов... Нажми 'q' для выхода.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Трекаем людей и их скелеты
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])
    annotated_frame = results[0].plot()

    # Проверяем, нашел ли он скелеты и выдал ли ID
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        # Получаем координаты точек (x, y) скелета и ID рабочих
        keypoints = results[0].keypoints.xy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # Координаты рамочек для текста

        for i, track_id in enumerate(track_ids):
            kp = keypoints[i]

            # Индексы COCO датасета: 10 - правая кисть, 16 - правая лодыжка (нога)
            right_wrist_y = kp[10][1]
            right_ankle_x = kp[16][0]

            # Если точка не видна (закрыта коробкой), YOLO выдает координаты 0. Пропускаем.
            if right_wrist_y == 0 or right_ankle_x == 0:
                continue

            if track_id in history:
                # Достаем координаты из прошлого кадра
                prev_wrist_y = history[track_id]['wrist_y']
                prev_ankle_x = history[track_id]['ankle_x']

                # Считаем разницу: насколько сдвинулась рука и нога (в пикселях)
                delta_wrist_y = abs(right_wrist_y - prev_wrist_y)
                delta_ankle_x = abs(right_ankle_x - prev_ankle_x)

                # --- ТА САМАЯ ЖЕСТКАЯ ЛОГИКА IF/ELSE ---
                action = "Idle"  # По умолчанию считаем, что рабочий просто стоит

                # Если нога сдвинулась больше чем на 5 пикселей -> Идет
                if delta_ankle_x > 5:
                    action = "Moving"
                # Если нога стоит (<2px), а рука дергается вверх-вниз (>5px) -> Сортирует
                elif delta_wrist_y > 5 and delta_ankle_x < 2:
                    action = "Sorting"

                # Пишем статус прямо над головой человека
                x1, y1 = int(bboxes[i][0]), int(bboxes[i][1])
                cv2.putText(annotated_frame, f"Action: {action}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

            # Запоминаем текущие координаты для следующего кадра
            history[track_id] = {'wrist_y': right_wrist_y, 'ankle_x': right_ankle_x}

    # Ресайз и вывод на экран
    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("YOLOv8 Pose Logic", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()