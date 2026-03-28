import cv2
from ultralytics import YOLO

# Тестування моделі, яка визначає скелет

model = YOLO('../models/yolov8s-pose.pt')
model.to('cuda')

cap = cv2.VideoCapture('../data/testvideo4.mp4')

history = {}

print("Запускаємо аналіз скелетів... Натисни 'q' для виходу.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Трекаємо людей і їхні скелети
    results = model.track(frame, persist=True, tracker="bytetrack1.yaml", classes=[0])
    annotated_frame = results[0].plot()

    # Перевіряємо, чи знайдено скелети та чи видано ID
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        # Отримуємо координати точок (x, y) скелета та ID працівників
        keypoints = results[0].keypoints.xy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # Координати рамок для тексту

        for i, track_id in enumerate(track_ids):
            kp = keypoints[i]

            # Індекси COCO-датасета: 10 - права кисть, 16 - права кісточка (нога)
            right_wrist_y = kp[10][1]
            right_ankle_x = kp[16][0]

            # Якщо точка не видима (закрита коробкою), YOLO повертає координати 0. Пропускаємо.
            if right_wrist_y == 0 or right_ankle_x == 0:
                continue

            if track_id in history:
                # Дістаємо координати з попереднього кадру
                prev_wrist_y = history[track_id]['wrist_y']
                prev_ankle_x = history[track_id]['ankle_x']

                # Рахуємо різницю: наскільки змістилися рука й нога (у пікселях)
                delta_wrist_y = abs(right_wrist_y - prev_wrist_y)
                delta_ankle_x = abs(right_ankle_x - prev_ankle_x)

                # --- Та сама жорстка логіка IF/ELSE ---
                action = "Idle"  # За замовчуванням вважаємо, що працівник просто стоїть

                # Якщо нога змістилася більш ніж на 5 пікселів -> Йде
                if delta_ankle_x > 5:
                    action = "Moving"
                # Якщо нога стоїть (<2px), а рука рухається вгору-вниз (>5px) -> Вивантажує
                elif delta_wrist_y > 5 and delta_ankle_x < 2:
                    action = "Unloading"

                # Пишемо статус прямо над головою людини
                x1, y1 = int(bboxes[i][0]), int(bboxes[i][1])
                cv2.putText(annotated_frame, f"Action: {action}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

            # Запам'ятовуємо поточні координати для наступного кадру
            history[track_id] = {'wrist_y': right_wrist_y, 'ankle_x': right_ankle_x}

    # Зміна розміру та вивід на екран
    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("YOLOv8 Pose Logic", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()