import cv2
from ultralytics import YOLO

# import torch
# print("CUDA доступна:", torch.cuda.is_available())

model = YOLO('yolov8m.pt') # Беремо Medium версію
model.to('cuda')           # Примусово закидаємо її в пам'ять RTX 4050

# 2. Открываем тестовое видео из нашей общей папки
video_path = '../data/testvideo4.mp4'
cap = cv2.VideoCapture(video_path)

print("Запускаем инференс... Нажми 'q' в окне видео для выхода.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Видео закончилось или не найдено.")
        break

    # 3. МАГИЯ YOLO + ByteTrack
    # persist=True - заставляет трекер запоминать ID объектов между кадрами
    # tracker="bytetrack.yaml" - используем продвинутый алгоритм трекинга
    # classes=[0] - просим YOLO искать ТОЛЬКО людей (класс 0 в датасете COCO)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])

    # 4. Рисуем результаты прямо на кадре (боксы, вероятности и ID)
    annotated_frame = results[0].plot()
    resized_frame = cv2.resize(annotated_frame, (1280, 720))
    # 5. Показываем видео на экране (локальный дебаг)
    cv2.imshow("YOLOv8 + ByteTrack MVP", resized_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()