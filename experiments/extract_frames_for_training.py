import cv2
import os

# Ваші відеофайли (замініть назви на реальні)
videos = ['../data/video1.mkv', '../data/video2.mkv', '../data/video3.mkv']

# Папка, де зберігатимуться зображення для розмітки
output_dir = '../data/dataset/images'
os.makedirs(output_dir, exist_ok=True)

saved_count = 0
frames_to_skip = 50  # Якщо відео 30 FPS, то 150 кадрів = 1 кадр кожні 5 секунд

print("Починаємо нарізку відео. Зачекай...")

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не вдалося відкрити {video_path}")
        continue

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Зберігаємо лише кожен 150-й кадр
        if frame_count % frames_to_skip == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

print(f"Готово! Нарізано {saved_count} кадрів і збережено в {output_dir}")
