import cv2
import os

# Твои видеофайлы (поменяй названия на реальные)
videos = ['../data/video1.mkv', '../data/video2.mkv', '../data/video3.mkv']

# Папка, где будут лежать картинки для разметки
output_dir = '../data/dataset/images'
os.makedirs(output_dir, exist_ok=True)

saved_count = 0
frames_to_skip = 50  # Если видео 30 FPS, то 150 кадров = 1 кадр каждые 5 секунд

print("Начинаем нарезку видео. Жди...")

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не смог открыть {video_path}")
        continue

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Сохраняем только каждый 150-й кадр
        if frame_count % frames_to_skip == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

print(f"Готово! Нарезано {saved_count} кадров и сохранено в {output_dir}")