from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "images"

videos = sorted(
    path for path in MODELS_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in {".avi", ".mkv", ".mov", ".mp4"}
) if MODELS_DIR.exists() else []

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

saved_count = 0
frames_to_skip = 50

if not videos:
    raise FileNotFoundError(f"No video files were found in {MODELS_DIR}")

print(f"Extracting frames from {len(videos)} video file(s)...")

for video_path in videos:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frames_to_skip == 0:
            filename = OUTPUT_DIR / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

print(f"Done. Saved {saved_count} frame(s) into {OUTPUT_DIR}")
