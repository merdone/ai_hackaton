from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


def project_root() -> Path:
    return PROJECT_ROOT


def data_dir() -> Path:
    raw_path = os.environ.get("AI_HACKATON_DATA_DIR")
    path = Path(raw_path) if raw_path else PROJECT_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def events_db_path() -> Path:
    return data_dir() / "events.db"


def models_dir() -> Path:
    env_path = os.environ.get("AI_HACKATON_MODELS_DIR")
    candidates = [Path(env_path)] if env_path else []
    candidates.extend([PROJECT_ROOT / "Models", PROJECT_ROOT / "models"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else PROJECT_ROOT / "Models"


def default_model_path() -> Path:
    env_path = os.environ.get("AI_HACKATON_MODEL_PATH")
    if env_path:
        return Path(env_path)

    return models_dir() / "best.pt"


def available_video_paths() -> list[Path]:
    root = models_dir()
    if not root.exists():
        return []

    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def default_video_path() -> Path:
    env_path = os.environ.get("AI_HACKATON_VIDEO_PATH")
    if env_path:
        return Path(env_path)

    preferred = models_dir() / "video_3.mkv"
    if preferred.exists():
        return preferred

    videos = available_video_paths()
    if videos:
        return videos[0]

    return preferred
