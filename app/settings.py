from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class AppSettings:
    base_dir: Path
    features_file: Path
    dataset_file: Path
    model_file: Path
    preview_video_path: Path
    original_video_path: Path
    actions: tuple[str, ...] = ("Idle", "Moving", "Sorting")
    train_features: tuple[str, ...] = (
        "speed_relative",
        "speed_relative_change",
        "aspect_ratio",
        "aspect_ratio_change",
    )


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def get_app_settings() -> AppSettings:
    base_dir = Path(__file__).resolve().parent

    features_file = _resolve_path(os.getenv("APP_FEATURES_FILE", "../data/features_temp.json"), PROJECT_ROOT)
    dataset_file = _resolve_path(os.getenv("APP_DATASET_FILE", "../data/labeled_dataset.csv"), PROJECT_ROOT)
    model_file = _resolve_path(os.getenv("APP_MODEL_FILE", "../models/rf_v1.pkl"), PROJECT_ROOT)
    preview_video_path = _resolve_path(os.getenv("APP_PREVIEW_VIDEO_PATH", "../data/preview_with_ids.mp4"), PROJECT_ROOT)
    original_video_path = _resolve_path(os.getenv("APP_ORIGINAL_VIDEO_PATH", "../data/video3.mkv"), PROJECT_ROOT)

    return AppSettings(
        base_dir=base_dir,
        features_file=features_file,
        dataset_file=dataset_file,
        model_file=model_file,
        preview_video_path=preview_video_path,
        original_video_path=original_video_path,
    )
