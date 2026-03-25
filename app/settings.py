from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
        "aspect_ratio",
        "aspect_ratio_change",
    )


def get_app_settings() -> AppSettings:
    base_dir = Path(__file__).resolve().parent

    def _resolve(path_value: str, fallback_relative: str) -> Path:
        if path_value:
            return Path(path_value).expanduser().resolve()
        return (base_dir / fallback_relative).resolve()

    features_file = _resolve(os.getenv("APP_FEATURES_FILE", ""), "../data/features_temp.json")
    dataset_file = _resolve(os.getenv("APP_DATASET_FILE", ""), "../data/labeled_dataset.csv")
    model_file = _resolve(os.getenv("APP_MODEL_FILE", ""), "../models/rf_v1.pkl")
    preview_video_path = _resolve(os.getenv("APP_PREVIEW_VIDEO_PATH", ""), "../data/preview_with_ids.mp4")
    original_video_path = _resolve(os.getenv("APP_ORIGINAL_VIDEO_PATH", ""), "../data/video3.mkv")

    return AppSettings(
        base_dir=base_dir,
        features_file=features_file,
        dataset_file=dataset_file,
        model_file=model_file,
        preview_video_path=preview_video_path,
        original_video_path=original_video_path,
    )

