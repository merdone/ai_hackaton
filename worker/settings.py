from __future__ import annotations

import os
from dataclasses import dataclass, field


def _read_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    return float(raw_value)


def _read_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    return int(raw_value)


def _read_classes(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    classes = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    return classes or default


DEFAULT_TRACKED_CLASSES = (0,)


@dataclass(frozen=True)
class WorkerSettings:
    sorting_aspect_ratio: float = 0.85
    running_speed_threshold: float = 15.0
    walking_speed_threshold: float = 2.0
    write_interval_seconds: float = 3.0
    track_confidence: float = 0.3
    image_size: int = 640
    tracked_classes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_TRACKED_CLASSES)

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        return cls(
            sorting_aspect_ratio=_read_float(
                "AI_HACKATON_SORTING_ASPECT_RATIO",
                cls.sorting_aspect_ratio,
            ),
            running_speed_threshold=_read_float(
                "AI_HACKATON_RUNNING_SPEED_THRESHOLD",
                cls.running_speed_threshold,
            ),
            walking_speed_threshold=_read_float(
                "AI_HACKATON_WALKING_SPEED_THRESHOLD",
                cls.walking_speed_threshold,
            ),
            write_interval_seconds=_read_float(
                "AI_HACKATON_WRITE_INTERVAL_SECONDS",
                cls.write_interval_seconds,
            ),
            track_confidence=_read_float(
                "AI_HACKATON_TRACK_CONFIDENCE",
                cls.track_confidence,
            ),
            image_size=_read_int(
                "AI_HACKATON_IMAGE_SIZE",
                cls.image_size,
            ),
            tracked_classes=_read_classes(
                "AI_HACKATON_TRACKED_CLASSES",
                DEFAULT_TRACKED_CLASSES,
            ),
        )


DEFAULT_WORKER_SETTINGS = WorkerSettings.from_env()
