from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkerSettings:
    sorting_aspect_ratio: float = 0.85
    running_speed_threshold: float = 15.0
    walking_speed_threshold: float = 2.0
    write_interval_seconds: float = 3.0
    track_confidence: float = 0.2
    image_size: int = 640
    tracked_classes: tuple[int, ...] = field(default_factory=lambda: (0,))


DEFAULT_WORKER_SETTINGS = WorkerSettings()
