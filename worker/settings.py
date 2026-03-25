from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSettings:
    # Live worker event generation settings
    db_path: str
    sleep_seconds: float
    worker_ids: tuple[str, ...]
    actions: tuple[str, ...]
    zones: tuple[str, ...]
    confidence_min: float
    confidence_max: float
    timestamp_format: str

    # YOLO feature extraction settings
    yolo_model_path: str
    yolo_video_path: str
    yolo_preview_output_path: str
    yolo_features_output_path: str
    yolo_tracker: str
    yolo_classes: tuple[int, ...]
    yolo_img_size: int
    yolo_confidence: float
    yolo_device: str
    yolo_preview_width: int
    yolo_preview_height: int
    yolo_window_name: str
    yolo_show_window: bool
    yolo_fourcc: str


def _parse_csv(value: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or fallback


def _parse_csv_int(value: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return fallback
    return tuple(int(part) for part in parts)


def _parse_bool(value: str, fallback: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return fallback


def get_settings() -> WorkerSettings:
    """Load worker settings with defaults that preserve current behavior."""
    project_root = Path(__file__).resolve().parents[1]

    defaults = WorkerSettings(
        db_path="/app/data/events.db",
        sleep_seconds=2.0,
        worker_ids=("ID_1", "ID_2", "ID_3"),
        actions=("Moving", "Sorting", "Idle"),
        zones=("Zone_A", "Zone_B"),
        confidence_min=0.65,
        confidence_max=0.99,
        timestamp_format="%Y-%m-%d %H:%M:%S",
        yolo_model_path=str(project_root / "models" / "best.pt"),
        yolo_video_path=str(project_root / "data" / "video2.mkv"),
        yolo_preview_output_path=str(project_root / "data" / "preview_with_ids.mp4"),
        yolo_features_output_path=str(project_root / "data" / "features_temp.json"),
        yolo_tracker="bytetrack.yaml",
        yolo_classes=(0,),
        yolo_img_size=640,
        yolo_confidence=0.2,
        yolo_device="cuda",
        yolo_preview_width=1280,
        yolo_preview_height=720,
        yolo_window_name="Features Extractor",
        yolo_show_window=True,
        yolo_fourcc="avc1",
    )

    return WorkerSettings(
        db_path=os.getenv("WORKER_DB_PATH", defaults.db_path),
        sleep_seconds=float(os.getenv("WORKER_SLEEP_SECONDS", str(defaults.sleep_seconds))),
        worker_ids=_parse_csv(os.getenv("WORKER_IDS", ",".join(defaults.worker_ids)), defaults.worker_ids),
        actions=_parse_csv(os.getenv("WORKER_ACTIONS", ",".join(defaults.actions)), defaults.actions),
        zones=_parse_csv(os.getenv("WORKER_ZONES", ",".join(defaults.zones)), defaults.zones),
        confidence_min=float(os.getenv("WORKER_CONFIDENCE_MIN", str(defaults.confidence_min))),
        confidence_max=float(os.getenv("WORKER_CONFIDENCE_MAX", str(defaults.confidence_max))),
        timestamp_format=os.getenv("WORKER_TIMESTAMP_FORMAT", defaults.timestamp_format),
        yolo_model_path=os.getenv("WORKER_YOLO_MODEL_PATH", defaults.yolo_model_path),
        yolo_video_path=os.getenv("WORKER_YOLO_VIDEO_PATH", defaults.yolo_video_path),
        yolo_preview_output_path=os.getenv("WORKER_YOLO_PREVIEW_PATH", defaults.yolo_preview_output_path),
        yolo_features_output_path=os.getenv("WORKER_YOLO_FEATURES_PATH", defaults.yolo_features_output_path),
        yolo_tracker=os.getenv("WORKER_YOLO_TRACKER", defaults.yolo_tracker),
        yolo_classes=_parse_csv_int(os.getenv("WORKER_YOLO_CLASSES", ",".join(str(i) for i in defaults.yolo_classes)),
                                    defaults.yolo_classes),
        yolo_img_size=int(os.getenv("WORKER_YOLO_IMGSZ", str(defaults.yolo_img_size))),
        yolo_confidence=float(os.getenv("WORKER_YOLO_CONF", str(defaults.yolo_confidence))),
        yolo_device=os.getenv("WORKER_YOLO_DEVICE", defaults.yolo_device),
        yolo_preview_width=int(os.getenv("WORKER_YOLO_PREVIEW_WIDTH", str(defaults.yolo_preview_width))),
        yolo_preview_height=int(os.getenv("WORKER_YOLO_PREVIEW_HEIGHT", str(defaults.yolo_preview_height))),
        yolo_window_name=os.getenv("WORKER_YOLO_WINDOW_NAME", defaults.yolo_window_name),
        yolo_show_window=_parse_bool(os.getenv("WORKER_YOLO_SHOW_WINDOW", str(defaults.yolo_show_window)),
                                     defaults.yolo_show_window),
        yolo_fourcc=os.getenv("WORKER_YOLO_FOURCC", defaults.yolo_fourcc),
    )
