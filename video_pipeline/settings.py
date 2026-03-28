from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class WorkerSettings:
    # YOLO feature extraction settings
    yolo_model_path: str
    yolo_video_path: str
    live_analysis_video_path: str
    live_analysis_zones_path: str
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
    zone_annotator_window_name: str
    yolo_show_window: bool
    yolo_fourcc: str
    yolo_zones_path: str
    yolo_draw_zones: bool


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


def _resolve_path(path_value: str, base_dir: Path) -> str:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _get_env(*keys: str, default: str) -> str:
    """Return the first non-empty env value from keys, else default."""
    for key in keys:
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return default


def get_settings() -> WorkerSettings:
    """Load worker settings with new YOLO_* keys and legacy WORKER_* fallbacks."""
    project_root = Path(__file__).resolve().parents[1]

    defaults = WorkerSettings(
        yolo_model_path=str(project_root / "models" / "best.pt"),
        yolo_video_path=str(project_root / "data" / "video2.mkv"),
        live_analysis_video_path=str(project_root / "data" / "video2.mkv"),
        live_analysis_zones_path=str(project_root / "data" / "output" / "zones_video_2.json"),
        yolo_preview_output_path=str(project_root / "data" / "preview_with_ids.mp4"),
        yolo_features_output_path=str(project_root / "data" / "features_temp_video_3.json"),
        yolo_tracker="bytetrack.yaml",
        yolo_classes=(0,),
        yolo_img_size=640,
        yolo_confidence=0.2,
        yolo_device="auto",
        yolo_preview_width=1280,
        yolo_preview_height=720,
        yolo_window_name="Features Extractor",
        zone_annotator_window_name="Zone Annotator",
        yolo_show_window=True,
        yolo_fourcc="avc1",
        yolo_zones_path=str(project_root / "data" / "output" / "zones_video_2.json"),
        yolo_draw_zones=True,
    )

    return WorkerSettings(
        yolo_model_path=_resolve_path(_get_env("YOLO_MODEL_PATH", default=defaults.yolo_model_path), project_root),
        yolo_video_path=_resolve_path(_get_env("YOLO_VIDEO_PATH", default=defaults.yolo_video_path), project_root),
        live_analysis_video_path=_resolve_path(
            _get_env("LIVE_ANALYSIS_VIDEO_PATH", default=defaults.live_analysis_video_path),
            project_root,
        ),
        live_analysis_zones_path=_resolve_path(
            _get_env("LIVE_ANALYSIS_ZONES_PATH", default=defaults.live_analysis_zones_path),
            project_root,
        ),
        yolo_preview_output_path=_resolve_path(
            _get_env("YOLO_PREVIEW_SAVE_PATH", default=defaults.yolo_preview_output_path),
            project_root,
        ),
        yolo_features_output_path=_resolve_path(
            _get_env("YOLO_FEATURES_SAVE_PATH", default=defaults.yolo_features_output_path),
            project_root,
        ),
        yolo_tracker=_get_env("YOLO_TRACKER", default=defaults.yolo_tracker),
        yolo_classes=_parse_csv_int(
            _get_env("YOLO_CLASSES", default=",".join(str(i) for i in defaults.yolo_classes)),
            defaults.yolo_classes,
        ),
        yolo_img_size=int(_get_env("YOLO_IMGSZ", default=str(defaults.yolo_img_size))),
        yolo_confidence=float(_get_env("YOLO_CONF", default=str(defaults.yolo_confidence))),
        yolo_device=_get_env("YOLO_DEVICE", default=defaults.yolo_device),
        yolo_preview_width=int(
            _get_env("YOLO_PREVIEW_WIDTH", default=str(defaults.yolo_preview_width))
        ),
        yolo_preview_height=int(
            _get_env("YOLO_PREVIEW_HEIGHT", default=str(defaults.yolo_preview_height))
        ),
        yolo_window_name=_get_env("YOLO_WINDOW_NAME", default=defaults.yolo_window_name),
        zone_annotator_window_name=_get_env(
            "ZONE_ANNOTATOR_WINDOW_NAME",
            default=defaults.zone_annotator_window_name,
        ),
        yolo_show_window=_parse_bool(
            _get_env("YOLO_SHOW_WINDOW", default=str(defaults.yolo_show_window)),
            defaults.yolo_show_window,
        ),
        yolo_fourcc=_get_env("YOLO_FOURCC", default=defaults.yolo_fourcc),
        yolo_zones_path=_resolve_path(
            _get_env("YOLO_ZONES_SAVE_PATH", default=defaults.yolo_zones_path),
            project_root,
        ),
        yolo_draw_zones=_parse_bool(
            _get_env("YOLO_DRAW_ZONES", default=str(defaults.yolo_draw_zones)),
            defaults.yolo_draw_zones,
        ),
    )
