from __future__ import annotations

from worker.settings import WorkerSettings


def classify_action(aspect_ratio: float, speed_relative: float, settings: WorkerSettings) -> str:
    if aspect_ratio > settings.sorting_aspect_ratio:
        return "Sorting"
    if speed_relative > settings.running_speed_threshold:
        return "Running"
    if speed_relative > settings.walking_speed_threshold:
        return "Walking"
    return "Idle"


def determine_zone(center_x: float, frame_width: int) -> str:
    return "Zone_A" if center_x < frame_width / 2 else "Zone_B"


def should_write_event(
    last_event: dict[str, float | str] | None,
    action: str,
    zone: str,
    now_monotonic: float,
    settings: WorkerSettings,
) -> bool:
    if last_event is None:
        return True
    if last_event["action"] != action or last_event["zone"] != zone:
        return True
    return now_monotonic - float(last_event["written_at"]) >= settings.write_interval_seconds
