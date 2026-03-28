from __future__ import annotations

from pathlib import Path


def get_file_version_token(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


def list_live_video_sources(default_path: str) -> list[str]:
    candidates: list[str] = []

    preferred = Path(default_path).expanduser()
    if preferred.exists():
        candidates.append(str(preferred.resolve()))

    search_dir = preferred.parent if preferred.parent.exists() else Path.cwd() / "data" / "input"
    video_patterns = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.m4v")
    if search_dir.exists():
        for pattern in video_patterns:
            for file_path in sorted(search_dir.glob(pattern)):
                resolved = str(file_path.resolve())
                if resolved not in candidates:
                    candidates.append(resolved)

    if not candidates:
        candidates.append(str(preferred.resolve()))

    return candidates


def list_live_zone_sources(default_path: str) -> list[str]:
    candidates: list[str] = []

    preferred = Path(default_path).expanduser()
    if preferred.exists():
        candidates.append(str(preferred.resolve()))

    search_dir = preferred.parent if preferred.parent.exists() else Path.cwd() / "data" / "output"
    if search_dir.exists():
        for file_path in sorted(search_dir.glob("*.json")):
            resolved = str(file_path.resolve())
            if resolved not in candidates:
                candidates.append(resolved)

    if not candidates:
        candidates.append(str(preferred.resolve()))

    return candidates

