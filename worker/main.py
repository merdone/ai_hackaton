from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import default_model_path, default_video_path, events_db_path
from worker.service import run_worker


def main() -> None:
    run_worker(
        model_path=default_model_path(),
        video_path=default_video_path(),
        db_path=events_db_path(),
    )


if __name__ == "__main__":
    main()
