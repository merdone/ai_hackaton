import random
import sqlite3
import time
from datetime import datetime

from settings import get_settings


def ensure_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS events (
            timestamp TEXT,
            worker_id TEXT,
            action TEXT,
            zone TEXT,
            confidence REAL
        )
        '''
    )


def main() -> None:
    settings = get_settings()

    conn = sqlite3.connect(settings.db_path)
    cursor = conn.cursor()

    ensure_table(cursor)
    conn.commit()

    print("Worker запущен. Начинаю генерацию тестовых событий...")

    while True:
        now = datetime.now().strftime(settings.timestamp_format)
        worker_id = random.choice(settings.worker_ids)
        action = random.choice(settings.actions)
        zone = random.choice(settings.zones)
        confidence = round(random.uniform(settings.confidence_min, settings.confidence_max), 2)

        cursor.execute(
            'INSERT INTO events VALUES (?, ?, ?, ?, ?)',
            (now, worker_id, action, zone, confidence),
        )
        conn.commit()

        print(f"Событие записано: {now} | {worker_id} | {action}")
        time.sleep(settings.sleep_seconds)


if __name__ == "__main__":
    main()
