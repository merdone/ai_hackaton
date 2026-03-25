from __future__ import annotations

from datetime import datetime
import sqlite3


def ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            timestamp TEXT,
            worker_id TEXT,
            action TEXT,
            zone TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()


def write_event(conn: sqlite3.Connection, worker_id: str, action: str, zone: str, confidence: float) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (timestamp, worker_id, action, zone, confidence),
    )
    conn.commit()
    print(f"Stored event: {timestamp} | {worker_id} | {action} | {zone} | conf={confidence:.3f}")
