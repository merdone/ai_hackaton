from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = ["timestamp", "worker_id", "action", "zone", "confidence"]


def load_recent_events(db_path: Path, limit: int) -> pd.DataFrame:
    query = """
        SELECT timestamp, worker_id, action, zone, confidence
        FROM events
        ORDER BY timestamp DESC
        LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        events = pd.read_sql_query(query, conn, params=(limit,))

    if not events.empty:
        events["confidence"] = events["confidence"].round(3)

    return events
