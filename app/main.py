from pathlib import Path
import sqlite3
import sys
import time

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime import events_db_path

DEFAULT_LIMIT = 25


def load_events(limit: int) -> pd.DataFrame:
    query = """
        SELECT timestamp, worker_id, action, zone, confidence
        FROM events
        ORDER BY timestamp DESC
        LIMIT ?
    """

    with sqlite3.connect(events_db_path()) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))


st.set_page_config(page_title="Warehouse Analytics", layout="wide")
st.title("Warehouse Analytics")
st.caption(f"SQLite database: {events_db_path()}")

with st.sidebar:
    st.header("Dashboard")
    auto_refresh = st.toggle("Auto refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5)
    limit = st.slider("Rows to display", min_value=10, max_value=100, value=DEFAULT_LIMIT, step=5)
    st.button("Refresh now")

db_path = events_db_path()

if not db_path.exists():
    st.info("Waiting for the worker to create data/events.db.")
else:
    try:
        events = load_events(limit)
        if events.empty:
            st.info("The database exists, but no events have been written yet.")
        else:
            total_events = len(events)
            latest_timestamp = events.iloc[0]["timestamp"]
            unique_workers = events["worker_id"].nunique()

            metric_one, metric_two, metric_three = st.columns(3)
            metric_one.metric("Loaded events", total_events)
            metric_two.metric("Visible workers", unique_workers)
            metric_three.metric("Latest event", latest_timestamp)

            st.dataframe(events, use_container_width=True, hide_index=True)

    except sqlite3.Error as exc:
        st.error(f"Could not read the SQLite database yet: {exc}")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
