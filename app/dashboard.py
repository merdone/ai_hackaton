from __future__ import annotations

from pathlib import Path
import sqlite3
import time

import pandas as pd
import streamlit as st

from app.data_access import load_recent_events
from common.runtime import default_video_path, events_db_path

DEFAULT_LIMIT = 25
DEFAULT_REFRESH_SECONDS = 5


@st.cache_data(show_spinner=False)
def load_video_bytes(video_path: str) -> bytes:
    return Path(video_path).read_bytes()


def render_sidebar() -> tuple[bool, int, int]:
    with st.sidebar:
        st.header("Dashboard")
        auto_refresh = st.toggle("Auto refresh", value=False)
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=2,
            max_value=30,
            value=DEFAULT_REFRESH_SECONDS,
        )
        limit = st.slider(
            "Rows to display",
            min_value=10,
            max_value=100,
            value=DEFAULT_LIMIT,
            step=5,
        )
        st.button("Refresh now")
        st.caption("Tip: leave auto refresh off while watching the video, or playback may restart.")

    return auto_refresh, refresh_interval, limit


def render_video_panel(video_path: Path, auto_refresh: bool) -> None:
    st.subheader("Analyzed video")
    st.caption(f"Current source: {video_path}")

    if not video_path.exists():
        st.warning("The configured video file was not found.")
        return

    if auto_refresh:
        st.info("Auto refresh is on. The video panel may restart when the page refreshes.")

    st.video(load_video_bytes(str(video_path)))


def render_guide() -> None:
    st.subheader("How to read this dashboard")
    st.markdown(
        """
        This app shows the same video the worker is analyzing and the event rows produced from that analysis.

        `worker_id` is the tracked person ID assigned by the lightweight tracker.

        `action` is a heuristic label based on bounding-box movement and shape, not a ground-truth human label.

        `zone` is a simple left-versus-right split of the frame.

        `confidence` is the detection confidence reported by the YOLO model for that person.
        """
    )


def render_summary(events: pd.DataFrame) -> None:
    latest_timestamp = events.iloc[0]["timestamp"]
    unique_workers = events["worker_id"].nunique()
    average_confidence = float(events["confidence"].mean())

    metric_one, metric_two, metric_three, metric_four = st.columns(4)
    metric_one.metric("Loaded events", len(events))
    metric_two.metric("Visible workers", unique_workers)
    metric_three.metric("Average confidence", f"{average_confidence:.3f}")
    metric_four.metric("Latest event", latest_timestamp)

    charts_left, charts_right = st.columns(2)
    with charts_left:
        st.subheader("Actions")
        st.bar_chart(events["action"].value_counts())

    with charts_right:
        st.subheader("Zones")
        st.bar_chart(events["zone"].value_counts())


def main() -> None:
    st.set_page_config(page_title="Warehouse Analytics", layout="wide")
    st.title("Warehouse Analytics")

    db_path = events_db_path()
    video_path = default_video_path()
    st.caption(f"SQLite database: {db_path}")

    auto_refresh, refresh_interval, limit = render_sidebar()

    video_column, info_column = st.columns([1.4, 1.0])
    with video_column:
        render_video_panel(video_path, auto_refresh)

    with info_column:
        render_guide()

    if not db_path.exists():
        st.info("Waiting for the worker to create data/events.db.")
    else:
        try:
            events = load_recent_events(db_path, limit)
            if events.empty:
                st.info("The database exists, but no events have been written yet.")
            else:
                render_summary(events)
                st.subheader("Recent events")
                st.dataframe(events, use_container_width=True, hide_index=True)
        except sqlite3.Error as exc:
            st.error(f"Could not read the SQLite database yet: {exc}")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
