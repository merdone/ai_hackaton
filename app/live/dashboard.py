from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils.action_normalization import normalize_action_value
from app.utils.analytics_helpers import safe_primary


def show_observability_dashboard(db):
    """Відображає таблицю подій та розширену аналітику по зонах і працівниках."""
    st.subheader("📊 Observability Dashboard")

    try:
        analytics = db.get_zone_analytics()
        if analytics:
            df_analytics = pd.DataFrame(analytics)
            if "task_classification" in df_analytics.columns:
                df_analytics["task_classification"] = df_analytics["task_classification"].map(normalize_action_value)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Загальна статистика (Zone Analytics)**")
                st.dataframe(df_analytics, width="stretch")
            with col2:
                st.write("**Кількість подій**")
                chart_data = df_analytics.groupby("task_classification")["event_count"].sum()
                st.bar_chart(chart_data)

        st.divider()

        stats_query = (
            "SELECT worker_id, zone, task_classification, duration, confidence_score, timestamp_start "
            "FROM operations_log"
        )
        stats_rows = db.execute_query(stats_query)

        if stats_rows:
            df_stats = pd.DataFrame(stats_rows)
            df_stats["task_classification"] = df_stats["task_classification"].map(normalize_action_value)
            df_stats["duration"] = pd.to_numeric(df_stats["duration"], errors="coerce").fillna(0.0)
            df_stats["confidence_score"] = pd.to_numeric(df_stats["confidence_score"], errors="coerce").fillna(0.0)
            df_stats["zone"] = df_stats["zone"].fillna("None").astype(str)
            df_stats["worker_id"] = df_stats["worker_id"].fillna("Unknown").astype(str)

            st.write("**Статистика по зонах**")
            zone_stats = (
                df_stats.groupby("zone", as_index=False)
                .agg(
                    events=("worker_id", "count"),
                    workers=("worker_id", "nunique"),
                    total_duration_sec=("duration", "sum"),
                    avg_confidence=("confidence_score", "mean"),
                )
                .sort_values("events", ascending=False)
            )
            zone_stats["total_duration_min"] = (zone_stats["total_duration_sec"] / 60.0).round(2)
            zone_stats["avg_confidence"] = zone_stats["avg_confidence"].round(3)
            st.dataframe(
                zone_stats[["zone", "events", "workers", "total_duration_min", "avg_confidence"]],
                width="stretch",
            )

            st.write("**Статистика по ID працівників**")
            worker_stats = (
                df_stats.groupby("worker_id", as_index=False)
                .agg(
                    events=("zone", "count"),
                    zones_covered=("zone", "nunique"),
                    total_duration_sec=("duration", "sum"),
                    avg_confidence=("confidence_score", "mean"),
                    dominant_zone=("zone", safe_primary),
                )
                .sort_values("events", ascending=False)
            )
            worker_stats["total_duration_min"] = (worker_stats["total_duration_sec"] / 60.0).round(2)
            worker_stats["avg_confidence"] = worker_stats["avg_confidence"].round(3)
            st.dataframe(
                worker_stats[["worker_id", "events", "zones_covered", "dominant_zone", "total_duration_min", "avg_confidence"]],
                width="stretch",
            )

            st.write("**Матриця Worker x Zone (кількість подій)**")
            worker_zone_matrix = pd.crosstab(df_stats["worker_id"], df_stats["zone"])
            st.dataframe(worker_zone_matrix, width="stretch")

        st.divider()

        st.write("**Останні цифрові події (Operations Log)**")
        query = "SELECT event_id, worker_id, task_classification, zone, timestamp_start, duration, confidence_score FROM operations_log ORDER BY timestamp_start DESC"
        recent_events = db.execute_query(query)

        if recent_events:
            df_events = pd.DataFrame(recent_events)
            if "task_classification" in df_events.columns:
                df_events["task_classification"] = df_events["task_classification"].map(normalize_action_value)
            st.dataframe(df_events, width="stretch", height=600)
        else:
            st.info("Подій ще немає. Запустіть live-аналіз.")

    except Exception as e:
        st.error(f"Помилка під час зчитування БД: {e}")


def render_safe_db_cleanup(db) -> None:
    st.subheader("Безпечне очищення БД")
    total = db.count_operations_log()
    st.caption(f"Записів в operations_log: {total}")

    confirm_checked = st.checkbox("Підтверджую видалення всіх подій із БД")
    confirm_phrase = st.text_input("Введіть DELETE для підтвердження", value="")

    if st.button("Очистити БД", type="primary"):
        if not confirm_checked:
            st.warning("Поставте позначку підтвердження.")
            return
        if confirm_phrase.strip() != "DELETE":
            st.warning("Невірна фраза підтвердження. Введіть DELETE.")
            return

        removed = db.clear_operations_log()
        st.success(f"Видалено записів: {removed}")
        st.rerun()

