import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.database import db
from worker.settings import get_settings
from worker.yolo_final import (
    build_scaled_zones,
    detect_zone_for_bbox_xywh,
    draw_zone_highlight,
    draw_zones,
    load_zones_payload,
)


class ActiveEventState(TypedDict):
    start_time: datetime
    confidences: list[float]
    zone: str
    features: dict[str, float]


def process_live_stream(video_placeholder, worker_settings):
    """Live-потік з тією ж CV-логікою, що і в yolo_final (зони + нормалізовані фічі)."""
    yolo_model = YOLO(worker_settings.yolo_model_path)
    yolo_model.to(worker_settings.yolo_device)

    cap = cv2.VideoCapture(worker_settings.yolo_video_path)
    if not cap.isOpened():
        st.error(f"Не вдалося відкрити відео: {worker_settings.yolo_video_path}")
        return

    history: dict[int, dict[str, float]] = {}
    active_events: dict[int, ActiveEventState] = {}

    zones_payload = load_zones_payload(worker_settings.yolo_zones_path) if worker_settings.yolo_draw_zones else None
    frame_zones = None

    try:
        while cap.isOpened() and st.session_state.get("is_running", False):
            success, frame = cap.read()
            if not success:
                st.warning("Відеопотік завершився.")
                break

            results = yolo_model.track(
                frame,
                persist=True,
                tracker=worker_settings.yolo_tracker,
                classes=list(worker_settings.yolo_classes),
                imgsz=worker_settings.yolo_img_size,
                conf=worker_settings.yolo_confidence,
                verbose=False,
            )
            annotated_frame = results[0].plot()

            if zones_payload is not None:
                if frame_zones is None:
                    frame_height, frame_width = annotated_frame.shape[:2]
                    frame_zones = build_scaled_zones(zones_payload, frame_width, frame_height)
                if frame_zones:
                    draw_zones(annotated_frame, frame_zones)

            now = datetime.now()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                bboxes = results[0].boxes.xywh.cpu().numpy()
                confs = results[0].boxes.conf.cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    cx, cy, w, h = bboxes[i]
                    conf = float(confs[i])
                    current_zone = detect_zone_for_bbox_xywh(cx, cy, frame_zones or [])
                    draw_zone_highlight(annotated_frame, cx, cy, w, h, current_zone)

                    current_aspect_ratio = w / h if h > 0 else 0.0
                    speed_relative = 0.0
                    speed_relative_change = 0.0
                    aspect_ratio_change = 0.0
                    if track_id in history:
                        prev_cx = history[track_id]["center_x"]
                        prev_cy = history[track_id]["center_y"]
                        prev_ar = history[track_id]["aspect_ratio"]
                        prev_speed = history[track_id].get("speed_relative", 0.0)

                        distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                        speed_relative = distance / h if h > 0 else 0.0
                        speed_relative_change = speed_relative - prev_speed
                        aspect_ratio_change = current_aspect_ratio - prev_ar

                    history[track_id] = {
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "aspect_ratio": float(current_aspect_ratio),
                        "speed_relative": float(speed_relative),
                    }

                    # Лог подій у БД з тією ж структурою фічей, що в train/preprocess.
                    if track_id not in active_events:
                        active_events[track_id] = {
                            "start_time": now,
                            "confidences": [conf],
                            "zone": current_zone,
                            "features": {
                                "speed_relative": round(float(speed_relative), 4),
                                "speed_relative_change": round(float(speed_relative_change), 4),
                                "aspect_ratio": round(float(current_aspect_ratio), 4),
                                "aspect_ratio_change": round(float(aspect_ratio_change), 4),
                            },
                        }
                    else:
                        state = active_events[track_id]
                        state["confidences"].append(conf)
                        state["zone"] = current_zone
                        state["features"] = {
                            "speed_relative": round(float(speed_relative), 4),
                            "speed_relative_change": round(float(speed_relative_change), 4),
                            "aspect_ratio": round(float(current_aspect_ratio), 4),
                            "aspect_ratio_change": round(float(aspect_ratio_change), 4),
                        }

                        if (now - state["start_time"]).total_seconds() > 3:
                            avg_conf = sum(state["confidences"]) / max(len(state["confidences"]), 1)
                            db.log_event(
                                worker_id=f"Worker_{track_id}",
                                classification="Detected (No ML)",
                                zone=state["zone"],
                                start_time=state["start_time"],
                                end_time=now,
                                confidence=float(avg_conf),
                                metadata=state["features"],
                            )
                            active_events[track_id] = {
                                "start_time": now,
                                "confidences": [conf],
                                "zone": current_zone,
                                "features": state["features"],
                            }

            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", width='stretch')
            time.sleep(0.01)
    finally:
        cap.release()


def show_observability_dashboard():
    """Відображає таблицю та аналітику з бази даних."""
    st.subheader("📊 Observability Dashboard")

    try:
        analytics = db.get_zone_analytics()
        if analytics:
            df_analytics = pd.DataFrame(analytics)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Загальна статистика (Zone Analytics)**")
                st.dataframe(df_analytics, width='stretch')
            with col2:
                st.write("**Кількість подій**")
                chart_data = df_analytics.groupby("task_classification")["event_count"].sum()
                st.bar_chart(chart_data)

        st.divider()

        st.write("**Останні цифрові події (Operations Log)**")
        query = "SELECT event_id, worker_id, task_classification, zone, timestamp_start, duration, confidence_score FROM operations_log ORDER BY timestamp_start DESC"
        recent_events = db.execute_query(query)

        if recent_events:
            df_events = pd.DataFrame(recent_events)
            # Додали height=600, щоб розгорнути таблицю на всю висоту і прибрати внутрішній скрол
            st.dataframe(df_events, width='stretch', height=600)
        else:
            st.info("Подій ще немає. Запустіть live-аналіз.")

    except Exception as e:
        st.error(f"Помилка при зчитуванні БД: {e}")


#TODO перенести в файл БД
def clear_operations_log() -> int:
    rows = db.execute_query("SELECT COUNT(*) AS cnt FROM operations_log")
    total = int(rows[0]["cnt"]) if rows else 0
    if total > 0:
        db.execute_query("DELETE FROM operations_log", fetch=False)
    return total


def render_safe_db_cleanup() -> None:
    st.subheader("Безопасная очистка БД")
    rows = db.execute_query("SELECT COUNT(*) AS cnt FROM operations_log")
    total = int(rows[0]["cnt"]) if rows else 0
    st.caption(f"Записей в operations_log: {total}")

    confirm_checked = st.checkbox("Подтверждаю удаление всех событий из БД")
    confirm_phrase = st.text_input("Введите DELETE для подтверждения", value="")

    if st.button("Очистить БД", type="primary"):
        if not confirm_checked:
            st.warning("Поставте галочку подтверждения.")
            return
        if confirm_phrase.strip() != "DELETE":
            st.warning("Неверная фраза подтверждения. Введите DELETE.")
            return

        removed = clear_operations_log()
        st.success(f"Удалено записей: {removed}")
        st.rerun()


def main():
    st.set_page_config(page_title="Live Аналіз Логістики", layout="wide")
    st.title("🎥 Інструмент Live-аналізу")

    # Гарантуємо, що таблиця існує
    db.init_schema()

    worker_settings = get_settings()

    if not Path(worker_settings.yolo_model_path).exists():
        st.error(f"YOLO модель не знайдена: {worker_settings.yolo_model_path}")
        st.stop()

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    col_video, col_controls = st.columns([3, 1])

    with col_controls:
        st.header("Управління")
        st.write(f"**Джерело:** `{worker_settings.yolo_video_path}`")
        st.write(f"**ML Класифікація:** Вимкнена 🔴")

        if not st.session_state.is_running:
            if st.button("▶️ Почати Live Аналіз", width='stretch'):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("⏹ Зупинити Аналіз", type="primary", width='stretch'):
                st.session_state.is_running = False
                st.rerun()

        st.divider()
        if st.button("🔄 Оновити Дашборд", width='stretch'):
            pass

        st.divider()
        render_safe_db_cleanup()

    with col_video:
        video_placeholder = st.empty()

        if st.session_state.is_running:
            process_live_stream(video_placeholder, worker_settings)
        else:
            video_placeholder.info("Натисніть 'Почати Live Аналіз' для запуску відеопотоку.")

    st.divider()
    show_observability_dashboard()


if __name__ == "__main__":
    main()