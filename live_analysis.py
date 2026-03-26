import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from worker.database.db import db
from worker.settings import get_settings


def process_live_stream(video_placeholder, worker_settings):
    """Обробка відео: тільки YOLO трекінг та логування виявлених робітників."""
    # Ініціалізуємо тільки YOLO (без Random Forest)
    yolo_model = YOLO(worker_settings.yolo_model_path)
    yolo_model.to(worker_settings.yolo_device)

    cap = cv2.VideoCapture(worker_settings.yolo_video_path)
    if not cap.isOpened():
        st.error(f"Не вдалося відкрити відео: {worker_settings.yolo_video_path}")
        return

    active_events = {}

    try:
        while cap.isOpened() and st.session_state.get("is_running", False):
            success, frame = cap.read()
            if not success:
                st.warning("Відеопотік завершився.")
                break

            # 1. Трекінг YOLO
            results = yolo_model.track(
                frame,
                persist=True,
                tracker=worker_settings.yolo_tracker,
                classes=list(worker_settings.yolo_classes),
                imgsz=worker_settings.yolo_img_size,
                conf=worker_settings.yolo_confidence,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # 2. Базове логування подій у БД (без ML-класифікації)
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    conf = confs[i]
                    now = datetime.now()

                    # Зберігаємо стан для кожного робітника
                    if track_id not in active_events:
                        active_events[track_id] = {
                            "start_time": now,
                            "confidences": [conf]
                        }
                    else:
                        current_state = active_events[track_id]
                        # Закриваємо подію і пишемо в базу кожні 3 секунди для наочності
                        if (now - current_state["start_time"]).total_seconds() > 3:
                            avg_conf = sum(current_state["confidences"]) / len(current_state["confidences"])

                            db.log_event(
                                worker_id=f"Worker_{track_id}",
                                classification="Detected (No ML)",  # Заглушка замість дій
                                zone="Zone_A",
                                start_time=current_state["start_time"],
                                end_time=now,
                                confidence=avg_conf
                            )

                            # Оновлюємо таймер
                            active_events[track_id] = {
                                "start_time": now,
                                "confidences": [conf]
                            }
                        else:
                            current_state["confidences"].append(conf)

            # Конвертуємо для Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

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
                st.dataframe(df_analytics, use_container_width=True)
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
            st.dataframe(df_events, use_container_width=True, height=600)
        else:
            st.info("Подій ще немає. Запустіть live-аналіз.")

    except Exception as e:
        st.error(f"Помилка при зчитуванні БД: {e}")


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
            if st.button("▶️ Почати Live Аналіз", use_container_width=True):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("⏹ Зупинити Аналіз", type="primary", use_container_width=True):
                st.session_state.is_running = False
                st.rerun()

        st.divider()
        if st.button("🔄 Оновити Дашборд", use_container_width=True):
            pass

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