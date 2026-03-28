import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import cv2
import joblib
import pandas as pd
import streamlit as st
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.database import db
from app.settings import AppSettings, get_app_settings
from worker.settings import get_settings
from worker.yolo_final import (
    build_scaled_zones,
    detect_zone_for_bbox_xywh,
    draw_zone_highlight,
    draw_zones,
    load_zones_payload,
    resolve_yolo_device,
)


# Keep only true legacy aliases here; Sorting is now a standalone class.
ACTION_ALIASES = {}


def normalize_action_value(value: object) -> str:
    action = str(value).strip()
    return ACTION_ALIASES.get(action, action)


class ActiveEventState(TypedDict):
    start_time: datetime
    confidences: list[float]
    zone: str
    features: dict[str, float]


@st.cache_resource
def load_rf_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_action(
    model: Any,
    features: dict[str, float],
    train_features: tuple[str, ...],
    fallback_confidence: float,
) -> tuple[str, float]:
    if model is None:
        return "Detected (No ML)", fallback_confidence

    feature_row = {name: float(features.get(name, 0.0)) for name in train_features}
    X = pd.DataFrame([feature_row], columns=list(train_features))

    predicted_label = normalize_action_value(model.predict(X)[0])
    confidence = fallback_confidence
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())
    return predicted_label, confidence


def draw_action_label(frame, cx: float, cy: float, w: float, h: float, action: str, confidence: float) -> None:
    x1 = int(cx - (w / 2.0))
    y1 = int(cy - (h / 2.0))
    label_y = max(18, y1 - 50)
    label = f"{action} ({confidence:.2f})"
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def process_live_stream(video_placeholder, worker_settings, app_settings: AppSettings, rf_model: Any, effective_device: str):
    """Live-потік із тією ж CV-логікою, що і в yolo_final (зони + нормалізовані фічі)."""
    yolo_model = YOLO(worker_settings.yolo_model_path)
    yolo_model.to(effective_device)

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

                    current_features = {
                        "speed_relative": round(float(speed_relative), 4),
                        "speed_relative_change": round(float(speed_relative_change), 4),
                        "aspect_ratio": round(float(current_aspect_ratio), 4),
                        "aspect_ratio_change": round(float(aspect_ratio_change), 4),
                    }

                    current_action, current_action_conf = predict_action(
                        model=rf_model,
                        features=current_features,
                        train_features=app_settings.train_features,
                        fallback_confidence=conf,
                    )
                    draw_action_label(annotated_frame, cx, cy, w, h, current_action, current_action_conf)

                    history[track_id] = {
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "aspect_ratio": float(current_aspect_ratio),
                        "speed_relative": float(speed_relative),
                    }

                    if track_id not in active_events:
                        active_events[track_id] = {
                            "start_time": now,
                            "confidences": [conf],
                            "zone": current_zone,
                            "features": current_features,
                        }
                    else:
                        state = active_events[track_id]
                        state["confidences"].append(conf)
                        state["zone"] = current_zone
                        state["features"] = current_features

                        if (now - state["start_time"]).total_seconds() > 3:
                            avg_conf = sum(state["confidences"]) / max(len(state["confidences"]), 1)
                            predicted_class, predicted_conf = predict_action(
                                model=rf_model,
                                features=state["features"],
                                train_features=app_settings.train_features,
                                fallback_confidence=float(avg_conf),
                            )

                            metadata = dict(state["features"])
                            metadata["detector_avg_confidence"] = round(float(avg_conf), 4)

                            db.log_event(
                                worker_id=f"Worker_{track_id}",
                                classification=predicted_class,
                                zone=state["zone"],
                                start_time=state["start_time"],
                                end_time=now,
                                confidence=float(predicted_conf),
                                metadata=metadata,
                            )
                            active_events[track_id] = {
                                "start_time": now,
                                "confidences": [conf],
                                "zone": current_zone,
                                "features": state["features"],
                            }

            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", width="stretch")
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
            if "task_classification" in df_analytics.columns:
                df_analytics["task_classification"] = df_analytics["task_classification"].map(normalize_action_value)
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
            if "task_classification" in df_events.columns:
                df_events["task_classification"] = df_events["task_classification"].map(normalize_action_value)
            # Розгортаємо таблицю на всю висоту, щоб прибрати внутрішній скрол.
            st.dataframe(df_events, width='stretch', height=600)
        else:
            st.info("Подій ще немає. Запустіть live-аналіз.")

    except Exception as e:
        st.error(f"Помилка під час зчитування БД: {e}")


def render_safe_db_cleanup() -> None:
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


def main():
    st.set_page_config(page_title="Live аналіз логістики", layout="wide")
    st.title("🎥 Інструмент live-аналізу")

    db.init_schema()

    worker_settings = get_settings()
    app_settings = get_app_settings()
    effective_device = resolve_yolo_device(worker_settings.yolo_device)

    if not Path(worker_settings.yolo_model_path).exists():
        st.error(f"YOLO модель не знайдена: {worker_settings.yolo_model_path}")
        st.stop()

    rf_model = None
    rf_error = None
    try:
        rf_model = load_rf_model(str(app_settings.model_file))
        if rf_model is None:
            rf_error = f"RF модель не знайдена: {app_settings.model_file}"
    except Exception as exc:
        rf_error = f"Не вдалося завантажити RF модель: {exc}"

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    col_video, col_controls = st.columns([3, 1])

    with col_controls:
        st.header("Керування")
        st.write(f"**Джерело:** `{worker_settings.yolo_video_path}`")
        device_badge = "🟢 GPU (CUDA)" if effective_device.startswith("cuda") else "🟠 CPU"
        st.write(f"**Рендер/інференс:** {device_badge}")
        if effective_device != worker_settings.yolo_device:
            st.caption(f"Запитаний пристрій: `{worker_settings.yolo_device}` -> фактичний: `{effective_device}`")
        if rf_model is not None:
            st.write("**ML класифікація:** RandomForest увімкнено 🟢")
        else:
            st.write("**ML класифікація:** резервний режим без ML 🟠")
            if rf_error:
                st.caption(rf_error)

        if not st.session_state.is_running:
            if st.button("▶️ Почати live-аналіз", width="stretch"):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("⏹ Зупинити аналіз", type="primary", width="stretch"):
                st.session_state.is_running = False
                st.rerun()

        st.divider()
        if st.button("🔄 Оновити дашборд", width='stretch'):
            pass

        st.divider()
        render_safe_db_cleanup()

    with col_video:
        video_placeholder = st.empty()

        if st.session_state.is_running:
            process_live_stream(video_placeholder, worker_settings, app_settings, rf_model, effective_device)
        else:
            video_placeholder.info("Натисніть 'Почати live-аналіз' для запуску відеопотоку.")

    st.divider()
    show_observability_dashboard()


if __name__ == "__main__":
    main()