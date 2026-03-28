from pathlib import Path
import os
import sys

import streamlit as st

# Support both `streamlit run app/live_analysis.py` and package-style imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database.database import db
from app.live.dashboard import render_safe_db_cleanup, show_observability_dashboard
from app.live.ml import load_rf_model
from app.live.stream import process_live_stream
from app.settings import get_app_settings
from app.utils.path_sources import list_live_video_sources, list_live_zone_sources
from video_pipeline.settings import get_settings
from video_pipeline.yolo_final import resolve_yolo_device


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
    if "ui_yolo_confidence" not in st.session_state:
        st.session_state.ui_yolo_confidence = float(worker_settings.yolo_confidence)
    if "ui_rf_confidence_min" not in st.session_state:
        st.session_state.ui_rf_confidence_min = 0.60
    if "live_video_path" not in st.session_state:
        st.session_state.live_video_path = worker_settings.live_analysis_video_path
    if "live_zones_path" not in st.session_state:
        st.session_state.live_zones_path = worker_settings.live_analysis_zones_path

    available_sources = list_live_video_sources(st.session_state.live_video_path)
    if st.session_state.live_video_path not in available_sources:
        available_sources.insert(0, st.session_state.live_video_path)

    available_zone_sources = list_live_zone_sources(st.session_state.live_zones_path)
    if st.session_state.live_zones_path not in available_zone_sources:
        available_zone_sources.insert(0, st.session_state.live_zones_path)

    col_video, col_controls = st.columns([3, 1])

    with col_controls:
        st.header("Керування")

        current_source_idx = available_sources.index(st.session_state.live_video_path)
        selected_source = st.selectbox(
            "Джерело live-відео",
            options=available_sources,
            index=current_source_idx,
            help="Оберіть файл для live-аналізу. Нове джерело застосовується кнопкою нижче.",
        )
        if st.button("Застосувати джерело", width="stretch"):
            st.session_state.live_video_path = selected_source
            if st.session_state.is_running:
                st.session_state.is_running = False
                st.info("Джерело оновлено. Натисніть 'Почати live-аналіз' для перезапуску з новим відео.")
            st.rerun()

        current_zones_idx = available_zone_sources.index(st.session_state.live_zones_path)
        selected_zones = st.selectbox(
            "Файл зон (JSON)",
            options=available_zone_sources,
            index=current_zones_idx,
            help="Оберіть файл зон для live-аналізу. Новий файл застосовується кнопкою нижче.",
        )
        if st.button("Застосувати зони", width="stretch"):
            st.session_state.live_zones_path = selected_zones
            if st.session_state.is_running:
                st.session_state.is_running = False
                st.info("Файл зон оновлено. Натисніть 'Почати live-аналіз' для перезапуску.")
            st.rerun()

        st.write(f"**Джерело:** `{st.session_state.live_video_path}`")
        st.write(f"**Зони:** `{st.session_state.live_zones_path}`")
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

        st.subheader("Пороги confidence")
        ui_yolo_confidence = st.slider(
            "YOLO confidence",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.ui_yolo_confidence),
            step=0.01,
            help="Мінімальний confidence для детекцій YOLO.",
        )
        ui_rf_confidence_min = st.slider(
            "RF confidence min",
            min_value=0.05,
            max_value=0.99,
            value=float(st.session_state.ui_rf_confidence_min),
            step=0.01,
            help="Події логуються в БД лише якщо confidence RandomForest >= цього порогу.",
        )
        st.session_state.ui_yolo_confidence = float(ui_yolo_confidence)
        st.session_state.ui_rf_confidence_min = float(ui_rf_confidence_min)

        if not st.session_state.is_running:
            if st.button("▶️ Почати live-аналіз", width="stretch"):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("⏹ Зупинити аналіз", type="primary", width="stretch"):
                st.session_state.is_running = False
                st.rerun()

        st.divider()
        if st.button("🔄 Оновити дашборд", width="stretch"):
            pass

        st.divider()
        render_safe_db_cleanup(db)

    with col_video:
        video_placeholder = st.empty()

        if st.session_state.is_running:
            process_live_stream(
                video_placeholder,
                worker_settings,
                app_settings,
                rf_model,
                effective_device,
                float(st.session_state.ui_yolo_confidence),
                float(st.session_state.ui_rf_confidence_min),
                str(st.session_state.live_video_path),
                str(st.session_state.live_zones_path),
                db,
            )
        else:
            video_placeholder.info("Натисніть 'Почати live-аналіз' для запуску відеопотоку.")

    st.divider()
    show_observability_dashboard(db)


if __name__ == "__main__":
    main()
