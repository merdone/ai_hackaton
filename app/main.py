import json
from pathlib import Path

import cv2
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from settings import AppSettings, get_app_settings


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


@st.cache_data
def load_features(features_file: str) -> pd.DataFrame:
    path = Path(features_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


def select_frame_range(id_data: pd.DataFrame, fps: float) -> tuple[int, int]:
    min_frame = int(id_data["frame_id"].min())
    max_frame = int(id_data["frame_id"].max())

    if min_frame == max_frame:
        st.warning("Слишком мало кадров для этого ID.")
        return min_frame, max_frame

    min_sec = float(min_frame / fps)
    max_sec = float(max_frame / fps)
    selected_sec = st.slider(
        "Интервал (секунды):",
        min_value=min_sec,
        max_value=max_sec,
        value=(min_sec, max_sec),
        step=0.5,
        format="%.1f s",
    )
    return int(selected_sec[0] * fps), int(selected_sec[1] * fps)


def append_to_dataset(dataset_file: Path, df_selected: pd.DataFrame) -> None:
    if dataset_file.exists():
        df_existing = pd.read_csv(dataset_file)
        df_combined = pd.concat([df_existing, df_selected], ignore_index=True)
    else:
        df_combined = df_selected

    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(dataset_file, index=False)


def run_training(settings: AppSettings) -> None:
    if not settings.dataset_file.exists():
        st.info("Сначала разметьте немного данных, чтобы запустить обучение.")
        return

    df_dataset = pd.read_csv(settings.dataset_file)

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.write(f"**Размер датасета:** {len(df_dataset)} записей")
    with col_stat2:
        st.write("**Баланс классов:**")
        st.dataframe(df_dataset["action"].value_counts())

    if st.button("Обучить модель"):
        X = df_dataset[list(settings.train_features)]
        y = df_dataset["action"]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        settings.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, settings.model_file)

        st.success(f"Модель успешно обучена и сохранена в `{settings.model_file}`!")
        st.metric("Точность (Accuracy)", f"{clf.score(X, y):.2%}")


def main() -> None:
    settings = get_app_settings()
    fps = get_video_fps(settings.original_video_path)

    st.set_page_config(page_title="Разметка и Обучение", layout="wide")
    st.title("Инструмент обучения (День 3)")

    df_features = load_features(str(settings.features_file))
    if df_features.empty:
        st.warning("Нет данных от YOLO. Сначала запусти скрипт препроцессинга (День 2).")
        st.stop()

    st.header("1. Разметка действий")

    if settings.preview_video_path.exists():
        st.video(str(settings.preview_video_path))
    else:
        st.error(f"Не могу найти видео по пути: {settings.preview_video_path}")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        track_ids = df_features["track_id"].unique()
        selected_id = st.selectbox("ID рабочего:", track_ids)

    with col2:
        id_data = df_features[df_features["track_id"] == selected_id]
        selected_frames = select_frame_range(id_data, fps)

    with col3:
        action = st.selectbox("Действие:", settings.actions)

    if st.button("Добавить в датасет"):
        mask = (
            (df_features["track_id"] == selected_id)
            & (df_features["frame_id"] >= selected_frames[0])
            & (df_features["frame_id"] <= selected_frames[1])
        )

        df_selected = df_features[mask].copy()
        df_selected["action"] = action

        append_to_dataset(settings.dataset_file, df_selected)
        st.success(f"Успешно! {len(df_selected)} строк размечено как '{action}'.")

    st.divider()

    st.header("2. Обучение Random Forest")
    run_training(settings)


if __name__ == "__main__":
    main()

