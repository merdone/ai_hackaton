import json
from pathlib import Path

import cv2
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from settings import AppSettings, get_app_settings

try:
    from utils.action_normalization import normalize_action_column
    from utils.path_sources import get_file_version_token
except ImportError:
    from app.utils.action_normalization import normalize_action_column
    from app.utils.path_sources import get_file_version_token


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


@st.cache_data
def load_features(features_file: str, version_token: int) -> pd.DataFrame:
    _ = version_token
    path = Path(features_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


def get_video_meta(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    effective_fps = fps if fps and fps > 0 else 25.0
    return float(effective_fps), frame_count


def get_timeline_meta(settings: AppSettings) -> tuple[float, int, Path]:
    if settings.preview_video_path.exists():
        fps, frames = get_video_meta(settings.preview_video_path)
        if frames > 0:
            return fps, frames, settings.preview_video_path

    fps, frames = get_video_meta(settings.original_video_path)
    return fps, frames, settings.original_video_path


def select_frame_range(
        fps: float,
        slider_key: str,
        min_frame: int,
        max_frame: int,
) -> tuple[int, int]:
    if min_frame >= max_frame:
        st.warning("Замало кадрів для вибору інтервалу.")
        return min_frame, max_frame

    min_sec = float(min_frame / fps)
    max_sec = float(max_frame / fps)
    selected_sec = st.slider(
        "Інтервал (секунди):",
        min_value=min_sec,
        max_value=max_sec,
        value=(min_sec, max_sec),
        step=0.5,
        format="%.1f s",
        key=slider_key,
    )
    return int(selected_sec[0] * fps), int(selected_sec[1] * fps)


def append_to_dataset(dataset_file: Path, df_selected: pd.DataFrame) -> None:
    normalized_selected = normalize_action_column(df_selected)

    if dataset_file.exists():
        df_existing = pd.read_csv(dataset_file)
        df_existing = normalize_action_column(df_existing)
        df_combined = pd.concat([df_existing, normalized_selected], ignore_index=True)
    else:
        df_combined = normalized_selected

    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(dataset_file, index=False)


def clear_dataset_file(dataset_file: Path) -> bool:
    if not dataset_file.exists():
        return False

    dataset_file.unlink()
    return True


def run_training(settings: AppSettings) -> None:
    if not settings.dataset_file.exists():
        st.info("Спочатку розмітьте трохи даних, щоб запустити навчання.")
        return

    df_dataset = pd.read_csv(settings.dataset_file)
    df_dataset = normalize_action_column(df_dataset)

    required_features = list(settings.train_features)
    missing_features = [col for col in required_features if col not in df_dataset.columns]
    if missing_features:
        st.error(f"У датасеті відсутні колонки для навчання: {', '.join(missing_features)}")
        return

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.write(f"**Розмір датасету:** {len(df_dataset)} записів")
    with col_stat2:
        st.write("**Баланс класів:**")
        st.dataframe(df_dataset["action"].value_counts())

    if st.button("Навчити модель"):
        X = df_dataset[required_features]
        y = df_dataset["action"]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        settings.model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, settings.model_file)

        st.success(f"Модель успішно навчено та збережено у `{settings.model_file}`!")
        st.metric("Точність (Accuracy)", f"{clf.score(X, y):.2%}")


def main() -> None:
    settings = get_app_settings()

    st.set_page_config(page_title="Розмітка та навчання", layout="wide")
    st.title("Інструмент навчання (День 3)")

    fps, video_frames, timeline_source = get_timeline_meta(settings)

    features_version = get_file_version_token(settings.features_file)
    video_version = get_file_version_token(settings.preview_video_path)
    df_features = load_features(str(settings.features_file), features_version)
    if df_features.empty:
        st.warning("Немає даних від YOLO. Спочатку запусти скрипт препроцесингу (День 2).")
        st.stop()

    st.header("1. Розмітка дій")

    if settings.preview_video_path.exists():
        st.video(str(settings.preview_video_path))
    else:
        st.error(f"Не можу знайти відео за шляхом: {settings.preview_video_path}")

    if timeline_source != settings.preview_video_path:
        st.warning(
            "Не вдалося прочитати тривалість preview-відео, шкалу часу взято з original_video_path. "
            "Перевір, що preview згенеровано та доступне."
        )

    if video_frames > 0 and int(df_features["frame_id"].max()) > video_frames:
        st.warning(
            "Фічі довші за поточне preview-відео. Ймовірно, `features_temp_video_3.json` і `preview_with_ids.mp4` "
            "із різних запусків."
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        track_ids = df_features["track_id"].unique()
        selected_id = st.selectbox("ID працівника:", track_ids)

    with col2:
        id_data = df_features[df_features["track_id"] == selected_id]
        slider_key = f"frame_range_{selected_id}_{features_version}_{video_version}"

        features_min_frame = int(df_features["frame_id"].min())
        features_max_frame = int(df_features["frame_id"].max())
        global_min_frame = min(0, features_min_frame)

        if video_frames > 0:
            global_max_frame = min(features_max_frame, max(video_frames - 1, 0))
        else:
            global_max_frame = max(features_max_frame, 0)

        selected_frames = select_frame_range(
            fps=fps,
            slider_key=slider_key,
            min_frame=global_min_frame,
            max_frame=global_max_frame,
        )

        if id_data.empty:
            st.warning("Для вибраного ID немає треків у фічах.")
        else:
            id_min_sec = float(int(id_data["frame_id"].min()) / fps)
            id_max_sec = float(int(id_data["frame_id"].max()) / fps)
            st.caption(f"ID {selected_id} знайдено в діапазоні: {id_min_sec:.1f}s - {id_max_sec:.1f}s")

    with col3:
        action = st.selectbox("Дія:", settings.actions)

    if st.button("Додати в датасет"):
        mask = (
                (df_features["track_id"] == selected_id)
                & (df_features["frame_id"] >= selected_frames[0])
                & (df_features["frame_id"] <= selected_frames[1])
        )

        df_selected = df_features[mask].copy()
        df_selected["action"] = action

        append_to_dataset(settings.dataset_file, df_selected)
        st.success(f"Успішно! {len(df_selected)} рядків розмічено як '{action}'.")

    st.caption(f"Поточний файл датасету: `{settings.dataset_file}`")
    confirm_clear = st.checkbox("Підтверджую очищення датасету")
    if st.button("Очистити датасет"):
        if not confirm_clear:
            st.warning("Постав позначку підтвердження перед очищенням.")
        elif clear_dataset_file(settings.dataset_file):
            st.success("Датасет очищено.")
            st.rerun()
        else:
            st.info("Файл датасету вже відсутній.")

    st.divider()

    st.header("2. Навчання Random Forest")
    run_training(settings)


if __name__ == "__main__":
    main()
