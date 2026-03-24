import streamlit as st
import pandas as pd
import json
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib  # Для сохранения модели

# Прописываем пути к файлам (убедись, что они совпадают с твоей структурой)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Строим пути относительно BASE_DIR (поднимаемся на уровень выше и заходим куда нужно)
FEATURES_FILE = os.path.abspath(os.path.join(BASE_DIR, '../data/features_temp.json'))
DATASET_FILE = os.path.abspath(os.path.join(BASE_DIR, '../data/labeled_dataset.csv'))
MODEL_FILE = os.path.abspath(os.path.join(BASE_DIR, '../models/rf_v1.pkl'))

VIDEO_PATH = os.path.abspath(os.path.join(BASE_DIR, '../data/preview_with_ids.mp4'))

st.set_page_config(page_title="Разметка и Обучение", layout="wide")
st.title("Инструмент обучения (День 3)")

ORIGINAL_VIDEO_PATH = os.path.abspath(os.path.join(BASE_DIR, '../data/video3.mkv'))
cap = cv2.VideoCapture(ORIGINAL_VIDEO_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# --- 1. ЗАГРУЗКА ФИЧЕЙ ОТ YOLO ---
@st.cache_data
def load_features():
    print(os.path.exists(FEATURES_FILE))
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'r') as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


df_features = load_features()

if df_features.empty:
    st.warning("Нет данных от YOLO. Сначала запусти скрипт препроцессинга (День 2).")
    st.stop()

# --- 2. ИНТЕРАКТИВНАЯ РАЗМЕТКА ---
st.header("1. Разметка действий")

if os.path.exists(VIDEO_PATH):
    st.video(VIDEO_PATH)
else:
    st.error(f"Не могу найти видео по пути: {VIDEO_PATH}")
st.divider()

# Виджеты из плана: выбор ID, слайдер времени (кадров), селектор действия
col1, col2, col3 = st.columns(3)

with col1:
    track_ids = df_features['track_id'].unique()
    selected_id = st.selectbox("ID рабочего:", track_ids)

with col2:
    # Берем минимальный и максимальный кадр для выбранного ID
    id_data = df_features[df_features['track_id'] == selected_id]
    min_frame = int(id_data['frame_id'].min())
    max_frame = int(id_data['frame_id'].max())

    if min_frame == max_frame:
        st.warning("Слишком мало кадров для этого ID.")
        selected_frames = (min_frame, max_frame)
    else:
        # Переводим кадры в секунды
        min_sec = float(min_frame / FPS)
        max_sec = float(max_frame / FPS)

        # Ползунок теперь в секундах
        selected_sec = st.slider("Интервал (секунды):",
                                 min_value=min_sec,
                                 max_value=max_sec,
                                 value=(min_sec, max_sec),
                                 step=0.5,
                                 format="%.1f s")

        # Переводим выбранные секунды обратно в кадры для датасета
        selected_frames = (int(selected_sec[0] * FPS), int(selected_sec[1] * FPS))

with col3:
    action = st.selectbox("Действие:", ["Idle", "Moving", "Sorting"])

# Кнопка добавления в датасет
if st.button("Добавить в датасет"):
    # Фильтруем данные по ID и выбранному интервалу кадров
    mask = (df_features['track_id'] == selected_id) & \
           (df_features['frame_id'] >= selected_frames[0]) & \
           (df_features['frame_id'] <= selected_frames[1])

    df_selected = df_features[mask].copy()
    df_selected['action'] = action  # Присваиваем класс

    # Сохраняем или дописываем в общий CSV датасет
    if os.path.exists(DATASET_FILE):
        df_existing = pd.read_csv(DATASET_FILE)
        df_combined = pd.concat([df_existing, df_selected], ignore_index=True)
    else:
        df_combined = df_selected

    df_combined.to_csv(DATASET_FILE, index=False)
    st.success(f"Успешно! {len(df_selected)} строк размечено как '{action}'.")

st.divider()

# --- 3. ОБУЧЕНИЕ МОДЕЛИ ---
st.header("2. Обучение Random Forest")

if os.path.exists(DATASET_FILE):
    df_dataset = pd.read_csv(DATASET_FILE)

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.write(f"**Размер датасета:** {len(df_dataset)} записей")
    with col_stat2:
        st.write("**Баланс классов:**")
        st.dataframe(df_dataset['action'].value_counts())

    if st.button("Обучить модель"):
        # Подготовка X и y
        X = df_dataset[['speed_relative', 'aspect_ratio', 'aspect_ratio_change']]
        y = df_dataset['action']

        # Обучаем классификатор
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        # Сохраняем модель
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump(clf, MODEL_FILE)

        st.success(f"Модель успешно обучена и сохранена в `{MODEL_FILE}`!")
        # Выводим простую метрику (Accuracy на тренировочной выборке для наглядности)
        st.metric("Точность (Accuracy)", f"{clf.score(X, y):.2%}")
else:
    st.info("Сначала разметьте немного данных, чтобы запустить обучение.")