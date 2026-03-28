# AI Warehouse Analytics

Система для аналізу операцій працівників складу з розділенням на два незалежні контури:
- **Training Tool (Streamlit + ML)**: розмітка дій, підготовка датасету, навчання `RandomForest`.
- **Live Analysis Tool (Streamlit + SQLite + RF)**: live-візуалізація, інференс, події та observability.

## Актуальний workflow (ваш сценарій)

1. Запустити `worker/yolo_final.py` на відео, щоб отримати `preview_with_ids.mp4` і `features_temp.json`.
2. Запустити `worker/zone_annotator.py`, щоб створити/оновити `zones.json`.
3. Запустити `app/main.py` для розмітки дій та навчання `RandomForest`.
4. Запустити `app/live_analysis.py` для live-аналізу.

Це підтримується поточною конфігурацією: **у Docker запускаються тільки UI-сервіси**, а CV-підготовка виконується локально як dev-скрипти.

## Архітектура

```text
(Local dev) worker/yolo_final.py + worker/zone_annotator.py
                |
                v
  data/output/preview_with_ids.mp4, features_temp.json, zones.json
                |
                v
       app/main.py (розмітка + train RF -> models/rf_v1.pkl)
                |
                v
        app/live_analysis.py (live inference + events.db + dashboard)
```

Ключовий принцип: **важкий CV/ML ізольований від UI**.

## Структура проєкту

```text
app/         Streamlit застосунки (навчання + live-аналітика)
worker/      CV-скрипти підготовки (YOLO, зони)
models/      Ваги YOLO і збережені ML-моделі
data/        SQLite, preview, features, проміжні артефакти
experiments/ Допоміжні скрипти для досліджень
```

## Передумови

- Python `>=3.10, <3.13`
- `uv`
- Docker Desktop
- (Опційно) GPU/CUDA

## Швидкий старт

### 1) Підготовка артефактів локально (без Docker)

```bash
uv sync
uv run python worker/yolo_final.py
uv run python worker/zone_annotator.py
```

### 2) Запуск UI через Docker

```bash
docker-compose up --build
```

Після запуску:
- Training UI: `http://localhost:8501`
- Live UI: `http://localhost:8502`

Зупинка:

```bash
docker-compose down
```

## Локальний запуск без Docker (повна інструкція)

1) Встановіть залежності:

```bash
uv sync
```

2) Підготуйте артефакти для навчання (preview/features/zones):

```bash
uv run python worker/yolo_final.py
uv run python worker/zone_annotator.py
```

3) Запустіть **Training UI** (термінал №1):

```bash
uv run streamlit run app/main.py
```

4) Запустіть **Live Analysis UI** (термінал №2):

```bash
uv run streamlit run app/live_analysis.py --server.port 8502
```

5) Відкрийте інтерфейси у браузері:
- Training UI: `http://localhost:8501`
- Live UI: `http://localhost:8502`

> Якщо хочете, можна запускати тільки один із сервісів (лише training або лише live) відповідною командою вище.

## Локальний запуск UI (альтернатива Docker)

```bash
uv run streamlit run app/main.py
uv run streamlit run app/live_analysis.py --server.port 8502
```

## Що реалізовано

### 4.1 Інструмент навчання
- Перегляд preview-відео після CV-препроцесингу.
- Розмітка дій по `track_id` і часовому інтервалу.
- Формування датасету (`data/labeled_dataset.csv`).
- Навчання `RandomForestClassifier` і збереження моделі (`models/rf_v1.pkl`).

### 4.2 Інструмент live-аналізу
- Обробка відеопотоку/файлу в інтерфейсі live-аналізу.
- Класифікація дій через `rf_v1.pkl` (якщо модель доступна).
- Запис цифрових подій в SQLite (`data/events.db`).
- Observability (таблиці/графіки), безпечне очищення БД.

## Конфігурація через `.env`

### App
- `APP_FEATURES_FILE`
- `APP_DATASET_FILE`
- `APP_MODEL_FILE`
- `APP_PREVIEW_VIDEO_PATH`
- `APP_ORIGINAL_VIDEO_PATH`
- `APP_DB_PATH`

### Worker
- `WORKER_YOLO_MODEL_PATH`
- `WORKER_YOLO_VIDEO_PATH`
- `WORKER_YOLO_ZONES_PATH`
- `WORKER_YOLO_TRACKER`
- `WORKER_YOLO_CONF`
- `WORKER_YOLO_DEVICE`
- `WORKER_YOLO_DRAW_ZONES`

> Повний перелік: `app/settings.py`, `worker/settings.py`.

## Troubleshooting
- **Розсинхрон часу відео і шкали**: `features_temp.json` і `preview_with_ids.mp4` мають бути з одного запуску `yolo_final.py`.
- **RF не підхопився в live**: перевір `APP_MODEL_FILE` та файл `models/rf_v1.pkl`.
- **Немає подій у live-дашборді**: перевір шлях `APP_DB_PATH` і доступність `data/events.db`.

## Технології
- Streamlit
- Ultralytics YOLO + ByteTrack
- scikit-learn (`RandomForestClassifier`)
- SQLite
- Docker / Docker Compose
- uv
