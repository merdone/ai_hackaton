# AI Warehouse Analytics

Система для аналізу операцій працівників складу з розділенням на два незалежні контури:
- **Training Tool (Streamlit + ML)**: розмітка дій, підготовка датасету, навчання `RandomForest`.
- **Live Analysis Tool (Streamlit + SQLite + RF)**: live-візуалізація, інференс, події та observability.

## Актуальний workflow (ваш сценарій)

1. Запустити `video_pipeline/yolo_final.py` на відео, щоб отримати `preview_with_ids.mp4` і `features_temp.json`.
2. Запустити `video_pipeline/zone_annotator.py`, щоб створити/оновити `zones.json`.
3. Запустити `app/main.py` для розмітки дій та навчання `RandomForest`.
4. Запустити `app/live_analysis.py` для live-аналізу.

Це підтримується поточною конфігурацією: **у Docker запускаються тільки UI-сервіси**, а CV-підготовка виконується локально як dev-скрипти.

## Архітектура

```text
(Local dev) video_pipeline/yolo_final.py + video_pipeline/zone_annotator.py
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
video_pipeline/  CV-скрипти підготовки (YOLO, зони)
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
uv run python video_pipeline/yolo_final.py
uv run python video_pipeline/zone_annotator.py
```

### 2) Запуск UI через Docker (CPU режим за замовчуванням)

```powershell
docker-compose up --build
```

### 3) Запуск UI через Docker з CUDA (якщо GPU доступна)

```powershell
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Після запуску:
- Training UI: `http://localhost:8501`
- Live UI: `http://localhost:8502`

Зупинка:

```powershell
docker-compose down
```

## Локальний запуск без Docker (повна інструкція)

1) Встановіть залежності:

```bash
uv sync
```

2) Підготуйте артефакти для навчання (preview/features/zones):

```bash
uv run python video_pipeline/yolo_final.py
uv run python video_pipeline/zone_annotator.py
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

Нижче змінні згруповані за модулем, який їх реально використовує.

### Модуль `app/main.py` (Training UI)
- `APP_FEATURES_FILE` - JSON з фічами після `video_pipeline/yolo_final.py`.
- `APP_DATASET_FILE` - CSV для розмітки та навчання `RandomForest`.
- `APP_MODEL_FILE` - куди зберігати натреновану RF-модель (`.pkl`).
- `APP_PREVIEW_VIDEO_PATH` - preview-відео для інтерфейсу розмітки.
- `APP_ORIGINAL_VIDEO_PATH` - fallback-відео для таймлайна (якщо preview недоступне).

### Модуль `app/live_analysis.py` (Live UI)
- `APP_DB_PATH` - SQLite БД для подій та дашбордів (`events.db`).
- `LIVE_ANALYSIS_VIDEO_PATH` - джерело відео саме для live-аналізу.
- `LIVE_ANALYSIS_ZONES_PATH` - JSON файл зон саме для live-аналізу.

### Модуль `video_pipeline/yolo_final.py` (підготовка фіч)
- `YOLO_MODEL_PATH` - шлях до ваг YOLO.
- `YOLO_VIDEO_PATH` - відеоджерело для етапу підготовки (offline preprocessing).
- `YOLO_PREVIEW_SAVE_PATH` - куди зберігати preview-відео з боксами/ID.
- `YOLO_FEATURES_SAVE_PATH` - куди зберігати розраховані фічі (`features_temp*.json`).
- `YOLO_ZONES_SAVE_PATH` - JSON зон, що застосовується під час preprocessing.
- `YOLO_TRACKER` - конфіг трекера (`bytetrack.yaml` тощо).
- `YOLO_CLASSES` - класи детекції через кому (наприклад `0`).
- `YOLO_IMGSZ` - розмір вхідного кадру для YOLO.
- `YOLO_CONF` - мінімальний confidence детекції (0..1).
- `YOLO_DEVICE` - пристрій інференсу: `auto`, `cpu`, `cuda`, `cuda:0`.
- `YOLO_PREVIEW_WIDTH` - ширина output preview-відео.
- `YOLO_PREVIEW_HEIGHT` - висота output preview-відео.
- `YOLO_DRAW_ZONES` - чи малювати зони на кадрах (`true/false`, `1/0`, `yes/no`).
- `YOLO_SHOW_WINDOW` - чи показувати OpenCV-вікно під час обробки.
- `YOLO_FOURCC` - fourcc код для запису preview (наприклад `avc1`).

### Модуль `video_pipeline/zone_annotator.py` (розмітка зон)
- `ZONE_ANNOTATOR_WINDOW_NAME` - назва OpenCV-вікна розмітника зон.

## Usage Case: від відео до навчання і live-аналізу

Нижче один наскрізний сценарій, який можна повторити крок за кроком.

1. **Підготувати артефакти з відео (ID + фічі)**
   - Перевірити `YOLO_VIDEO_PATH`, `YOLO_MODEL_PATH`, `YOLO_FEATURES_SAVE_PATH`, `YOLO_PREVIEW_SAVE_PATH` у `.env`.
   - Запустити preprocessing:

```bash
uv run python video_pipeline/yolo_final.py
```

2. **Підготувати/оновити зони**
   - Перевірити `YOLO_ZONES_SAVE_PATH` (або окремо `LIVE_ANALYSIS_ZONES_PATH` для live).
   - Запустити розмітник зон:

```bash
uv run python video_pipeline/zone_annotator.py
```

3. **Розмітити дані і навчити модель**
   - Запустити Training UI:

```bash
uv run streamlit run app/main.py
```

   - У UI: вибрати `track_id` + інтервал + дію -> додати в датасет -> натиснути "Навчити модель".
   - Результат: модель зберігається у `APP_MODEL_FILE` (типово `models/rf_v1.pkl`).

4. **Запустити live-аналіз із натренованою моделлю**
   - Перевірити `LIVE_ANALYSIS_VIDEO_PATH`, `LIVE_ANALYSIS_ZONES_PATH`, `APP_DB_PATH`, `APP_MODEL_FILE`.
   - Запустити Live UI:

```bash
uv run streamlit run app/live_analysis.py --server.port 8502
```

   - У UI: обрати джерело відео/файл зон -> стартувати live -> переглядати події й метрики.

## Usage Case 2: Docker-сценарій (CPU/GPU)

Цей кейс зручний, коли UI треба запускати ізольовано в контейнерах.

1. **Підготувати артефакти локально (поза Docker)**
   - Налаштувати `.env` для preprocessing (`YOLO_VIDEO_PATH`, `YOLO_FEATURES_SAVE_PATH`, `YOLO_PREVIEW_SAVE_PATH`, `YOLO_ZONES_SAVE_PATH`).
   - Запустити підготовку:

```bash
uv sync
uv run python video_pipeline/yolo_final.py
uv run python video_pipeline/zone_annotator.py
```

2. **Запустити UI-контейнери у CPU-режимі**

```powershell
docker-compose up --build
```

   - Training UI: `http://localhost:8501`
   - Live UI: `http://localhost:8502`

3. **(Опційно) Запустити UI-контейнери у GPU-режимі**

```powershell
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

4. **Потік роботи в UI**
   - У Training UI: виконати розмітку і навчити модель (збереження у `APP_MODEL_FILE`).
   - У Live UI: вибрати джерело відео/зони, стартувати live, переглядати події й аналітику з `APP_DB_PATH`.

5. **Завершення роботи**

```powershell
docker-compose down
```

## Troubleshooting
- **Розсинхрон часу відео і шкали**: `features_temp.json` і `preview_with_ids.mp4` мають бути з одного запуску `yolo_final.py`.
- **RF не підхопився в live**: перевір `APP_MODEL_FILE` та файл `models/rf_v1.pkl`.
- **Немає подій у live-дашборді**: перевір шлях `APP_DB_PATH` і доступність `data/events.db`.
- **`Found no NVIDIA driver` у Docker**:
  1. Переконайтесь, що встановлено актуальний NVIDIA driver у Windows.
  2. У Docker Desktop увімкнений WSL2 backend.
  3. Запускайте GPU-режим через `docker-compose.gpu.yml`.
  4. Перевірте в контейнері: `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- **Повільно працює в Docker**: базовий `docker-compose.yml` запускає CPU-режім. Для GPU використовуйте команду з `-f docker-compose.gpu.yml`.
