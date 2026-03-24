# AI Warehouse Analytics

Hackathon project for warehouse worker tracking and action analytics with YOLO, ByteTrack, Streamlit, and SQLite.

## Current Architecture

- `app/main.py`: Streamlit dashboard that reads `data/events.db` and refreshes automatically.
- `worker/main.py`: YOLO-based worker that tracks people in a video, classifies actions, and writes events into SQLite.
- `common/runtime.py`: Shared path helpers so local runs and Docker use the same project layout.
- `worker/*.py`: Supporting experiments for frame extraction, bounding-box heuristics, and pose-based tests.

## Local Development

Requirements:

- Python 3.10 to 3.12
- `uv`

Install dependencies:

```bash
uv sync
```

Run the dashboard:

```bash
uv run streamlit run app/main.py
```

Run the worker:

```bash
uv run python worker/main.py
```

The worker will create `data/events.db` automatically. By default it uses:

- `Models/best.pt`
- `Models/video_3.mkv`

You can override paths with environment variables:

- `AI_HACKATON_DATA_DIR`
- `AI_HACKATON_MODELS_DIR`
- `AI_HACKATON_MODEL_PATH`
- `AI_HACKATON_VIDEO_PATH`

## Docker

Start both services:

```bash
docker compose up --build
```

Docker mounts:

- `./data` -> `/app/data`
- `./Models` -> `/app/Models`

The dashboard will be available at [http://localhost:8501](http://localhost:8501).

## Repository Structure

- `app/`: Streamlit UI code.
- `worker/`: Worker pipeline and experimentation scripts.
- `common/`: Shared runtime helpers.
- `Models/`: Trained models, videos, and source materials.
- `data/`: Generated SQLite database and derived artifacts.
