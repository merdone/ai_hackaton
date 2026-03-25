# AI Warehouse Analytics

Hackathon project for warehouse worker tracking and action analytics with YOLO, Streamlit, and SQLite.

## What Runs Today

- `app/main.py`: Streamlit entrypoint for the dashboard.
- `worker/main.py`: worker entrypoint that runs video inference and writes events into SQLite.
- `common/runtime.py`: shared path helpers for local runs and Docker.

The core product path is intentionally small:

- the worker reads a video and model from `Models/`
- the worker writes events into `data/events.db`
- the dashboard reads recent events from the same database

## Project Layout

- `app/`: dashboard package
  - `main.py`: Streamlit entrypoint
  - `dashboard.py`: page layout and UI rendering
  - `data_access.py`: SQLite reads for the dashboard
- `worker/`: production worker package
  - `main.py`: worker entrypoint
  - `service.py`: video processing loop
  - `rules.py`: action and zone heuristics
  - `storage.py`: SQLite writes
  - `settings.py`: worker configuration values
  - `tracking.py`: lightweight centroid tracker
- `common/`: project-wide shared helpers
  - `runtime.py`: project/data/model path resolution
- `experiments/`: optional scripts for debugging and data prep
  - `extract_frames.py`
  - `box_metrics_demo.py`
  - `box_actions_demo.py`
  - `pose_actions_demo.py`
- `Models/`: runtime assets such as trained models and source videos
- `reference/`: non-code background documents
- `data/`: generated runtime data such as `events.db`

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

Useful environment variables:

- `AI_HACKATON_DATA_DIR`
- `AI_HACKATON_MODELS_DIR`
- `AI_HACKATON_MODEL_PATH`
- `AI_HACKATON_POSE_MODEL_PATH`
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
