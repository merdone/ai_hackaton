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
- `training/`: dataset inspection, train/val split prep, evaluation, and training entrypoints
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
- `AI_HACKATON_TRACK_CONFIDENCE`

## Docker

Start both services:

```bash
docker compose up --build
```

Docker mounts:

- `./data` -> `/app/data`
- `./Models` -> `/app/Models`

The dashboard will be available at [http://localhost:8501](http://localhost:8501).

## Dataset And Training

The worker uses a trained YOLO detector from `Models/best.pt`. If you want to improve confidence, use the helpers in `training/` instead of training directly from a raw Desktop folder.

Why the extra prep step matters:

- Windows desktop paths with non-ASCII characters can break some OpenCV-based tooling.
- A single flat `images/` folder used as both train and val gives misleading evaluation results.
- The prep script creates a deterministic `train/` and `val/` split and writes a clean `dataset.yaml`.

Inspect a raw dataset:

```bash
uv run python training/inspect_dataset.py --source "C:\\path\\to\\worker_data"
```

Prepare a training-ready dataset inside the repo:

```bash
uv run python training/prepare_dataset.py --source "C:\\path\\to\\worker_data" --overwrite
```

Evaluate the current model at several confidence thresholds:

```bash
uv run python training/evaluate_model.py --source "C:\\path\\to\\worker_data"
```

Fine-tune the detector:

```bash
uv run python training/train_model.py --data data\\training\\worker_data\\dataset.yaml
```

For live inference, `AI_HACKATON_TRACK_CONFIDENCE` now defaults to `0.3`. Raising it reduces false positives and usually increases the dashboard average confidence, but it can also hide real detections if you set it too high.
