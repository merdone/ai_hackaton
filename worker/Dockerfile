FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Runtime libs required by opencv-python in Debian slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо файли залежностей із кореня проєкту
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Копіюємо весь репозиторій для доступу до worker/, models/ та data/.
COPY . .

CMD ["uv", "run", "python", "worker/yolo_final.py"]
