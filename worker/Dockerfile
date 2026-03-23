FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копируем файлы зависимостей из корня проекта
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Копируем содержимое папки worker внутрь контейнера
COPY worker/ ./

CMD ["uv", "run", "main.py"]