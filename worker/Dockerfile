FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копіюємо файли залежностей із кореня проєкту
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Копіюємо весь репозиторій для доступу до worker/, models/ та data/.
COPY . .

CMD ["uv", "run", "python", "worker/yolo_final.py"]
