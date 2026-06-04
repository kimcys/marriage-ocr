FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MARRIAGE_OCR_ENV_FILE=/app/.env

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY config /app/config
COPY plan.md /app/plan.md
COPY .env.example /app/.env.example
COPY README.md /app/README.md
COPY docs /app/docs

RUN pip install --upgrade pip \
    && pip install .

RUN mkdir -p /app/input /app/output /app/debug /app/data /app/logs

EXPOSE 8501

CMD ["python", "-m", "marriage_ocr.cli", "--help"]
