# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install wheels to a specific directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Final Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Install only runtime shared libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code
ENV CONFIG_DIR=/app/config
COPY src /app/src
COPY config ${CONFIG_DIR}
#COPY server /app/server

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV WEB_CONCURRENCY=3
EXPOSE 8080

# Cloud Run execution
CMD exec gunicorn -w ${WEB_CONCURRENCY} -b 0.0.0.0:${PORT} --timeout 120 --access-logfile - --error-logfile - src.app:app