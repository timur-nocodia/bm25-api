FROM python:3.11-slim

# Environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE=/cache \
    HF_HOME=/cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies including requests for health check
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    fastembed==0.3.4 \
    requests

# Copy application
COPY main.py .

# Create cache directory
RUN mkdir -p /cache && chmod 777 /cache

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app /cache

USER app

# Expose port (configurable via environment)
EXPOSE ${PORT:-8080}

# Health check (will be overridden by docker-compose)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8080}/health', timeout=3)" || exit 1

# Use environment variables for all settings
CMD uvicorn main:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8080} \
    --workers ${WORKERS:-1} \
    --log-level ${LOG_LEVEL:-info}