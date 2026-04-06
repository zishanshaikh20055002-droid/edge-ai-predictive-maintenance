# ── Stage 1: builder ──────────────────────────────────────────
# Install all dependencies in an isolated layer.
# This keeps the final image clean — build tools don't ship to production.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (needed to compile some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer.
# As long as requirements.txt doesn't change, pip install won't re-run.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────
# Start fresh from slim — copy only what's needed to run.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for the Docker health check (used in docker-compose.yml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/       ./src/
COPY src/app.py .

# data/ and models/ are NOT copied here — they are mounted as
# Docker volumes in docker-compose.yml so they stay on your host
# and persist between container restarts.

# Create the data directory in case the volume isn't mounted
RUN mkdir -p /app/data /app/models

# Don't run as root — security best practice
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Run with uvicorn
# --host 0.0.0.0 makes it reachable from outside the container
# --workers 1    keeps it single-worker (TFLite interpreter isn't thread-safe)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]