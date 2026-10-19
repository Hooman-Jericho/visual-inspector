# Multi-stage production Dockerfile for CPU inference
# Stage 1: Build & dependency resolution
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable package installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml ./
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "torch>=2.1.0" \
    "torchvision>=0.16.0" \
    "scipy>=1.11.0" \
    "scikit-learn>=1.3.0" \
    "Pillow>=10.0.0" \
    "pydantic>=2.0" \
    "pyyaml>=6.0" \
    "typer>=0.9.0" \
    "faiss-cpu>=1.7.4"

# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_DIR="/app/artifacts"

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/artifacts && \
    chown -R appuser:appuser /app

# Copy application source
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser configs /app/configs

ENV PYTHONPATH="/app/src"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "visual_inspector.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
