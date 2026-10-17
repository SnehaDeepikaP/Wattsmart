# Production Dockerfile for Wattsmart 2.0 (Compatible with Local, Docker Compose & Hugging Face Spaces)
FROM python:3.11-slim AS base

# Install system dependencies (libgomp1 required for XGBoost and LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (uid 1000 is required for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application and model artifacts
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/
COPY docs/ docs/

# Install wattsmart package
RUN pip install --no-cache-dir -e . --no-deps

# Set permissions
RUN chown -R appuser:appuser /app

USER appuser

# Environment variables (Port 7860 is native Hugging Face Spaces port)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "60", "app.main:create_app()"]
