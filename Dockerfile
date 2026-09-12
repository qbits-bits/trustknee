# ==============================================================================
# Stage 1: Builder (Dependency compilation and virtualenv creation)
# ==============================================================================
FROM python:3.13-slim-bookworm AS builder

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

# Configure Python, uv, and isolated virtual environment paths
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Copy dependency specifications first to maximize layer caching
COPY requirements.txt pyproject.toml ./

# Create virtual environment and install pinned dependencies
RUN uv venv /opt/venv && \
    uv pip install --no-cache -r requirements.txt

# ==============================================================================
# Stage 2: Tester (CI validation: static analysis, linting, unit tests)
# ==============================================================================
FROM builder AS tester

ENV PYTHONPATH="/build"

# Copy source tree and test suite
COPY src/ ./src/
COPY tests/ ./tests/
COPY db/ ./db/
COPY docs/ ./docs/
COPY models/ ./models/

# Execute linter, format check, and test suite during build validation
RUN ruff check . && \
    ruff format --check . && \
    pytest -v --cov=src --cov-report=term-missing --cov-fail-under=70

# ==============================================================================
# Stage 3: Runtime (Minimal, secure production image)
# ==============================================================================
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Create a non-root system user and group (UID/GID 10001)
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy virtual environment and application code with non-root ownership
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser db/ /app/db/

# Switch to non-root user
USER appuser

# Healthcheck to verify Python runtime & imports are operational
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import src.config as cfg; print(f'TrustKnee pipeline ready. {cfg.N_SENSORS} sensors configured.')"]
