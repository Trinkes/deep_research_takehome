# Stage 1: Build stage
FROM python:3.13.5-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.23 /uv /uvx /bin/

WORKDIR /app

COPY README.md README.md
COPY pyproject.toml uv.lock ./

# Install dependencies and ensure CLI scripts are installed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable


# Stage 2: Runtime stage
FROM python:3.13.5-slim

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app
COPY . /app/

# Set Python path to use the venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
