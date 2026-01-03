# Production Dockerfile for code2logic
FROM python:3.11-slim

# Set metadata
LABEL maintainer="team@code2logic.dev"
LABEL description="code2logic - Convert codebase structure to logical representations"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY CHANGELOG.md .

# Install code2logic and dependencies
RUN pip install --upgrade pip && \
    pip install -e .[all]

# Create non-root user for security
RUN groupadd -r code2logic && \
    useradd -r -g code2logic -d /app -s /bin/bash code2logic && \
    chown -R code2logic:code2logic /app

USER code2logic

# Create directories for work
RUN mkdir -p /app/workspace /app/output

# Copy the rest of the application
COPY --chown=code2logic:code2logic code2logic/ ./code2logic/
COPY --chown=code2logic:code2logic examples/ ./examples/
COPY --chown=code2logic:code2logic docs/ ./docs/

# Set up entrypoint
COPY --chown=code2logic:code2logic docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose port for MCP server
EXPOSE 8080

# Set default command
ENTRYPOINT ["entrypoint.sh"]
CMD ["code2logic", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD code2logic --version || exit 1
