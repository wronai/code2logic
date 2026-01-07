FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY code2logic/ ./code2logic/

RUN pip install --no-cache-dir .[full]

ENTRYPOINT ["code2logic"]
CMD ["--help"]
