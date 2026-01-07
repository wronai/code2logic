FROM python:3.12-slim

WORKDIR /app

COPY logic2test/ ./logic2test/

RUN pip install --no-cache-dir ./logic2test

ENTRYPOINT ["logic2test"]
CMD ["--help"]
