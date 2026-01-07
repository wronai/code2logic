FROM python:3.12-slim

WORKDIR /app

COPY logic2test/ ./logic2test/
COPY logic2code/ ./logic2code/

RUN pip install --no-cache-dir ./logic2test && \
    pip install --no-cache-dir ./logic2code

ENTRYPOINT ["logic2code"]
CMD ["--help"]
