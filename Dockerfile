FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# timeout setting
ENV GUNICORN_TIMEOUT=300
ENV WORKERS=1

# update launch command
CMD exec gunicorn \
    --bind :$PORT \
    --workers $WORKERS \
    --timeout $GUNICORN_TIMEOUT \
    --worker-class uvicorn.workers.UvicornWorker \
    --preload \
    main:app