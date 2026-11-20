FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway.py index.html ./

EXPOSE 8000
CMD ["sh", "-c", "uvicorn gateway:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]