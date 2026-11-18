FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app files
COPY gateway.py index.html ./

# Render injects the PORT environment variable automatically
CMD ["sh", "-c", "uvicorn gateway:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]