import argparse
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"


def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        conn = psycopg2.connect(database_url, hostaddr="52.76.246.190)
    else:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "l1_support_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
        )

    # Ensure pgvector extension, table, and HNSW index exist before registering
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_tenant_hnsw
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
            """
        )
    conn.commit()

    register_vector(conn)
    return conn


def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 120, overlap: int = 25) -> list[str]:
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    words = full_text.split()

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_document(pdf_path: str, tenant_id: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    source_file = os.path.basename(pdf_path)
    print(f"[{tenant_id}] Chunking {source_file}...")
    chunks = extract_and_chunk_pdf(pdf_path)

    print(f"[{tenant_id}] Generating embeddings for {len(chunks)} chunks...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, normalize_embeddings=True)

    rows = [
        (tenant_id, source_file, chunk, emb.tolist())
        for chunk, emb in zip(chunks, embeddings)
    ]

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Remove stale chunks for this exact tenant + file to prevent duplicates
            cur.execute(
                "DELETE FROM document_chunks WHERE tenant_id = %s AND source_file = %s;",
                (tenant_id, source_file),
            )
            execute_values(
                cur,
                """
                INSERT INTO document_chunks (tenant_id, source_file, content, embedding)
                VALUES %s
                """,
                rows,
            )
        conn.commit()
        print(f"[{tenant_id}] Successfully synced {len(rows)} chunks from {source_file} into PostgreSQL.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a tenant PDF into PostgreSQL pgvector.")
    parser.add_argument("--file", required=True, help="Path to the PDF file")
    parser.add_argument("--tenant", required=True, help="Tenant ID (e.g., zara, nike, amazon)")
    args = parser.parse_args()

    ingest_document(pdf_path=args.file, tenant_id=args.tenant)