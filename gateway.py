import asyncio
import os
import re
from contextlib import asynccontextmanager
import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pgvector.asyncpg import register_vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.35

model = SentenceTransformer(MODEL_NAME)
db_pool: asyncpg.Pool | None = None

# In-memory caches (simulating Redis in production)
embedding_cache: dict[str, list[float]] = {}
response_cache: dict[tuple, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        db_pool = await asyncpg.create_pool(
            dsn=database_url,
            min_size=2,
            max_size=20,
            init=register_vector,
        )
    else:
        db_pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "l1_support_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            min_size=5,
            max_size=30,
            init=register_vector,
        )
    yield
    if db_pool:
        await db_pool.close()


app = FastAPI(
    title="Enterprise L1 Support Engine",
    version="2.2.0",
    lifespan=lifespan,
)


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


class QueryRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = 3
    min_similarity: float = MIN_SIMILARITY


def mask_pii(text: str) -> str:
    """Mask emails, phone numbers, and 16-digit credit card numbers."""
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{10}\b|\b\d{3}[-\.\s]\d{3}[-\.\s]\d{4}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[REDACTED_CC]", text)
    return text


@app.post("/support/query")
async def support_query(req: QueryRequest):
    clean_query = mask_pii(req.query)

    # 1. Response cache hit — skips embedding and PostgreSQL completely
    cache_key = (req.tenant_id, clean_query, req.top_k, req.min_similarity)
    if cache_key in response_cache:
        return response_cache[cache_key]

    # 2. Embedding cache — only offload to a thread on a cache miss
    if clean_query in embedding_cache:
        query_embedding = embedding_cache[clean_query]
    else:
        query_embedding = await asyncio.to_thread(
            lambda: model.encode(clean_query, normalize_embeddings=True).tolist()
        )
        embedding_cache[clean_query] = query_embedding

    # 3. Async multi-tenant vector similarity search
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_file, content, (embedding <=> $1::vector) AS distance
                FROM document_chunks
                WHERE tenant_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3;
                """,
                query_embedding,
                req.tenant_id,
                req.top_k,
            )

        matches = [
            {
                "source_file": row["source_file"],
                "content": row["content"],
                "similarity_score": round(float(1 - row["distance"]), 4),
            }
            for row in rows
            if (1 - row["distance"]) >= req.min_similarity
        ]

        if not matches:
            result = {
                "status": "fallback",
                "tenant_id": req.tenant_id,
                "masked_query": clean_query,
                "message": "I couldn't find a policy that confidently answers your question. Escalating to a human agent.",
                "matches": [],
            }
        else:
            result = {
                "status": "matched",
                "tenant_id": req.tenant_id,
                "masked_query": clean_query,
                "matches": matches,
            }

        response_cache[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))