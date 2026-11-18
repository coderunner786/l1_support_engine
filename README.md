# Enterprise L1 Support Engine

A high-concurrency, multi-tenant **Retrieval-Augmented Generation (RAG)** gateway built with **Async FastAPI**, **PostgreSQL (`pgvector` + HNSW)**, and **SentenceTransformers**. Designed to automate Tier-1 customer support across multiple brands from a single low-latency service.

---

## Why This Replaces Traditional L1 BPO Workflows

In a traditional BPO (Business Process Outsourcing) setup, a single support floor manages multiple client brands simultaneously, with agents juggling 3–5 concurrent chat windows while manually searching PDF policy manuals for return windows, refund timelines, and damage claims.

This engine replaces repetitive L1 lookup work by handling **multi-brand isolation and high-concurrency chat traffic** natively:

* **One Engine, Multiple Client Brands (Multi-Tenancy):** A single deployment isolates knowledge bases by `tenant_id` (e.g., `zara`, `nike`, `amazon`) at the database query level—mirroring how a BPO handles multiple enterprise accounts without cross-brand data leakage.
* **Massive Concurrency Over Human Chat Limits:** While a human agent caps out at a few concurrent chat windows, the async FastAPI + `asyncpg` connection pool processes **1,000+ concurrent requests** across tenants in seconds.
* **Built-In Compliance & PII Guardrails:** Customer emails, phone numbers, and credit card numbers are automatically redacted *before* text ever hits the embedding model or vector database.
* **Human-in-the-Loop Escalation (Confidence Fallback):** Queries that fall below the configurable cosine similarity threshold (`min_similarity`) automatically trigger a structured fallback response, routing only complex L2/L3 edge cases to human agents.

---

## What It Does

* **Live Support Query Mode:** Test single customer inquiries against any tenant's policy manual with real-time latency tracking, similarity scores, PII redaction previews, and confidence fallback testing.
* **1,000-Request Load Simulator:** Built-in browser stress test that fires 1,000 concurrent requests across 25 async workers to benchmark throughput (`req/s`), average latency, P95 latency, PII redactions, and fallback rates.
* **Single-Service Deployment:** FastAPI serves both the REST API (`/support/query`) and the standalone monochrome HTML/CSS/JS frontend (`/`) with zero external CDN dependencies.

---

## How It Works

```text
Customer Query (tenant_id + text)
        │
        ▼
┌───────────────────┐
│  1. PII Masking   │  Regex redacts emails, 10-digit phones, and 16-digit cards
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 2. Response Cache │  In-memory cache hit → returns in ~1ms (skips model & DB)
└─────────┬─────────┘
          ▼ (cache miss)
┌───────────────────┐
│ 3. Embedding Gen  │  all-MiniLM-L6-v2 (384-dim, normalized) via asyncio.to_thread
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 4. pgvector HNSW  │  Async cosine similarity search filtered by tenant_id
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 5. Guardrail Gate │  Score >= min_similarity → Verified Chunks | Else → Human Escalation
└───────────────────┘
```

1. **Ingestion (`ingest.py`):** Extracts text from tenant PDF manuals, splits them into overlapping 120-word chunks, generates 384-dimensional normalized embeddings (`all-MiniLM-L6-v2`), and syncs them into PostgreSQL while clearing stale chunks for that file.
2. **Retrieval (`gateway.py`):** Uses an `asyncpg` connection pool with `pgvector` HNSW indexing (`vector_cosine_ops`) to execute non-blocking similarity searches strictly scoped to `WHERE tenant_id = $2`.
3. **Caching:** Maintains both an embedding cache and a response cache in memory to eliminate redundant CPU inference on repeated L1 FAQs (e.g., *"What is the return policy?"*).

---

## Technical Requirements & Stack

| Layer | Technology |
|---|---|
| **Backend API** | Python 3.11+, FastAPI, Uvicorn (`async`) |
| **Database & Vector Search** | PostgreSQL 16 + `pgvector` (HNSW Cosine Index) |
| **Database Drivers** | `asyncpg` (async runtime), `psycopg2-binary` (local ingestion) |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`, CPU-only PyTorch) |
| **Frontend** | Standalone HTML5 / CSS3 / Vanilla JS (`index.html` via `FileResponse`) |
| **Cloud Hosting** | Neon (Serverless PostgreSQL) + Render (Docker Web Service) |

### Production Dependencies (`requirements.txt`)
```txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
fastapi
uvicorn[standard]
asyncpg
pgvector
python-dotenv
sentence-transformers
```

---

## Setup & Deployment

### 1. Database Schema (PostgreSQL / Neon)
Enable `pgvector` and create the indexed table:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    source_file TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_chunks_tenant_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql://user:password@your-neon-host.neon.tech/neondb?sslmode=require
```

### 3. Ingest Tenant Manuals
```bash
python ingest.py --file zara_massive_manual.pdf --tenant zara
python ingest.py --file nike_massive_manual.pdf --tenant nike
python ingest.py --file amazon_massive_manual.pdf --tenant amazon
```

### 4. Run Locally
```bash
uvicorn gateway:app --host 127.0.0.1 --port 8000 --no-access-log
```
Open **`[http://127.0.0.1:8000](http://127.0.0.1:8000)`** in your browser.

### 5. Deploy to Render (Docker)
1. Push `gateway.py`, `index.html`, `requirements.txt`, and `Dockerfile` to GitHub (keep `.env` in `.gitignore`).
2. Create a new **Web Service** on Render connected to the repository.
3. Add `DATABASE_URL` under **Environment Variables** and deploy.