-- Requires Postgres with pgvector extension installed
-- (https://github.com/pgvector/pgvector)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,           -- e.g. filename or URL
    chunk_text  TEXT NOT NULL,           -- the text chunk itself
    -- 384 = dimension of bge-small-en-v1.5. If you change embedding model,
    -- this dimension MUST match Embedder::dim() on the client side.
    embedding   vector(384) NOT NULL,
    category    TEXT NOT NULL DEFAULT 'Other', -- Thematic classification (e.g., "Technology, Science and Computing", "Health, Biology and Wellbeing", etc.)
    subdomain   TEXT,                    -- Subdomain (e.g., "AI", "DEV", "MED", etc.)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS original_documents (
    filename    TEXT PRIMARY KEY,
    file_data   BYTEA NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Approximate index for cosine similarity nearest neighbor search.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

