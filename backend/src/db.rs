use anyhow::Result;
use pgvector::Vector;
use sqlx::{FromRow, PgPool};

use crate::models::Chunk;

#[derive(FromRow)]
struct SearchRow {
    chunk_text: String,
    source: String,
    category: String,
    subdomain: Option<String>,
    score: Option<f64>,
}

/// Searches for the `top_k` most similar chunks to the given embedding, using
/// cosine distance (the `<=>` operator from pgvector). Requires that the
/// `embedding` column has the `vector_cosine_ops` index from the
/// migration 001_init.sql for high performance.
pub async fn search_similar(pool: &PgPool, embedding: Vec<f32>, top_k: i64) -> Result<Vec<Chunk>> {
    let query_vec = Vector::from(embedding);

    // `<=>` returns cosine distance (0 = identical, 2 = opposite);
    // we convert it to a more intuitive similarity "score" (1 = identical).
    let rows: Vec<SearchRow> = sqlx::query_as(
        r#"
        SELECT
            chunk_text,
            source,
            category,
            subdomain,
            1 - (embedding <=> $1) AS score
        FROM documents
        ORDER BY embedding <=> $1
        LIMIT $2
        "#,
    )
    .bind(query_vec)
    .bind(top_k)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| Chunk {
            text: r.chunk_text,
            source: r.source,
            score: r.score.unwrap_or(0.0) as f32,
            category: r.category,
            subdomain: r.subdomain,
        })
        .collect())
}

/// Inserts a batch of chunks with their embeddings into the database.
pub async fn insert_documents(
    pool: &PgPool,
    docs: Vec<crate::models::IngestDocument>,
) -> Result<usize> {
    if docs.is_empty() {
        return Ok(0);
    }

    let mut count = 0;
    let mut tx = pool.begin().await?;

    for doc in docs {
        let vec = Vector::from(doc.embedding);
        let category = doc.category.unwrap_or_else(|| "Other".to_string());
        sqlx::query(
            r#"
            INSERT INTO documents (source, chunk_text, embedding, category, subdomain)
            VALUES ($1, $2, $3, $4, $5)
            "#,
        )
        .bind(doc.source)
        .bind(doc.chunk_text)
        .bind(vec)
        .bind(category)
        .bind(doc.subdomain)
        .execute(&mut *tx)
        .await?;
        count += 1;
    }

    tx.commit().await?;
    Ok(count)
}
