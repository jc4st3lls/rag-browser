use anyhow::Result;
use pgvector::Vector;
use sqlx::PgPool;

use crate::models::IngestDocument;

/// Inserts a batch of chunks with their embeddings into the database.
pub async fn insert_documents(
    pool: &PgPool,
    docs: Vec<IngestDocument>,
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

/// Saves the original PDF data to original_documents table for persistence/fallback.
pub async fn insert_original_document(
    pool: &PgPool,
    filename: &str,
    file_data: &[u8],
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO original_documents (filename, file_data)
        VALUES ($1, $2)
        ON CONFLICT (filename) DO UPDATE SET file_data = EXCLUDED.file_data
        "#,
    )
    .bind(filename)
    .bind(file_data)
    .execute(pool)
    .await?;

    Ok(())
}

/// Elimina todos los chunks de la base de datos y la versión original de un documento anterior
/// que coincidan con el nombre del archivo para evitar basura o registros duplicados.
pub async fn delete_document_data(pool: &PgPool, filename: &str) -> Result<()> {
    let mut tx = pool.begin().await?;

    // Eliminamos los chunks de documents cuyo 'source' comience por el nombre de archivo
    let file_prefix = format!("{}%", filename);
    sqlx::query(
        r#"
        DELETE FROM documents
        WHERE source LIKE $1
        "#,
    )
    .bind(&file_prefix)
    .execute(&mut *tx)
    .await?;

    // Eliminamos la versión guardada en original_documents
    sqlx::query(
        r#"
        DELETE FROM original_documents
        WHERE filename = $1
        "#,
    )
    .bind(filename)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(())
}
