mod db;
mod embedder;
mod models;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use models::{IngestRequest, IngestResponse, SearchRequest, SearchResponse};

#[derive(Clone)]
struct AppState {
    pool: PgPool,
    embedder: Arc<embedder::LocalEmbedder>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let database_url = std::env::var("DATABASE_URL")
        .expect("Missing DATABASE_URL, e.g. postgres://postgres:postgres@localhost/ragdb");

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    // Load embedding model locally in the backend on startup
    tracing::info!("Loading local embedding model (BGE-small-en)...");
    let embedder = Arc::new(embedder::LocalEmbedder::load_default().expect("Could not load the embedder"));
    tracing::info!("Embedding model loaded successfully");

    let state = AppState { pool, embedder };

    // Run hot migrations if the original_documents table does not exist
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS original_documents (
            filename    TEXT PRIMARY KEY,
            file_data   BYTEA NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        "#
    )
    .execute(&state.pool)
    .await
    .expect("Could not create original_documents table");

    // In dev, the frontend runs on another port (vite) -> open CORS.
    // In production, restrict `allow_origin` to your real domain.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Configure layer to allow larger HTTP request bodies (e.g., 200MB for giant PDFs)
    let body_limit = axum::extract::DefaultBodyLimit::max(200 * 1024 * 1024); // 200 Megabytes

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/search", post(search))
        .route("/api/ingest", post(ingest))
        .route("/api/ingest-pdf", post(ingest_pdf))
        .route("/api/document/{name}", get(get_document))
        .layer(cors)
        .layer(body_limit)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Backend listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    if req.embedding.is_empty() {
        return Err(AppError::BadRequest("empty embedding".into()));
    }

    println!("--> [POST /api/search] Searching top_{} for embedding (dim: {})", req.top_k, req.embedding.len());
    let chunks = db::search_similar(&state.pool, req.embedding, req.top_k.clamp(1, 50)).await?;
    println!("<-- [POST /api/search] {} relevant chunks found", chunks.len());

    Ok(Json(SearchResponse { chunks }))
}

async fn ingest(
    State(state): State<AppState>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, AppError> {
    if req.documents.is_empty() {
        return Err(AppError::BadRequest("no documents sent".into()));
    }

    println!("--> [POST /api/ingest] Received {} chunks to index", req.documents.len());

    for (i, doc) in req.documents.iter().enumerate() {
        if doc.embedding.is_empty() {
            return Err(AppError::BadRequest(format!(
                "empty embedding in document {}",
                i + 1
            )));
        }
    }

    let count = db::insert_documents(&state.pool, req.documents).await?;
    println!("<-- [POST /api/ingest] {} chunks successfully saved", count);
    Ok(Json(IngestResponse { inserted: count }))
}

/// Endpoint to upload a PDF directly, process it, and index it into the Postgres database.
async fn ingest_pdf(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<IngestResponse>, AppError> {
    let mut filename = "document.pdf".to_string();
    let mut pdf_data = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| AppError::BadRequest(e.to_string()))? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            filename = field.file_name().unwrap_or("document.pdf").to_string();
            pdf_data = field.bytes().await.map_err(|e| AppError::Internal(e.into()))?.to_vec();
        }
    }

    if pdf_data.is_empty() {
        return Err(AppError::BadRequest("No file was sent or the file is empty".into()));
    }

    println!("--> [POST /api/ingest-pdf] Processing uploaded file: '{}' ({} bytes)", filename, pdf_data.len());

    // Save physical PDF to disk volume assigned to backend
    let files_dir = std::env::var("FILES_DIR").unwrap_or_else(|_| "/data/files".to_string());
    let _ = tokio::fs::create_dir_all(&files_dir).await;
    let file_path = std::path::Path::new(&files_dir).join(&filename);
    if let Err(e) = tokio::fs::write(&file_path, &pdf_data).await {
        tracing::warn!("Could not write file to disk ({:?}): {}", file_path, e);
    } else {
        tracing::info!("File saved in data volume: {:?}", file_path);
    }

    // Save original PDF to database for persistence and download
    sqlx::query(
        r#"
        INSERT INTO original_documents (filename, file_data)
        VALUES ($1, $2)
        ON CONFLICT (filename) DO UPDATE SET file_data = EXCLUDED.file_data
        "#
    )
    .bind(&filename)
    .bind(&pdf_data)
    .execute(&state.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    // 1. Extract text by pages
    let pages = embedder::extract_pdf_pages(&pdf_data)
        .map_err(|e| AppError::BadRequest(format!("Error extracting text from PDF: {}", e)))?;

    if pages.is_empty() {
        return Err(AppError::BadRequest("The PDF document contains no extractable text.".into()));
    }

    // Classify document by joining text from the first pages to get thematic content
    let classification_sample: String = pages.iter().take(3).map(|(_, t)| t.clone()).collect::<Vec<_>>().join(" ");
    let (category, subdomain) = embedder::classify_text(&classification_sample);
    println!("--> Document classification: Category='{}', Subdomain='{:?}'", category, subdomain);

    // 2. Fragment into chunks
    let chunks = embedder::chunk_text(&pages, 450, 80);
    let mut docs_to_ingest = Vec::new();

    // 3. Generate embeddings and index
    for (page_num, chunk_text) in chunks {
        let embedding = state.embedder.embed(&chunk_text)
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Error generating embedding: {}", e)))?;

        docs_to_ingest.push(models::IngestDocument {
            source: format!("{} (page {})", filename, page_num), // Store with page in source for the chunk
            chunk_text,
            embedding,
            category: Some(category.clone()),
            subdomain: subdomain.clone(),
        });
    }

    let count = db::insert_documents(&state.pool, docs_to_ingest).await?;
    println!("<-- [POST /api/ingest-pdf] PDF processed and indexed successfully ({} chunks saved)", count);

    Ok(Json(IngestResponse { inserted: count }))
}

/// Endpoint to download or preview the original PDF.
async fn get_document(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Response, AppError> {
    // 1. Try to serve from disk volume
    let files_dir = std::env::var("FILES_DIR").unwrap_or_else(|_| "/data/files".to_string());
    let file_path = std::path::Path::new(&files_dir).join(&name);
    if let Ok(bytes) = tokio::fs::read(&file_path).await {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/pdf")
            .header("Content-Disposition", format!("inline; filename=\"{}\"", name))
            .body(axum::body::Body::from(bytes))
            .map_err(|e| AppError::Internal(e.into()))?;
        return Ok(response);
    }

    // 2. Fallback to database
    let row: Option<(Vec<u8>,)> = sqlx::query_as(
        "SELECT file_data FROM original_documents WHERE filename = $1"
    )
    .bind(&name)
    .fetch_optional(&state.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    if let Some(r) = row {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/pdf")
            .header("Content-Disposition", format!("inline; filename=\"{}\"", name))
            .body(axum::body::Body::from(r.0))
            .map_err(|e| AppError::Internal(e.into()))?;
        Ok(response)
    } else {
        Err(AppError::BadRequest("Document not found".into()))
    }
}

/// Unified error mapping to readable HTTP codes for the client WASM.
enum AppError {
    BadRequest(String),
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            AppError::Internal(e) => {
                tracing::error!("internal error: {e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".to_string())
            }
        };
        (status, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}
