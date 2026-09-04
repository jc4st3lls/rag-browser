use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Already normalized embedding (L2), generated on the client by
    /// `crates/embedder`. The backend never sees the original text of the
    /// query, only the vector.
    pub embedding: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: i64,
}

fn default_top_k() -> i64 {
    5
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Serialize)]
pub struct Chunk {
    pub text: String,
    pub source: String,
    /// Cosine similarity (1.0 = identical, 0.0 = orthogonal). Calculated as
    /// 1 - cosine_distance, see query in db.rs.
    pub score: f32,
    pub category: String,
    pub subdomain: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IngestDocument {
    pub source: String,
    pub chunk_text: String,
    pub embedding: Vec<f32>,
    pub category: Option<String>,
    pub subdomain: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub documents: Vec<IngestDocument>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub inserted: usize,
}
