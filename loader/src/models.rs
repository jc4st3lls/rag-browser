use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestDocument {
    pub source: String,
    pub chunk_text: String,
    pub embedding: Vec<f32>,
    pub category: Option<String>,
    pub subdomain: Option<String>,
}
