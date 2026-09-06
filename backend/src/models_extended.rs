use serde::Serialize;

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct DocumentInfo {
    pub filename: String,
    pub category: String,
    pub subdomain: Option<String>,
    pub chunk_count: i64,
    pub summary: String,
}
