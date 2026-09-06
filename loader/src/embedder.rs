//! PDF extractor and local embedding generator using Candle.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::HFClientSync;
use lopdf::Document as PdfDocument;
use tokenizers::Tokenizer;

pub struct LocalEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl LocalEmbedder {
    /// Downloads from Hugging Face and loads the multilingual embedding model.
    pub fn load_default() -> Result<Self> {
        let device = Device::Cpu; // Run on CPU
        let client = HFClientSync::new()?;
        let repo = client.model("sentence-transformers", "paraphrase-multilingual-MiniLM-L12-v2");

        // Download required files
        let weights_path = repo.download_file().filename("model.safetensors").send()?;
        let tokenizer_path = repo.download_file().filename("tokenizer.json").send()?;
        let config_path = repo.download_file().filename("config.json").send()?;

        // Load configuration
        let config_str = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Error loading tokenizer: {}", e))?;

        // Load weights using VarBuilder
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(weights_path)?,
            DTYPE,
            &device,
        )?;
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Generates the normalized embedding vector (L2) for a text chunk.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Error encoding text: {}", e))?;

        let ids = encoding.get_ids();
        let token_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let output = self.model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling
        let (_n_batch, n_tokens, _hidden) = output.dims3()?;
        let pooled = (output.sum(1)? / (n_tokens as f64))?;

        // L2 normalization
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        let vec: Vec<f32> = normalized
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()?;

        Ok(vec)
    }
}

#[cfg(unix)]
struct Silence {
    original_stdout: std::os::unix::io::RawFd,
    original_stderr: std::os::unix::io::RawFd,
}

#[cfg(unix)]
impl Silence {
    fn new() -> Option<Self> {
        use std::os::unix::io::AsRawFd;
        let null_file = std::fs::OpenOptions::new().write(true).open("/dev/null").ok()?;
        let null_fd = null_file.as_raw_fd();
        
        let original_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
        let original_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };
        
        if original_stdout < 0 || original_stderr < 0 {
            if original_stdout >= 0 { unsafe { libc::close(original_stdout); } }
            if original_stderr >= 0 { unsafe { libc::close(original_stderr); } }
            return None;
        }
        
        if unsafe { libc::dup2(null_fd, libc::STDOUT_FILENO) } < 0 {
            unsafe {
                libc::close(original_stdout);
                libc::close(original_stderr);
            }
            return None;
        }
        
        if unsafe { libc::dup2(null_fd, libc::STDERR_FILENO) } < 0 {
            unsafe {
                let _ = libc::dup2(original_stdout, libc::STDOUT_FILENO);
                libc::close(original_stdout);
                libc::close(original_stderr);
            }
            return None;
        }
        
        Some(Self {
            original_stdout,
            original_stderr,
        })
    }
}

#[cfg(unix)]
impl Drop for Silence {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::dup2(self.original_stdout, libc::STDOUT_FILENO);
            libc::close(self.original_stdout);
            
            let _ = libc::dup2(self.original_stderr, libc::STDERR_FILENO);
            libc::close(self.original_stderr);
        }
    }
}

/// Extracts text from a PDF in memory and splits it into pages.
pub fn extract_pdf_pages(bytes: &[u8]) -> Result<Vec<(usize, String)>> {
    // Silenciar cualquier advertencia interna o println! emitido directamente
    // por la biblioteca lopdf que ensucie la consola o la barra de progreso.
    #[cfg(unix)]
    let _silence = Silence::new();

    let doc = PdfDocument::load_mem(bytes)
        .map_err(|e| anyhow!("Could not decode PDF: {}", e))?;
    
    let mut pages = Vec::new();
    
    for page_num in 1..=doc.get_pages().len() {
        if let Ok(text) = doc.extract_text(&[page_num as u32]) {
            let cleaned = text
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" ");

            // Replace multiple whitespaces with a single space natively in Rust
            let mut normalized = String::new();
            let mut last_was_space = false;
            for c in cleaned.chars() {
                if c.is_whitespace() {
                    if !last_was_space {
                        normalized.push(' ');
                        last_was_space = true;
                    }
                } else {
                    normalized.push(c);
                    last_was_space = false;
                }
            }

            let normalized = normalized.trim().to_string();

            if !normalized.is_empty() {
                pages.push((page_num, normalized));
            }
        }
    }
    
    Ok(pages)
}

/// Splits text into logical chunks with overlap, correctly handling UTF-8 characters.
pub fn chunk_text(pages: &[(usize, String)], chunk_size: usize, overlap: usize) -> Vec<(usize, String)> {
    let mut chunks = Vec::new();

    for &(page_num, ref text) in pages {
        // In Rust, char_indices() gives us the exact byte position of the unicode character
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let total_chars = chars.len();

        if total_chars <= chunk_size {
            chunks.push((page_num, text.clone()));
            continue;
        }

        let mut start_idx = 0;
        while start_idx < total_chars {
            let mut end_idx = start_idx + chunk_size;
            if end_idx < total_chars {
                // Try to find the last space inside the character slice to cut cleanly
                let sub_slice = &chars[start_idx..end_idx];
                if let Some(space_pos) = sub_slice.iter().rposition(|&(_, c)| c == ' ') {
                    if space_pos > chunk_size / 2 {
                        end_idx = start_idx + space_pos;
                    }
                }
            } else {
                end_idx = total_chars;
            }

            // Obtain actual byte offsets corresponding to the character unicode positions
            let byte_start = chars[start_idx].0;
            let byte_end = if end_idx < total_chars {
                chars[end_idx].0
            } else {
                text.len()
            };

            let chunk_content = text[byte_start..byte_end].trim().to_string();
            if !chunk_content.is_empty() {
                chunks.push((page_num, chunk_content));
            }

            if end_idx >= total_chars {
                break;
            }

            // Move the index based on the character overlap
            if end_idx > overlap {
                start_idx = end_idx - overlap;
            } else {
                start_idx = end_idx;
            }
        }
    }

    chunks
}

/// Classifies the text thematically based on keywords and the provided taxonomy.
/// Returns a tuple of (Category, Optional Subdomain)
pub fn classify_text(text: &str) -> (String, Option<String>) {
    let lower = text.to_lowercase();
    
    // Taxonomy based on the brains/subdomains JSON
    let taxonomies = vec![
        // 1. Technology, Science and Computing
        ("AI", "Technology, Science and Computing", vec!["artificial intelligence", "machine learning", "deep learning", "neural networks", "neural", "nlp", "llm", "algorithm", "data science"]),
        ("DEV", "Technology, Science and Computing", vec!["software", "development", "programming", "rust", "python", "c++", "compile", "refactor", "api", "git", "github", "bug", "debugging", "cmake"]),
        ("SEC", "Technology, Science and Computing", vec!["cybersecurity", "security", "exploits", "vulnerabilities", "malware", "virus", "encryption", "kernel", "network", "hacking", "firewall", "networks"]),
        ("SCI", "Technology, Science and Computing", vec!["mathematics", "physics", "chemistry", "engineering", "logic", "statistics", "civil", "electrical", "mechanical", "calculus"]),
        
        // 2. Health, Biology and Wellbeing
        ("HCS", "Health, Biology and Wellbeing", vec!["clinical", "hospital operations", "medical records", "digital health", "ehr", "hospital", "healthcare", "medical", "clinic"]),
        ("MED", "Health, Biology and Wellbeing", vec!["medicine", "anatomy", "physiology", "disease", "treatment", "surgery", "patient", "doctor", "drug", "health", "surgical", "symptom"]),
        ("BIO", "Health, Biology and Wellbeing", vec!["biology", "genetics", "ecology", "earth", "geology", "meteorology", "sustainability", "environment", "planet", "species", "biodiversity"]),
        
        // 3. Society, Economy and Governance
        ("FIN", "Society, Economy and Governance", vec!["finance", "economy", "stock market", "bank", "investment", "inflation", "gdp", "spending", "money", "microeconomics", "macroeconomics"]),
        ("DLT", "Society, Economy and Governance", vec!["blockchain", "web3", "smart contract", "bitcoin", "ethereum", "cryptocurrency", "defi", "ledger", "transactions"]),
        ("LAW", "Society, Economy and Governance", vec!["law", "regulation", "statute", "courts", "patent", "copyright", "legal", "judge", "compliance"]),
        ("SOC", "Society, Economy and Governance", vec!["geopolitics", "politics", "history", "sociology", "democracy", "government", "culture", "society", "state", "nation"]),
        
        // 4. Strategy, Business and Productivity
        ("BUS", "Society, Economy and Governance", vec!["strategy", "operation", "startup", "revenue", "marketing", "sales", "business", "company", "corporate", "revenues"]),
        ("PM", "Society, Economy and Governance", vec!["project", "scrum", "kanban", "agile", "schedule", "resource", "milestone", "scrummaster", "planning"]),
        ("PROD", "Society, Economy and Governance", vec!["productivity", "time", "learning", "habits", "focus", "meta-knowledge", "concentration"]),
        
        // 5. Humanities, Culture and Communication
        ("PHIL", "Humanities, Culture and Communication", vec!["philosophy", "ethics", "epistemology", "metaphysics", "morals", "logic", "philosopher"]),
        ("LANG", "Humanities, Culture and Communication", vec!["linguistics", "language", "grammar", "translation", "syntax", "vocabulary", "languages"]),
        ("ART", "Humanities, Culture and Communication", vec!["art", "music", "painting", "design", "architecture", "cinema", "movie", "fine arts"]),
        ("LIFE", "Humanities, Culture and Communication", vec!["travel", "leisure", "gastronomy", "food", "hobby", "tourism", "recipe", "restaurant"]),
    ];

    let mut best_subdomain = None;
    let mut best_category = "Other".to_string();
    let mut max_matches = 0;

    for (code, category, kw_list) in taxonomies {
        let mut matches = 0;
        for kw in kw_list {
            if lower.contains(kw) {
                matches += 1;
            }
        }
        if matches > max_matches {
            max_matches = matches;
            best_subdomain = Some(code.to_string());
            best_category = category.to_string();
        }
    }

    (best_category, best_subdomain)
}
