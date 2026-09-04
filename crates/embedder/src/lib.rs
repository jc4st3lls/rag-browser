//! Text embedder for the browser, compiled to WASM with Candle.
//!
//! Intended for small, multilingual BERT-like models (bge-small,
//! gte-small, bge-m3...) downloaded in safetensors + config.json
//! + tokenizer.json format from Hugging Face.
//!
//! Weights are NOT downloaded from Rust: the JS in `www/` downloads them
//! (with caching via Cache API, see modelCache.js) and passes the bytes
//! already in memory to `Embedder::load`. This keeps the network/cache
//! logic in JS, which is where it naturally belongs in a browser.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    // Translates Rust panics into readable messages in the browser console
    // instead of an opaque "unreachable executed".
    console_error_panic_hook::set_once();
}

#[derive(serde::Deserialize)]
struct ConfigHelper {
    #[serde(default = "default_hidden_size")]
    hidden_size: usize,
}

fn default_hidden_size() -> usize {
    384
}

#[wasm_bindgen]
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hidden_size: usize,
}

#[wasm_bindgen]
impl Embedder {
    /// Loads the model from bytes already downloaded by JS.
    ///
    /// - `weights`: contents of the .safetensors file
    /// - `tokenizer_json`: contents of tokenizer.json (as a string)
    /// - `config_json`: contents of config.json (as a string)
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: &[u8],
        tokenizer_json: &str,
        config_json: &str,
    ) -> Result<Embedder, JsValue> {
        let device = Device::Cpu; // in WASM always CPU (candle has no wgpu backend here)

        let config: BertConfig =
            serde_json::from_str(config_json).map_err(|e| js_err(format!("config: {e}")))?;

        let helper: ConfigHelper =
            serde_json::from_str(config_json).unwrap_or(ConfigHelper { hidden_size: 384 });
        let hidden_size = helper.hidden_size;

        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .map_err(|e| js_err(format!("tokenizer: {e}")))?;

        // safetensors in memory (no filesystem in the browser)
        let vb = VarBuilder::from_buffered_safetensors(weights.to_vec(), DTYPE, &device)
            .map_err(|e| js_err(format!("safetensors: {e}")))?;

        let model = BertModel::load(vb, &config).map_err(|e| js_err(format!("bert load: {e}")))?;

        Ok(Embedder {
            model,
            tokenizer,
            device,
            hidden_size,
        })
    }

    /// Returns the L2-normalized embedding of a text as a Float32Array.
    /// Uses mean pooling over the last hidden layer, the standard approach
    /// for bge/gte-like models used as sentence encoders.
    #[wasm_bindgen]
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, JsValue> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| js_err(format!("encode: {e}")))?;

        let ids = encoding.get_ids();
        let token_ids = Tensor::new(ids, &self.device)
            .map_err(|e| js_err(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| js_err(e.to_string()))?;

        let token_type_ids = token_ids
            .zeros_like()
            .map_err(|e| js_err(e.to_string()))?;

        let output = self
            .model
            .forward(&token_ids, &token_type_ids, None)
            .map_err(|e| js_err(format!("forward: {e}")))?;

        // Mean pooling over the tokens dimension (dim=1)
        let (_n_batch, n_tokens, _hidden) =
            output.dims3().map_err(|e| js_err(e.to_string()))?;
        let pooled = (output.sum(1).map_err(|e| js_err(e.to_string()))? / (n_tokens as f64))
            .map_err(|e| js_err(e.to_string()))?;

        // L2 normalization, essential for cosine similarity in
        // the backend to be correct (or to use direct dot product).
        let norm = pooled
            .sqr()
            .map_err(|e| js_err(e.to_string()))?
            .sum_keepdim(1)
            .map_err(|e| js_err(e.to_string()))?
            .sqrt()
            .map_err(|e| js_err(e.to_string()))?;
        let normalized = pooled.broadcast_div(&norm).map_err(|e| js_err(e.to_string()))?;

        let vec: Vec<f32> = normalized
            .squeeze(0)
            .map_err(|e| js_err(e.to_string()))?
            .to_dtype(DType::F32)
            .map_err(|e| js_err(e.to_string()))?
            .to_vec1()
            .map_err(|e| js_err(e.to_string()))?;

        Ok(vec)
    }

    /// Output vector dimension, useful for configuring the vector database
    /// in the backend (e.g. pgvector needs to know the column size).
    #[wasm_bindgen]
    pub fn dim(&self) -> usize {
        self.hidden_size
    }
}

fn js_err(msg: String) -> JsValue {
    JsValue::from_str(&msg)
}
