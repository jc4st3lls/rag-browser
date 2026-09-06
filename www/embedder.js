// embedder.js
//
// Carga el paquete wasm generado por `wasm-pack build crates/embedder
// --target web` y expone una función de alto nivel para obtener el
// embedding de una query, usando modelCache.js para no re-descargar.

import init, { Embedder } from "../crates/embedder/pkg/embedder.js";
import embedderWasmUrl from "../crates/embedder/pkg/embedder_bg.wasm?url";
import { fetchWithCache, fetchTextWithCache } from "./modelCache.js";

// Ejemplo con un modelo multilingüe de alto rendimiento y 384 dimensiones
// que permite buscar en español, catalán, inglés, etc. sin alterar la base de datos.
const MODEL_BASE =
  "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main";

let embedderInstance = null;
let embedderPromise = null;

export async function loadEmbedder(onProgress) {
  if (embedderInstance) return embedderInstance;
  if (embedderPromise) return embedderPromise;

  embedderPromise = (async () => {
    try {
      // Inicializa el runtime wasm con la URL correcta gestionada por Vite.
      await init(embedderWasmUrl);

      const [weights, tokenizerJson, configJson] = await Promise.all([
        fetchWithCache(`${MODEL_BASE}/model.safetensors`, (loaded, total) =>
          onProgress?.({ file: "weights", loaded, total })
        ),
        fetchTextWithCache(`${MODEL_BASE}/tokenizer.json`),
        fetchTextWithCache(`${MODEL_BASE}/config.json`),
      ]);

      embedderInstance = new Embedder(new Uint8Array(weights), tokenizerJson, configJson);
      return embedderInstance;
    } catch (err) {
      console.error("Error in loadEmbedder:", err);
      embedderPromise = null;
      throw new Error(`Embedder (Candle BERT): ${err.message || err}`);
    }
  })();

  return await embedderPromise;
}

/** Devuelve un Float32Array normalizado, listo para enviar al backend. */
export async function embedQuery(text) {
  const embedder = await loadEmbedder();
  return embedder.embed(text);
}
