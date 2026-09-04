// modelCache.js
//
// Descarga y cachea artefactos de modelo (pesos, tokenizer, config...) en
// el navegador usando la Cache API, para no volver a descargarlos en
// visitas posteriores. Compartido tanto por el embedder (Candle) como
// por el generador (flarellm).

const CACHE_NAME = "rag-models-v2";

function isHtmlResponse(bytes) {
  if (bytes.length >= 4) {
    // Check for "<!do" or "<htm"
    const header = new TextDecoder().decode(bytes.subarray(0, 15)).trim().toLowerCase();
    if (header.startsWith("<!doctype") || header.startsWith("<html")) {
      return true;
    }
  }
  return false;
}

/**
 * Descarga (con caché) una URL y devuelve un ArrayBuffer.
 * @param {string} url
 * @param {(loaded: number, total: number) => void} [onProgress]
 */
export async function fetchWithCache(url, onProgress) {
  let cache = null;
  try {
    if (typeof caches !== "undefined") {
      cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        const buf = await cached.arrayBuffer();
        // Validar que no sea un HTML corrupto cacheado previamente
        if (!isHtmlResponse(new Uint8Array(buf))) {
          return buf;
        }
        console.warn(`Invalid cache for ${url}, re-downloading...`);
        await cache.delete(url);
      }
    }
  } catch (e) {
    console.warn("Could not read from Cache API:", e);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not download ${url}: ${response.status} ${response.statusText}`);
  }

  const total = Number(response.headers.get("content-length") ?? 0);
  let loaded = 0;
  const reader = response.body.getReader();
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  if (isHtmlResponse(buffer)) {
    throw new Error(`The URL ${url} returned an HTML document instead of model binary data.`);
  }

  if (cache) {
    try {
      await cache.put(url, new Response(buffer.slice(0)));
    } catch (e) {
      console.warn("Could not persist in Cache API (quota exceeded or not supported):", e);
    }
  }

  return buffer.buffer;
}

/** Descarga como texto (para tokenizer.json / config.json) con la misma caché. */
export async function fetchTextWithCache(url) {
  const buf = await fetchWithCache(url);
  return new TextDecoder("utf-8").decode(buf);
}

/** Borra toda la caché de modelos. */
export async function clearModelCache() {
  try {
    if (typeof caches !== "undefined") {
      const keys = await caches.keys();
      for (const key of keys) {
        if (key.startsWith("rag-models")) {
          await caches.delete(key);
        }
      }
    }
  } catch (e) {
    console.warn("Error clearing cache:", e);
  }
}
