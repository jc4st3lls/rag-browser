// main.js
//
// Orchestrates the complete RAG flow on the client:
//   1. Embedding of the query (Candle/wasm, local)
//   2. Vector search against the backend (only sends the vector, not the
//      text — the backend is a "dumb" search engine)
//   3. Constructing the prompt with the retrieved context
//   4. Generating the response (WebLLM, local)

import { embedQuery, loadEmbedder } from "./embedder.js";
import { generateStream, loadGenerator } from "./generator.js";

const BACKEND_SEARCH_URL = "/api/search";
const BACKEND_INGEST_PDF_URL = "/api/ingest-pdf";

async function searchBackend(embedding, topK = 5) {
  let res;
  try {
    res = await fetch(BACKEND_SEARCH_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embedding: Array.from(embedding), top_k: topK }),
    });
  } catch (err) {
    throw new Error(
      `Could not connect to backend (${BACKEND_SEARCH_URL}). Check that the backend is running on http://localhost:8080: ${err.message}`
    );
  }

  if (!res.ok) {
    const errBody = await res.text().catch(() => "");
    throw new Error(`Backend search failed (${res.status}): ${errBody || res.statusText}`);
  }

  return res.json(); // expected: { chunks: [{ text, source, score }, ...] }
}

/**
 * Ingests a PDF file by directly sending binary bytes to the Backend.
 * PDF processing, chunking, and embedding generation
 * now happen 100% on the Rust server, guaranteeing maximum speed and freeing up the browser.
 */
export async function ingestPdf(file, { onProgress } = {}) {
  onProgress?.({ stage: "saving", current: 0, total: 1, text: `Sending "${file.name}" to server...` });

  const formData = new FormData();
  formData.append("file", file);

  let res;
  try {
    res = await fetch(BACKEND_INGEST_PDF_URL, {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    throw new Error(`Could not connect to backend (${BACKEND_INGEST_PDF_URL}): ${err.message}`);
  }

  if (!res.ok) {
    const errBody = await res.text().catch(() => "");
    throw new Error(`Backend PDF ingest failed (${res.status}): ${errBody || res.statusText}`);
  }

  const result = await res.json();
  return { totalChunks: result.inserted, inserted: result.inserted };
}

function buildPrompt(query, chunks) {
  const context = chunks
    .map((c, i) => `[${i + 1}] ${c.text}`)
    .join("\n\n");

  return [
    "Answer the question using only the provided context.",
    "If the context does not contain the answer, say so explicitly.",
    "",
    `Context:\n${context}`,
    "",
    `Question: ${query}`,
    "Response:",
  ].join("\n");
}

/** Preload both models with combined progress — call this on entering the app. */
export async function warmUp(onProgress, modelName = "Llama-3.2-1B-Instruct-q4f16_1-MLC") {
  await Promise.all([
    loadEmbedder((p) => onProgress?.({ ...p, stage: "embedder" })),
    loadGenerator((p) => onProgress?.({ ...p, stage: "generator" }), modelName),
  ]);
}

/** Complete flow: user question -> local embedding -> backend search -> local generation. */
export async function ask(query, { onToken, onStatus, onChunks, topK = 5, modelName = "Llama-3.2-1B-Instruct-q4f16_1-MLC" } = {}) {
  onStatus?.("1/3 Calculating local embedding (Candle WASM)...");
  const t0 = performance.now();
  const embedding = await embedQuery(query);
  const embedMs = (performance.now() - t0).toFixed(0);
  console.log(`[RAG] Embedding generated in ${embedMs}ms (dimension: ${embedding.length})`);

  onStatus?.("2/3 Querying vector database in backend (/api/search)...");
  const tBackend0 = performance.now();
  const { chunks } = await searchBackend(embedding, topK);
  const backendMs = (performance.now() - tBackend0).toFixed(0);
  console.log(`[RAG] Backend responded in ${backendMs}ms with ${chunks?.length || 0} chunks:`, chunks);

  // Immediately notify retrieved chunks to front-end
  onChunks?.(chunks || [], { embedMs, backendMs });

  if (!chunks?.length) {
    const msg = "No relevant chunks were found in the indexed documents to answer your question.";
    onToken?.(msg);
    onStatus?.("Search completed (no matches).");
    return msg;
  }

  onStatus?.(`3/3 ${chunks.length} chunks received from backend. Starting LLM generator...`);

  // Build the retrieved context
  const context = chunks
    .map((c, i) => `[Chunk ${i + 1} - ${c.source}]:\n${c.text}`)
    .join("\n\n");

  const messages = [
    {
      role: "system",
      content: `You are an expert, professional, and analytical assistant who answers in English. Your goal is to draft rich, exhaustive, structured, and highly professional answers based on the information in the Context.

Follow these professional guidelines:
1. Answer completely, in detail, and in-depth. Do not give one-sentence summarized answers if there is sufficient information.
2. Organize the answer using a structured format (bullet points, bold text, subheaders, or clear sections) to facilitate reading.
3. Mention the sources or pages of the chunks to add credibility to your analysis (e.g., "according to Chunk X of document Y...").
4. Ensure the tone is corporate, formal, and of expert consulting.
5. Use solely and exclusively the factual information contained in the provided Context. If the information is not sufficient to answer in detail, state it professionally.`,
    },
    {
      role: "user",
      content: `Context:\n${context}\n\nQuestion: ${query}`,
    },
  ];

  return generateStream(messages, onToken, onStatus, modelName);
}
