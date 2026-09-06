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
      content: `You are an expert and precise assistant. Your goal is to answer the user's question thoroughly and accurately based ONLY on the provided Context.

Guidelines:
1. Answer the question in the same language the user uses (e.g. Spanish if asked in Spanish, English if asked in English).
2. Use solely and exclusively the facts, data, and explanations present in the provided Context. Organize the response clearly with bullet points and bold text where appropriate.
3. Do not invent any facts, do not make assumptions, and do not use outside knowledge.
4. If the provided Context does not contain information to answer the question, state clearly that the indexed documents do not contain that information.`,
    },
    {
      role: "user",
      content: `Context:\n${context}\n\nQuestion: ${query}`,
    },
  ];

  return generateStream(messages, onToken, onStatus, modelName);
}
