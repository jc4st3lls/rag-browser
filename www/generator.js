// generator.js
//
// Local LLM inference in the browser with WebLLM (@mlc-ai/web-llm).
// Runs hardware-accelerated language models via WebGPU directly in the client.

import { CreateMLCEngine } from "@mlc-ai/web-llm";

let engine = null;
let initPromise = null;
let currentModel = null;

export async function loadGenerator(onProgress, modelName = "Llama-3.2-1B-Instruct-q4f16_1-MLC") {
  if (engine && currentModel === modelName) return engine;

  // If we already had an engine but it's a different model, reset and unload it
  if (engine && currentModel !== modelName) {
    console.info(`Switching model from ${currentModel} to ${modelName}...`);
    try {
      await engine.unload(); // release old model weights from GPU memory
    } catch (e) {
      console.warn("Could not unload previous engine:", e);
    }
    engine = null;
    initPromise = null;
    currentModel = null;
  }

  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      currentModel = modelName;
      engine = await CreateMLCEngine(modelName, {
        initProgressCallback: (report) => {
          console.log("[WebLLM Progress]", report.text);
          // report.progress is a value from 0 to 1
          onProgress?.({
            file: report.text || "model-weights",
            loaded: Math.round((report.progress || 0) * 100),
            total: 100,
          });
        },
      });

      console.info(`WebLLM: Engine initialized successfully with model: ${modelName}`);
      return engine;
    } catch (err) {
      console.error(`Error in loadGenerator (WebLLM) for model ${modelName}:`, err);
      initPromise = null;
      engine = null;
      currentModel = null;
      throw new Error(`Generator (WebLLM): ${err.message || err}`);
    }
  })();

  return await initPromise;
}

/**
 * Generates a streaming response from the chat messages.
 * Calls onToken for each generated text chunk.
 */
export async function generateStream(messages, onToken, onStatus, modelName = "Llama-3.2-1B-Instruct-q4f16_1-MLC") {
  onStatus?.("Starting LLM engine in browser...");
  const eng = await loadGenerator(onStatus, modelName);

  // Forzar la limpieza del historial interno del motor de chat de WebLLM
  // antes de cada pregunta. Esto asegura que no se arrastren contextos, prompts,
  // ni respuestas previas de sesiones anteriores, garantizando una sesión 100% limpia.
  try {
    await eng.resetChat();
  } catch (e) {
    console.warn("Could not reset WebLLM chat history:", e);
  }

  onStatus?.("Generating response in streaming...");
  const t0 = performance.now();

  const chunks = await eng.chat.completions.create({
    messages,
    temperature: 0.3,
    max_tokens: 2048,
    stream: true,
  });

  let full = "";
  let tokenCount = 0;

  for await (const chunk of chunks) {
    const delta = chunk.choices[0]?.delta?.content || "";
    if (delta) {
      full += delta;
      onToken?.(delta);
      tokenCount++;
    }
  }

  const totalTime = ((performance.now() - t0) / 1000).toFixed(2);
  const speed = tokenCount > 0 && parseFloat(totalTime) > 0 ? (tokenCount / parseFloat(totalTime)).toFixed(1) : "0";
  console.log(`[WebLLM] Generated ${tokenCount} chunks in ${totalTime}s (${speed} chunks/s) using model ${modelName}`);
  onStatus?.(`Completed in ${totalTime}s ✅`);

  return full;
}
