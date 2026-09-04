# RAG en el navegador (Rust + WASM)

Esqueleto de RAG donde tanto la interpretación de la query (embeddings)
como la generación de la respuesta corren **enteramente en el cliente**.
El backend solo hace de motor de búsqueda vectorial: recibe un embedding,
devuelve los chunks más relevantes, y no toca ningún LLM.

## Arquitectura

```
Usuario escribe query
      │
      ▼
[embedder.js]  → wasm de Candle (crates/embedder) → vector embedding
      │
      ▼
POST /api/search  { embedding, top_k }   ← único tráfico de red por turno
      │
      ▼
Backend: búsqueda kNN en la BD vectorial → { chunks }
      │
      ▼
[main.js] construye el prompt con los chunks recuperados
      │
      ▼
[generator.js] → flarellm (GGUF, WebGPU/SIMD) → respuesta en streaming
```

Los pesos de ambos modelos se descargan una vez y se cachean en el
navegador vía Cache API (`www/modelCache.js`); en visitas posteriores no
hay descarga, solo lectura de disco local.

## Por qué no hay un crate "generator" en Rust

`flarellm` ya se compila y publica como paquete wasm+JS
(`@sauravpanda/flare`). Envolverlo en otro crate Rust nuestro añadiría una
capa de indirección sin beneficio — se consume directamente desde JS,
igual que harías con cualquier motor wasm de terceros. El único crate
Rust propio de este proyecto es `embedder`, porque ahí sí queremos control
total sobre el modelo de embeddings y su pooling/normalización.

## Despliegue con Docker Compose (Recomendado)

El proyecto incluye un entorno multicontenedor completo con inicialización automática.

⚠️ **NOTA CRÍTICA PARA EL DESPLIEGUE EN DOCKER**:
Para evitar compilar el compilador de Rust, `cargo` y `wasm-pack` dentro de la imagen del frontend (lo cual ralentiza el despliegue y genera errores de compilación de dependencias del sandbox de Docker), **el frontend se construye asumiendo que el paquete WASM del embedder ya ha sido precompilado en tu máquina local**.

Por lo tanto, **SIEMPRE debes compilar el WASM localmente antes de levantar o reconstruir Docker Compose**:

```bash
# 1. Compilar el paquete WASM localmente en tu host (se realiza en 1 segundo):
cd crates/embedder
wasm-pack build --target web --release
cd ../..

# 2. Levantar el despliegue con Docker Compose:
docker compose up -d --build
```

### Servicios incluidos:
1. **`db` (`pgvector/pgvector:pg16`)**:
   - Inicializa automáticamente la extensión `vector` y las tablas a partir de `backend/migrations/001_init.sql`.
   - Volumen persistente en tu máquina local mapeado a: `${HOME}/Data/rag-browser/database`.
2. **`backend` (Rust / Axum + Candle)**:
   - Extrae texto de PDFs de forma asíncrona, clasifica temáticamente el documento usando la taxonomía de la aplicación y calcula los embeddings vectoriales en el servidor de forma instantánea.
   - Volumen persistente en tu máquina para almacenar los PDFs originales y las cachés del modelo: `${HOME}/Data/rag-browser/files`.
3. **`frontend` (Nginx + WebLLM + Candle WASM)**:
   - Servido en los puertos `http://localhost:3000` (o `http://localhost:80`).
   - Posee un archivo de configuración de Nginx (`www/nginx.conf`) que realiza un proxy inverso automático para que las consultas `/api/` apunten de forma transparente al contenedor del backend, evitando problemas de CORS en producción y permitiendo subidas de PDFs pesados de hasta 200MB.

---

## Build Local (Desarrollo sin Docker)

## Modelos usados por defecto (cambiables)

- **Embeddings**: `BAAI/bge-small-en-v1.5` — cámbialo por `bge-m3` si
  necesitas mejor cobertura multilingüe (catalán/español/inglés mezclados).
- **Generación**: `Qwen2.5-1.5B-Instruct` cuantizado Q4_K_M en GGUF —
  buena relación tamaño/calidad y buen soporte multilingüe. Alternativas:
  SmolLM2, Llama-3.2-1B/3B-Instruct, Gemma2-2B.

## Backend (incluido: axum + pgvector)

El directorio `backend/` es un backend mínimo en Rust que expone
exactamente el contrato que espera `www/main.js`:

```
POST /api/search
Body:  { "embedding": [f32...], "top_k": 5 }
Resp:  { "chunks": [{ "text": "...", "source": "...", "score": 0.87 }] }
```

No contiene ningún LLM ni lógica de generación — solo hace kNN sobre
Postgres+pgvector. Podrías sustituirlo por Qdrant, Weaviate,
sqlite-vec... el contrato HTTP con el cliente no cambiaría.

### Levantar el backend

```bash
# 1. Postgres con pgvector (vía Docker, más rápido para probar)
docker run -d --name rag-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb \
  pgvector/pgvector:pg16

# 2. Aplicar la migración
psql postgres://postgres:postgres@localhost:5432/ragdb \
  -f backend/migrations/001_init.sql

# 3. (opcional) poblar con datos de prueba
pip install sentence-transformers psycopg2-binary
python backend/ingest_example.py

# 4. Arrancar el servidor
cd backend
cp .env.example .env
cargo run
```

El servidor queda escuchando en `http://localhost:8080`. Ajusta
`BACKEND_SEARCH_URL` en `www/main.js` si no usas un proxy que lo sirva
bajo el mismo origen que el frontend.

⚠️ **La dimensión del vector debe coincidir** entre `crates/embedder`
(`Embedder::dim()`), la columna `vector(384)` de la migración, y el
modelo usado en `ingest_example.py`. Si cambias de modelo de embeddings,
actualiza los tres sitios.

## Notas y advertencias

- **flarellm es un proyecto joven**: verifica los nombres de métodos
  (`load`, `init_gpu`, `begin_stream`, `next_token`) contra su versión
  actual antes de desplegar — pueden cambiar entre releases.
- **Tamaño de descarga inicial**: el modelo GGUF de 1.5B en Q4 ronda
  ~1GB. Considera avisar al usuario y mostrar progreso (ya incluido en
  `index.html`) antes de la primera carga.
- **Fallback sin WebGPU**: Safari/navegadores antiguos caerán a SIMD por
  CPU, más lento pero funcional. Comprueba `navigator.gpu` si quieres
  degradar la UX conscientemente (p.ej. avisar que la primera respuesta
  tardará más).
