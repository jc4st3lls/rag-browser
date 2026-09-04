"""
ingest_example.py — script auxiliar SOLO para poblar datos de prueba.

No forma parte del backend Rust: es un atajo rápido para meter unos
chunks de ejemplo en la BD usando un modelo de embeddings en Python
(sentence-transformers), útil mientras desarrollas el resto del flujo
sin tener que generar los vectores a mano.

En producción, la ingesta real probablemente tenga su propio pipeline
(chunking de documentos, etc.) — esto es solo para arrancar rápido.

Requisitos:
    pip install sentence-transformers psycopg2-binary

Uso:
    python ingest_example.py
"""

import psycopg2
from sentence_transformers import SentenceTransformer

# Mismo modelo que carga el cliente en www/embedder.js — IMPORTANTE que
# sea el mismo, o los vectores no serán comparables.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

DOCS = [
    ("manual.pdf", "El sistema soporta autenticación mediante OAuth2 y API keys."),
    ("manual.pdf", "Los límites de la API gratuita son de 1000 peticiones por día."),
    ("faq.md", "Para restablecer tu contraseña, usa el enlace de 'olvidé mi contraseña'."),
]

conn = psycopg2.connect("postgres://postgres:postgres@localhost:5432/ragdb")
cur = conn.cursor()

for source, text in DOCS:
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    cur.execute(
        "INSERT INTO documents (source, chunk_text, embedding) VALUES (%s, %s, %s)",
        (source, text, embedding),
    )

conn.commit()
cur.close()
conn.close()
print(f"Insertados {len(DOCS)} chunks de ejemplo.")
