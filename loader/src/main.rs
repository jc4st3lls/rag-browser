mod db;
mod embedder;
mod models;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use sqlx::postgres::PgPoolOptions;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    
    // Configuración de logs con filtro para silenciar las advertencias ruidosas de 'lopdf'
    // que rompen la renderización limpia de la barra de progreso.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,lopdf=error"))
        )
        .init();

    println!("==================================================");
    println!("        INICIANDO CARGADOR DE DOCUMENTOS          ");
    println!("==================================================");

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/ragdb".to_string());

    let input_dir_str = std::env::var("INPUT_DIR")
        .unwrap_or_else(|_| "input".to_string());
    let output_dir_str = std::env::var("OUTPUT_DIR")
        .unwrap_or_else(|_| std::env::var("FILES_DIR").unwrap_or_else(|_| "output".to_string()));

    let input_path = Path::new(&input_dir_str);
    let output_path = Path::new(&output_dir_str);

    // Asegurar que las carpetas de entrada y salida existen
    fs::create_dir_all(input_path).await?;
    fs::create_dir_all(output_path).await?;

    println!("Directorio de entrada vigilado: {:?}", input_path);
    println!("Directorio de salida de documentos: {:?}", output_path);

    // Cargar modelo de embeddings Candle
    println!("Descargando y cargando el modelo de embeddings multilingüe (paraphrase-multilingual-MiniLM-L12-v2) en CPU...");
    let embedder = Arc::new(embedder::LocalEmbedder::load_default()?);
    println!("Modelo cargado correctamente.");

    // Conectar a la base de datos Postgres
    println!("Conectando a la base de datos Postgres...");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;
    println!("Conexión a la base de datos establecida.");

    println!("Vigilando la carpeta {:?} por archivos PDF nuevos...", input_path);
    println!("--------------------------------------------------");

    loop {
        let mut reader = match fs::read_dir(input_path).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error leyendo directorio de entrada: {e}");
                tokio::time::sleep(Duration::from_secs(3)).await;
                continue;
            }
        };

        while let Ok(Some(entry)) = reader.next_entry().await {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                let filename = match path.file_name().and_then(|s| s.to_str()) {
                    Some(name) => name.to_string(),
                    None => continue,
                };

                println!("\n[+] Archivo PDF detectado: {}", filename);
                
                // Procesar el documento. Si da error, eliminamos el archivo problemático de 'input/' 
                // para evitar entrar en un bucle infinito que lo vuelva a intentar procesar cada 2 segundos.
                if let Err(e) = process_pdf(&path, &filename, &pool, &embedder, output_path).await {
                    eprintln!("[-] Error al procesar '{}': {:#}", filename, e);
                    eprintln!("[-] Eliminando archivo corrupto/erróneo de la entrada para evitar bucles.");
                    let _ = fs::remove_file(&path).await;
                } else {
                    println!("[+] Archivo '{}' procesado e indexado con éxito.", filename);
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn process_pdf(
    file_path: &Path,
    filename: &str,
    pool: &sqlx::PgPool,
    embedder: &embedder::LocalEmbedder,
    output_dir: &Path,
) -> Result<()> {
    // 1. Leer los bytes del archivo
    let pdf_data = fs::read(file_path).await?;
    if pdf_data.is_empty() {
        return Err(anyhow::anyhow!("El archivo PDF está vacío."));
    }

    // 2. Si el documento ya existía previamente, eliminar toda su información antigua de la base de datos
    // para evitar duplicados, chunks basura o inconsistencias en la base de datos vectorial.
    println!("    Limpiando indexación previa si existía...");
    db::delete_document_data(pool, filename).await?;

    // 3. Guardar el PDF en la base de datos original_documents para persistencia y fallback
    db::insert_original_document(pool, filename, &pdf_data).await?;

    // 4. Extraer páginas de texto del PDF
    let pages = embedder::extract_pdf_pages(&pdf_data)?;
    if pages.is_empty() {
        return Err(anyhow::anyhow!("El documento PDF no contiene texto extraíble."));
    }

    // 5. Clasificar el documento
    let classification_sample: String = pages.iter().take(3).map(|(_, t)| t.clone()).collect::<Vec<_>>().join(" ");
    let (category, subdomain) = embedder::classify_text(&classification_sample);
    println!("    Categoría clasificada: '{}', Subdominio: '{:?}'", category, subdomain);

    // 6. Fragmentar en chunks
    // Aumentamos el tamaño de los chunks (chunk_size: 1000 caracteres, overlap: 150 caracteres)
    // para retener mucho más contexto semántico y que las respuestas sean más ricas.
    let chunks = embedder::chunk_text(&pages, 1000, 150);
    let total_chunks = chunks.len();
    println!("    Total de chunks generados: {}", total_chunks);

    // 7. Crear una barra de progreso elegante para la indexación
    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
            .progress_chars("#>-")
    );

    let mut docs_to_ingest = Vec::new();

    // 8. Generar embeddings por chunk actualizando la barra de progreso
    for (i, (page_num, chunk_text)) in chunks.into_iter().enumerate() {
        let chunk_idx = i + 1;
        pb.set_message(format!("Página {}", page_num));
        let embedding = embedder.embed(&chunk_text)?;

        docs_to_ingest.push(models::IngestDocument {
            source: format!("{} (page {})", filename, page_num),
            chunk_text,
            embedding,
            category: Some(category.clone()),
            subdomain: subdomain.clone(),
        });

        pb.inc(1);

        // Si la barra de progreso de indicatif está oculta (p.ej. ejecutando bajo Docker Compose
        // sin un TTY asignado), imprimimos líneas de progreso de texto tradicionales cada 50 chunks.
        if pb.is_hidden() {
            if chunk_idx % 50 == 0 || chunk_idx == total_chunks {
                let percent = (chunk_idx as f32 / total_chunks as f32 * 100.0) as u32;
                println!("    [+] Generando embeddings: {} / {} chunks ({}%)", chunk_idx, total_chunks, percent);
            }
        }
    }

    pb.finish_with_message("Embeddings generados.");

    // 9. Insertar los chunks e embeddings en la base de datos
    println!("    Insertando en la base de datos...");
    let inserted_count = db::insert_documents(pool, docs_to_ingest).await?;
    println!("    {} chunks guardados exitosamente.", inserted_count);

    // 10. Mover el archivo original al directorio de salida (donde el backend puede acceder a él)
    let dest_path = output_dir.join(filename);
    
    // Si ya existe el archivo físico en el directorio de salida, lo eliminamos antes de mover
    // para evitar cualquier error de bloqueo o conflicto al sobrescribir.
    if dest_path.exists() {
        let _ = fs::remove_file(&dest_path).await;
    }

    println!("    Moviendo archivo a la carpeta de salida: {:?}", dest_path);
    // Usamos fs::copy + fs::remove_file en lugar de fs::rename.
    // fs::rename falla con "Invalid cross-device link (os error 18)" en Docker o entornos 
    // donde el directorio de entrada y salida están en diferentes sistemas de archivos o particiones de disco.
    fs::copy(file_path, &dest_path).await?;
    let _ = fs::remove_file(file_path).await;

    Ok(())
}
