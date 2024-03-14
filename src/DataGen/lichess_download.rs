use futures::future::try_join_all;
use reqwest::Client;
use std::io::Cursor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_postgres::{Client as PgClient, NoTls};
use zstd::stream::copy_decode;

async fn download_and_decompress(client: &Client, url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let response = client.get(url).send().await?;
    let compressed_data = response.bytes().await?;
    let mut decompressed_data = Vec::new();
    copy_decode(Cursor::new(compressed_data), &mut decompressed_data)?;
    Ok(decompressed_data)
}

async fn download_and_decompress_cache(client: &Client, url: &str, cache_dir: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Generate a unique filename based on the URL
    let filename = generate_filename(url);
    let cache_path = format!("{}/{}", cache_dir, filename);

    // Check if the file exists in the cache
    if let Ok(cached_data) = tokio::fs::read(&cache_path).await {
        return Ok(cached_data);
    }

    // Send a GET request with conditional headers
    let response = client
        .get(url)
        .header(reqwest::header::IF_NONE_MATCH, "*")
        .send()
        .await?;

    // Check if the server returned a 304 Not Modified response
    if response.status() == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(Vec::new());
    }

    // Ensure the response is successful
    if !response.status().is_success() {
        return Err(format!("Failed to download file: {}", url).into());
    }

    // Get the total file size from the Content-Length header
    let total_size = response.content_length().unwrap_or(0);

    // Create a progress bar
    let progress_bar = indicatif::ProgressBar::new(total_size);

    // Download the file and decompress it
    let mut compressed_data = Vec::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        compressed_data.extend_from_slice(&chunk);
        progress_bar.inc(chunk.len() as u64);
    }

    progress_bar.finish_with_message("Download completed");

    // Decompress the downloaded data
    let mut decompressed_data = Vec::new();
    zstd::stream::copy_decode(Cursor::new(&compressed_data), &mut decompressed_data)?;

    // Save the decompressed data to the cache
    tokio::fs::write(&cache_path, &decompressed_data).await?;

    Ok(decompressed_data)
}

async fn process_games(decompressed_data: Vec<u8>, pg_client: &PgClient) -> Result<(), Box<dyn std::error::Error>> {
    // Parse the decompressed PGN data and extract relevant information
    // You can use a PGN parsing library like `chess` or `shakmaty` for this

    // Store the processed data in PostgreSQL
    for game in parsed_games {
        pg_client.execute(
            "INSERT INTO games (id, moves, white_elo, black_elo) VALUES ($1, $2, $3, $4)",
            &[&game.id, &game.moves, &game.white_elo, &game.black_elo],
        ).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec![
        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-02.pgn.zst",
        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst",
        // Add more URLs as needed
    ];

    let http_client = Client::new();
    let (pg_client, pg_connection) = PgClient::connect("postgres://user:password@localhost/chess", NoTls).await?;

    tokio::spawn(async move {
        if let Err(e) = pg_connection.await {
            eprintln!("PostgreSQL connection error: {}", e);
        }
    });

    pg_client.execute(
        "CREATE TABLE IF NOT EXISTS games (
            id SERIAL PRIMARY KEY,
            moves TEXT,
            white_elo INTEGER,
            black_elo INTEGER
        )",
        &[],
    ).await?;

    let download_tasks = urls
        .iter()
        .map(|url| download_and_decompress(&http_client, url));

    let decompressed_data_list = try_join_all(download_tasks).await?;

    let process_tasks = decompressed_data_list
        .into_iter()
        .map(|decompressed_data| process_games(decompressed_data, &pg_client));

    try_join_all(process_tasks).await?;

    Ok(())
}