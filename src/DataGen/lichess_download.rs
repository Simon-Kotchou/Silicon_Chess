use std::fs::File;
use std::io::Write;
use reqwest::blocking::get;
use zstd::stream::copy_decode;

fn download_and_decompress(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = get(url)?;
    let compressed_file = "compressed.zst";
    let mut compressed_output = File::create(compressed_file)?;
    compressed_output.write_all(&response.bytes()?)?;

    let decompressed_file = "decompressed.pgn";
    let mut decompressed_output = File::create(decompressed_file)?;
    copy_decode(File::open(compressed_file)?, &mut decompressed_output)?;

    Ok(())
}

fn main() {
    let urls = vec![
        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-02.pgn.zst",
        "https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst",
        // Add more URLs as needed
    ];

    let handles: Vec<_> = urls
        .into_iter()
        .map(|url| std::thread::spawn(move || download_and_decompress(url)))
        .collect();

    for handle in handles {
        handle.join().unwrap().unwrap();
    }
}