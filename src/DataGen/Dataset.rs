use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct ChessDataset {
    positions: Vec<BitBoard>,
}

impl ChessDataset {
    pub fn from_fen_file<P: AsRef<Path>>(path: P) -> Self {
        let file = File::open(path).expect("Unable to open FEN file");
        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        for line in reader.lines() {
            let fen = line.expect("Unable to read FEN line");
            let chess_board = ChessBoard::from_fen(&fen);
            let bit_board = BitBoard::from_chess_board(&chess_board);
            positions.push(bit_board);
        }

        ChessDataset { positions }
    }

    pub fn from_pgn_file<P: AsRef<Path>>(path: P) -> Self {
        // Implement PGN parsing and convert the resulting positions to BitBoards.
    }

    pub fn save_binary<P: AsRef<Path>>(&self, path: P) {
        let mut data = Vec::new();

        for bit_board in &self.positions {
            data.extend_from_slice(&bit_board.white_pieces);
            data.extend_from_slice(&bit_board.black_pieces);
        }

        let mut file = File::create(path).expect("Unable to create binary file");
        file.write_all(data.as_bytes()).expect("Unable to write binary data");
    }
}