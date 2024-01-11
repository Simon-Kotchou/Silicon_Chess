pub struct GameState {
    pub chess_board: ChessBoard,
    pub bit_board: BitBoard,
    pub zobrist: Zobrist,
    pub hash: u64, // Store the hash value of the current board state
}

impl GameState {
    pub fn random_child(&self) -> GameState {
        // Implement the logic to generate a random child GameState
        // This is just a placeholder, replace with your actual logic
        self.clone()
    }

    pub fn random_playout(&self) -> f64 {
        // Implement the logic to simulate a random playout and return the result
        // This is just a placeholder, replace with your actual logic
        0.0
    }

    pub fn from_str(s: &str) -> GameState {
        let chess_board = ChessBoard::from_str(s);
        let bit_board = BitBoard::from_chess_board(&chess_board);
        let zobrist = Zobrist::new(seed);
        let hash = zobrist.hash(&chess_board);
        GameState {
            chess_board,
            bit_board,
            zobrist,
            hash,
        }
    }
}

impl ToString for GameState {
    fn to_string(&self) -> String {
        self.chess_board.to_string()
    }
}

impl Clone for GameState {
    fn clone(&self) -> Self {
        GameState {
            chess_board: self.chess_board.clone(),
            bit_board: self.bit_board.clone(),
            zobrist: self.zobrist.clone(),
            hash: self.hash,
        }
    }
}