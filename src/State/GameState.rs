pub struct GameState {
    pub chess_board: ChessBoard,
    pub bit_board: BitBoard,
    pub zobrist: Zobrist,
    pub hash: u64, // Store the hash value of the current board state
}

impl GameState {
    pub fn random_child(&self) -> GameState {
        // Implement the logic to generate a random child GameState
        // This involves making a random legal move and updating the state accordingly.
        // You can use your chess library to generate random legal moves and apply one of them.
        let mut new_state = self.clone();

        // Generate a list of legal moves for the current state (use your chess library)
        let legal_moves = new_state.bit_board.moves();

        // Choose a random legal move (you can use a random number generator here)
        if let Some(random_move) = legal_moves.choose(&mut rand::thread_rng()) {
            // Apply the selected move to the state
            new_state.bit_board.apply_move(random_move);
        }

        // Calculate the new Zobrist hash for the updated state
        new_state.hash = new_state.zobrist.hash(&new_state.chess_board);

        new_state
    }

    pub fn random_playout(&self) -> f64 {
        // Implement the logic to simulate a random playout and return the result
        // This involves making random moves until the game ends and returning the result (e.g., win/loss or a score).

        let mut current_state = self.clone();

        // You can implement a simple playout strategy like playing random legal moves until the game ends
        while !current_state.chess_board.is_game_over() {
            let legal_moves = current_state.bit_board.moves();

            if let Some(random_move) = legal_moves.choose(&mut rand::thread_rng()) {
                // Apply a random move to the current state
                current_state.bit_board.apply_move(random_move);

                // Update the Zobrist hash
                current_state.hash = current_state.zobrist.hash(&current_state.chess_board);
            }
        }

        // You can return a value indicating the outcome of the playout (e.g., 1.0 for win, 0.0 for draw, -1.0 for loss)
        // This depends on your game's scoring system.
        // Placeholder implementation, replace with your actual logic.
        let result = 0.0;
        result

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