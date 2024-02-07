use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

impl DeepLearningAgent {
    fn select_move(&self, game_state: &GameState) -> Move {
        let num_moves = self.encoder.board_width() * self.encoder.board_height();
        let move_probs: Vec<f64> = self.predict(game_state)
            .iter::<f64>() // Assuming predict returns a 1D Tensor of probabilities
            .unwrap()
            .collect();

        // Apply transformations to move probabilities as in the Python code
        let move_probs: Vec<f64> = move_probs.iter()
            .map(|p| p.powf(3.0)) // Increase the distance between the most and least likely moves
            .map(|p| p.max(1e-6).min(1.0 - 1e-6)) // Prevent probs from getting stuck at 0 or 1
            .collect();
        let sum_probs: f64 = move_probs.iter().sum();
        let move_probs: Vec<f64> = move_probs.into_iter().map(|p| p / sum_probs).collect(); // Re-normalize

        // Set up weighted random choice
        let dist = WeightedIndex::new(&move_probs).unwrap();
        let mut rng = thread_rng();

        // Try to select a valid move that doesn't reduce eye-space, else pass
        for _ in 0..num_moves {
            let point_idx = dist.sample(&mut rng);
            let point = self.encoder.decode_point_index(point_idx); // Assuming this function exists within the encoder

            if game_state.is_valid_move(goboard::Move::Play(point)) && !is_point_an_eye(game_state.board(), point, game_state.next_player()) {
                return goboard::Move::Play(point);
            }
        }

        goboard::Move::Pass // If no valid move found, pass
    }
}