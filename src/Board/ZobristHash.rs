use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub struct Zobrist {
    table: [[[u64; 8]; 8]; 12],
}

impl Zobrist {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut table = [[[0; 8]; 8]; 12];

        for i in 0..12 {
            for j in 0..8 {
                for k in 0..8 {
                    table[i][j][k] = rng.gen();
                }
            }
        }

        Zobrist { table }
    }

    pub fn hash(&self, board: &ChessBoard) -> u64 {
        let mut hash = 0;

        for (i, row) in board.board.iter().enumerate() {
            for (j, piece_opt) in row.iter().enumerate() {
                if let Some(piece) = piece_opt {
                    let piece_index = (piece.color as usize) * 6 + (piece.kind as usize);
                    hash ^= self.table[piece_index][i][j];
                }
            }
        }

        hash
    }
}