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

// ****

use rand::Rng;

const NUM_SQUARES: usize = 64;
const NUM_PIECES: usize = 12;
const NUM_CASTLING_RIGHTS: usize = 16;
const NUM_EP_SQUARES: usize = 64;
const NUM_COLORS: usize = 2;

pub struct ZobristKeys {
    piece_keys: [[u64; NUM_PIECES]; NUM_SQUARES],
    castling_keys: [u64; NUM_CASTLING_RIGHTS],
    ep_keys: [u64; NUM_EP_SQUARES],
    color_key: u64,
}

impl ZobristKeys {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut piece_keys = [[0u64; NUM_PIECES]; NUM_SQUARES];

        for square in 0..NUM_SQUARES {
            for piece in 0..NUM_PIECES {
                piece_keys[square][piece] = rng.gen();
            }
        }

        let mut castling_keys = [0u64; NUM_CASTLING_RIGHTS];
        for i in 0..NUM_CASTLING_RIGHTS {
            castling_keys[i] = rng.gen();
        }

        let mut ep_keys = [0u64; NUM_EP_SQUARES];
        for i in 0..NUM_EP_SQUARES {
            ep_keys[i] = rng.gen();
        }

        let color_key = rng.gen();

        ZobristKeys {
            piece_keys,
            castling_keys,
            ep_keys,
            color_key,
        }
    }

    pub fn hash(&self, bitboard: &BitBoard) -> u64 {
        let mut hash: u64 = 0;

        for square in 0..NUM_SQUARES {
            if let Some(piece) = bitboard.get_piece_at_square(square) {
                let piece_index = piece_to_index(piece);
                hash ^= self.piece_keys[square][piece_index];
            }
        }

        let castling_rights = bitboard.get_castling_rights();
        let ep_square = bitboard.get_ep_square();
        let color = bitboard.get_color();

        hash ^= self.castling_keys[castling_rights as usize];
        if let Some(ep_square) = ep_square {
            let ep_index = square_to_index(ep_square);
            hash ^= self.ep_keys[ep_index];
        }

        if color == Color::White {
            hash ^= self.color_key;
        }

        hash
    }
}

fn piece_to_index(piece: Piece) -> usize {
    match piece.color {
        Color::White => piece.kind as usize,
        Color::Black => (piece.kind as usize) + NUM_PIECES / 2,
    }
}

fn square_to_index(square: chess::Square) -> usize {
    square.to_index() as usize
}