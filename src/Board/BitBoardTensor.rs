use crate::bitboard::BitBoard;
use crate::square::Square;
use crate::piece::{Piece, PieceType};
use crate::color::Color;

pub struct ChessTensor {
    pub bitboards: [BitBoard; 12],
    pub shape: (usize, usize, usize),
}

impl ChessTensor {
    pub fn new() -> ChessTensor {
        ChessTensor {
            bitboards: [BitBoard::default(); 12],
            shape: (8, 8, 12),
        }
    }

    pub fn from_fen(fen: &str) -> ChessTensor {
        // Parse the FEN string and create bitboards for each piece type and color
        let mut bitboards = vec![BitBoard::default(); 12];
        let mut square_index = 0;

        for c in fen.chars() {
            match c {
                'P' => bitboards[Piece::new(PieceType::Pawn, Color::White).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'N' => bitboards[Piece::new(PieceType::Knight, Color::White).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'B' => bitboards[Piece::new(PieceType::Bishop, Color::White).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'R' => bitboards[Piece::new(PieceType::Rook, Color::White).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'Q' => bitboards[Piece::new(PieceType::Queen, Color::White).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'K' => bitboards[Piece::new(PieceType::King, Color::White).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'p' => bitboards[Piece::new(PieceType::Pawn, Color::Black).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'n' => bitboards[Piece::new(PieceType::Knight, Color::Black).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'b' => bitboards[Piece::new(PieceType::Bishop, Color::Black).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'r' => bitboards[Piece::new(PieceType::Rook, Color::Black).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'q' => bitboards[Piece::new(PieceType::Queen, Color::Black).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                'k' => bitboards[Piece::new(PieceType::King, Color::Black).to_index()] |= BitBoard::from_square(Square::new(square_index)),
                '/' | ' ' => {}
                _ => {
                    let digit = c.to_digit(10).unwrap();
                    square_index += digit as usize;
                    continue;
                }
            }
            square_index += 1;
        }

        ChessTensor {
            bitboards,
            shape: (8, 8, 12),
        }
    }

    pub fn patchify(&self, patch_size: usize) -> Vec<ChessTensor> {
        let mut patches = Vec::new();

        for i in (0..8).step_by(patch_size) {
            for j in (0..8).step_by(patch_size) {
                let mut patch = ChessTensor::new();
                for k in 0..12 {
                    let mask = Self::patch_mask(i, j, patch_size);
                    patch.bitboards[k] = (self.bitboards[k] & mask).shift_right(i * 8 + j);
                }
                patch.shape = (patch_size, patch_size, 12);
                patches.push(patch);
            }
        }

        patches
    }

    fn patch_mask(i: usize, j: usize, patch_size: usize) -> BitBoard {
        let mut mask = BitBoard::default();
        for x in i..i+patch_size {
            for y in j..j+patch_size {
                mask |= BitBoard::from_square(Square::make_square(x as u8, y as u8));
            }
        }
        mask
    }