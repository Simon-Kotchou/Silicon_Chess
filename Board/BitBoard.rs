pub struct BitBoard {
    pub white_pieces: [u64; 6],
    pub black_pieces: [u64; 6],
}

impl BitBoard {
    pub fn new() -> Self {
        BitBoard {
            white_pieces: [0; 6],
            black_pieces: [0; 6],
        }
    }

    pub fn set_piece(&mut self, piece: Piece, position: (usize, usize)) {
        let index = Self::position_to_index(position);
        let piece_index = piece.kind as usize;

        match piece.color {
            Color::White => self.white_pieces[piece_index] |= 1 << index,
            Color::Black => self.black_pieces[piece_index] |= 1 << index,
        }
    }

    pub fn remove_piece(&mut self, piece: Piece, position: (usize, usize)) {
        let index = Self::position_to_index(position);
        let piece_index = piece.kind as usize;

        match piece.color {
            Color::White => self.white_pieces[piece_index] &= !(1 << index),
            Color::Black => self.black_pieces[piece_index] &= !(1 << index),
        }
    }

    fn position_to_index(position: (usize, usize)) -> u8 {
        (position.0 * 8 + position.1) as u8
    }

    pub fn occupancy(&self) -> u64 {
        self.white_pieces.iter().chain(self.black_pieces.iter()).map(|&x| x).sum()
    }

    pub fn occupancy_color(&self, color: Color) -> u64 {
        match color {
            Color::White => self.white_pieces.iter().map(|&x| x).sum(),
            Color::Black => self.black_pieces.iter().map(|&x| x).sum(),
        }
    }
}