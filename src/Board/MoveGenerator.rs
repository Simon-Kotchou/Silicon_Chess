pub struct MoveGenerator {
    bit_board: BitBoard,
    magic_bitboards: MagicBitboards,
}

impl MoveGenerator {
    pub fn new(bit_board: BitBoard, magic_bitboards: MagicBitboards) -> Self {
        MoveGenerator {
            bit_board,
            magic_bitboards,
        }
    }

    pub fn rook_moves(&self, position: (usize, usize), color: Color) -> u64 {
        let index = BitBoard::position_to_index(position) as usize;
        let occupancy = self.bit_board.occupancy() & self.magic_bitboards.rook_mask[index];
        let magic_index = (((occupancy.wrapping_mul(self.magic_bitboards.rook_magics[index])) >> self.magic_bitboards.rook_shift[index]) as usize) + self.magic_bitboards.rook_offsets[index];

        self.magic_bitboards.rook_table[magic_index] & !self.bit_board.occupancy_color(color)

        pub fn bishop_moves(&self, position: (usize, usize), color: Color) -> u64 {
            let index = BitBoard::position_to_index(position) as usize;
            let occupancy = self.bit_board.occupancy() & self.magic_bitboards.bishop_mask[index];
            let magic_index = (((occupancy.wrapping_mul(self.magic_bitboards.bishop_magics[index])) >> self.magic_bitboards.bishop_shift[index]) as usize) + self.magic_bitboards.bishop_offsets[index];
    
            self.magic_bitboards.bishop_table[magic_index] & !self.bit_board.occupancy_color(color)
        }
    
        pub fn knight_moves(&self, position: (usize, usize), color: Color) -> u64 {
            // Implement knight moves logic here using magic bitboards or other methods
            // Placeholder implementation, replace with actual logic.
            0
        }
    
        pub fn king_moves(&self, position: (usize, usize), color: Color) -> u64 {
            // Implement king moves logic here using magic bitboards or other methods
            // Placeholder implementation, replace with actual logic.
            0
        }
    
        pub fn pawn_moves(&self, position: (usize, usize), color: Color) -> u64 {
            // Implement pawn moves logic here using magic bitboards or other methods
            // Placeholder implementation, replace with actual logic.
            0
        }
    
        pub fn queen_moves(&self, position: (usize, usize), color: Color) -> u64 {
            // Implement queen moves logic here using magic bitboards or other methods
            // Placeholder implementation, replace with actual logic.
            0
        }
    
        pub fn legal_moves(&self, color: Color) -> Vec<((usize, usize), (usize, usize))> {
            let mut moves = Vec::new();
    
            for square in 0..64 {
                let position = BitBoard::index_to_position(square);
    
                if let Some((piece, piece_color)) = self.bit_board.get_piece_at(position) {
                    if piece_color == color {
                        match piece {
                            Piece::Rook => {
                                let rook_moves = self.rook_moves(position, color);
                                for dst_square in BitBoard::bits_to_positions(rook_moves) {
                                    moves.push((position, dst_square));
                                }
                            }
                            Piece::Bishop => {
                                let bishop_moves = self.bishop_moves(position, color);
                                for dst_square in BitBoard::bits_to_positions(bishop_moves) {
                                    moves.push((position, dst_square));
                                }
                            }
                            Piece::Knight => {
                                let knight_moves = self.knight_moves(position, color);
                                for dst_square in BitBoard::bits_to_positions(knight_moves) {
                                    moves.push((position, dst_square));
                                }
                            }
                            Piece::King => {
                                let king_moves = self.king_moves(position, color);
                                for dst_square in BitBoard::bits_to_positions(king_moves) {
                                    moves.push((position, dst_square));
                                }
                            }
                            Piece::Pawn => {
                                let pawn_moves = self.pawn_moves(position, color);
                                for dst_square in BitBoard::bits_to_positions(pawn_moves) {
                                    moves.push((position, dst_square));
                                }
                            }
                            Piece::Queen => {
                                let queen_moves = self.queen_moves(position, color);
                                for dst_square in BitBoard::bits_to_positions(queen_moves) {
                                    moves.push((position, dst_square));
                                }
                            }
                        }
                    }
                }
            }
    
            moves
        }
    }