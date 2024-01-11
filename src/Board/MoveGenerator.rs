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

    pub fn legal_moves(&self, color: Color) -> Vec<((usize, usize), (usize, usize))> {
        let mut moves = Vec::new();

        for square in 0..64 {
            let position = BitBoard::index_to_position(square);

            if self.bit_board.get_piece_at(position) == Some((Piece::Rook, color)) {
                let rook_moves = self.rook_moves(position, color);
                for dst_square in BitBoard::bits_to_positions(rook_moves) {
                    moves.push((position, dst_square));
                }
            }

            // Add similar blocks for other piece types, e.g. bishops, knights, kings, pawns, and queens
        }

        moves
    }
}