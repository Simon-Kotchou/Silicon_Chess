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
        // Implement move generation for each piece type using bitwise operations and magic bitboards
    }
}