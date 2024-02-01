use chess::{Board, Color, Piece};

pub struct StateBoard {
    board: Board,
}

impl StateBoard {
    pub fn new() -> Self {
        StateBoard {
            board: Board::default(),
        }
    }

    pub fn serialize(&self) -> [[u8; 8]; 8] {
        let mut state = [[0u8; 8]; 8];
        let serial_vals: [(char, u8); 12] = [
            ('P', 1),
            ('N', 2),
            ('B', 3),
            ('R', 4),
            ('Q', 5),
            ('K', 6),
            ('p', 11),
            ('n', 12),
            ('b', 13),
            ('r', 14),
            ('q', 16),
            ('k', 17),
        ];

        for i in 0..64 {
            let a_piece = self.board.piece_at(i);
            if let Some(piece) = a_piece {
                let symbol = piece.to_string().chars().next().unwrap();
                if let Some(&val) = serial_vals.iter().find(|&&(s, _)| s == symbol) {
                    state[i / 8][i % 8] = val.1;
                }
            }
        }

        if self.board.can_castle_kingside(Color::White) {
            assert_eq!(state[0][7], 4); // Ensure it's a white rook
            state[0][7] = 28;
        }

        if self.board.can_castle_queenside(Color::White) {
            assert_eq!(state[0][0], 4); // Ensure it's a white rook
            state[0][0] = 21;
        }

        if self.board.can_castle_kingside(Color::Black) {
            assert_eq!(state[7][7], 14); // Ensure it's a black rook
            state[7][7] = 38;
        }

        if self.board.can_castle_queenside(Color::Black) {
            assert_eq!(state[7][0], 14); // Ensure it's a black rook
            state[7][0] = 31;
        }

        if let Some(ep_square) = self.board.en_passant_square() {
            let ep_index = (ep_square.0 * 8 + ep_square.1) as usize;
            assert_eq!(state[ep_index / 8][ep_index % 8], 0);
            state[ep_index / 8][ep_index % 8] = 41;
        }

        state
    }

    pub fn moves(&self) -> Vec<chess::Move> {
        self.board.legal_moves().collect()
    }

    pub fn state_features(&self) -> (String, Color, chess::CastlingRights, Option<chess::Square>) {
        (
            self.board.board_fen(),
            self.board.side_to_move(),
            self.board.castling_rights(),
            self.board.en_passant_square(),
        )
    }