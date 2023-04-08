pub struct ChessBoard {
    pub board: [[Option<Piece>; 8]; 8],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
    pub position: (usize, usize),
}

impl Piece {
    pub fn legal_moves(&self, board: &ChessBoard) -> Vec<(usize, usize)> {
        match self.kind {
            PieceKind::King => self.king_moves(board),
            PieceKind::Queen => self.queen_moves(board),
            PieceKind::Rook => self.rook_moves(board),
            PieceKind::Bishop => self.bishop_moves(board),
            PieceKind::Knight => self.knight_moves(board),
            PieceKind::Pawn => self.pawn_moves(board),
        }
    }

    fn pawn_moves(&self, board: &ChessBoard) -> Vec<(usize, usize)> {
        let mut moves = Vec::new();
        let (x, y) = self.position;
        let direction = match self.color {
            Color::White => 1,
            Color::Black => -1,
        };

        // One step forward
        let new_x = (x as i32 + direction) as usize;
        if new_x < 8 && board.board[new_x][y].is_none() {
            moves.push((new_x, y));
        }

        // Two steps forward
        if !self.has_moved() && new_x < 8 && board.board[new_x][y].is_none() {
            let new_x2 = (new_x as i32 + direction) as usize;
            if new_x2 < 8 && board.board[new_x2][y].is_none() {
                moves.push((new_x2, y));
            }
        }

        // Capture
        for &new_y in &[y.wrapping_sub(1), y + 1] {
            if new_x < 8 && new_y < 8 {
                if let Some(ref other_piece) = board.board[new_x][new_y] {
                    if other_piece.color != self.color {
                        moves.push((new_x, new_y));
                    }
                }
            }
        }

        moves
    }

    // Implement move generation for other piece types
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
}