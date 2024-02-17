use rand::seq::SliceRandom;
use rand::thread_rng;

// This function generates a valid Chess960 starting position.
fn generate_chess960_position() -> Vec<Piece> {
    let mut pieces = vec![
        Piece::Rook,
        Piece::Knight,
        Piece::Bishop,
        Piece::Queen,
        Piece::King,
        Piece::Bishop,
        Piece::Knight,
        Piece::Rook,
    ];
    let mut rng = thread_rng();
    pieces.shuffle(&mut rng);
    while !is_valid_chess960_position(&pieces) {
        pieces.shuffle(&mut rng);
    }
    pieces
}

// This function checks if a position is valid under Chess960 rules.
fn is_valid_chess960_position(pieces: &[Piece]) -> bool {
    let king_index = pieces.iter().position(|&piece| piece == Piece::King).unwrap();
    let rook_indices: Vec<_> = pieces.iter().enumerate()
        .filter(|&(_, &piece)| piece == Piece::Rook)
        .map(|(index, _)| index)
        .collect();
    rook_indices[0] < king_index && king_index < rook_indices[1]
}

// This function validates a move under Chess960 rules.
fn validate_move_chess960(board: &Board, move: &Move) -> bool {
    if move.is_castling() {
        // In Chess960, a castling move is represented as a king move to the rook's square.
        // The king then moves two squares towards the rook, and the rook moves to the square the king skipped over.
        let king_index = board.pieces.iter().position(|&piece| piece == Piece::King).unwrap();
        let rook_index = move.destination();
        let (new_king_index, new_rook_index) = if rook_index < king_index {
            (king_index - 2, king_index - 1)
        } else {
            (king_index + 2, king_index + 1)
        };
        // Check that the squares between the king and the rook are empty.
        for index in king_index.min(rook_index) + 1..king_index.max(rook_index) {
            if board.pieces[index] != Piece::Empty {
                return false;
            }
        }
        // Check that the squares the king will cross are not attacked.
        for index in king_index..=new_king_index {
            if board.is_square_attacked(index, board.side_to_move) {
                return false;
            }
        }
        // Make the castling move.
        board.pieces[king_index] = Piece::Empty;
        board.pieces[rook_index] = Piece::Empty;
        board.pieces[new_king_index] = Piece::King;
        board.pieces[new_rook_index] = Piece::Rook;
    } else {
        // For non-castling moves, use the standard rules.
        validate_move_standard(board, move)
    }
}