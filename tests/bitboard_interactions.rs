#[cfg(test)]
mod bitboard_interactions {
    use super::*;

    #[test]
    fn test_setting_and_getting_pieces() {
        let mut bit_board = BitBoard::new();
        let mut chess_board = ChessBoard::new(); // Assuming you have a constructor for ChessBoard

        let piece = Piece {
            color: Color::White,
            kind: PieceKind::Pawn,
            position: (1, 2),
        };

        bit_board.set_piece(piece, piece.position);
        chess_board.set_piece(piece, piece.position); // Assuming a set_piece method for ChessBoard

        assert_eq!(bit_board.get_piece_at(piece.position), Some(piece));
        assert_eq!(chess_board.get_piece_at(piece.position), Some(piece));
    }

    #[test]
    fn test_move_generation_for_rook() {
        let bit_board = BitBoard::from_fen("8/8/8/8/8/8/8/R7 w KQkq - 0 1"); // Assuming a from_fen method
        let magic_bitboards = MagicBitboards::new();
        let move_generator = MoveGenerator::new(bit_board, magic_bitboards);

        let moves = move_generator.legal_moves(Color::White);
        
        // Check if the moves include expected moves for the rook
        assert!(moves.contains(&((0, 0), (0, 1)))); // One possible move for the rook
    }

    #[test]
    fn test_zobrist_hashing_consistency() {
        let chess_board_1 = ChessBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let chess_board_2 = ChessBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let zobrist = Zobrist::new(12345); // Some seed

        assert_eq!(zobrist.hash(&chess_board_1), zobrist.hash(&chess_board_2));
    }

    #[test]
    fn test_rook_attacks() {
        let magic_bitboards = MagicBitboards::new();
        let bit_board = BitBoard::new(); // Assume a clear board

        let rook_attacks = magic_bitboards.rook_attacks(0, bit_board.occupancy()); // Check rook attacks from corner
        let expected_attacks = 0x01010101010101FE; // Expected bit pattern for rook attacks from a1

        assert_eq!(rook_attacks, expected_attacks);
    }

    #[test]
    fn test_piece_removal_and_board_update() {
        let mut bit_board = BitBoard::new();
        let mut chess_board = ChessBoard::new();

        let piece = Piece {
            color: Color::White,
            kind: PieceKind::Knight,
            position: (2, 2),
        };

        bit_board.set_piece(piece, piece.position);
        chess_board.set_piece(piece, piece.position);
        bit_board.remove_piece(piece, piece.position);
        chess_board.remove_piece(piece, piece.position); // Assuming a remove_piece method for ChessBoard

        assert_eq!(bit_board.get_piece_at(piece.position), None);
        assert_eq!(chess_board.get_piece_at(piece.position), None);
    }
}