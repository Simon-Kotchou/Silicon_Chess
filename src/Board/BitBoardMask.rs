use crate::bitboard::BitBoard;
use crate::square::Square;
use chess::{Board, ChessMove};
use ndarray::Array2;

fn square_masking(board_size: usize, num_targets: usize, context_scale: (f32, f32), target_scale: (f32, f32), min_target_size: usize) -> (BitBoard, Vec<BitBoard>, BitBoard) {
    let num_squares = board_size * board_size;
    let mut context_mask = BitBoard::new(!0);
    let mut target_masks = Vec::new();
    let mut occupied_squares = BitBoard::empty();

    for _ in 0..num_targets {
        let target_size = (num_squares as f32 * rand::thread_rng().gen_range(target_scale.0..target_scale.1)).round() as usize;
        let target_size = target_size.max(min_target_size).min(num_squares - occupied_squares.popcnt() as usize);

        let available_squares = !occupied_squares;
        let mut target_mask = BitBoard::empty();
        for _ in 0..target_size {
            let target_square = available_squares.to_square();
            target_mask |= BitBoard::from_square(target_square);
            occupied_squares |= BitBoard::from_square(target_square);
        }
        target_masks.push(target_mask);

        context_mask &= !target_mask;
    }

    let combined_mask = target_masks.iter().fold(context_mask, |mask, target_mask| mask & !target_mask);

    (context_mask, target_masks, combined_mask)
}

fn bitboard_to_tensor(bitboard: BitBoard, board_size: usize) -> Array2<f32> {
    let mut tensor = Array2::zeros((board_size, board_size));
    for square in bitboard {
        let (rank, file) = (square.get_rank().to_index(), square.get_file().to_index());
        tensor[(rank, file)] = 1.0;
    }
    tensor
}

fn tensor_to_bitboard(tensor: &Array2<f32>) -> BitBoard {
    let mut bitboard = BitBoard::empty();
    for ((rank, file), &value) in tensor.indexed_iter() {
        if value > 0.5 {
            let square = Square::make_square(Rank::from_index(rank), File::from_index(file));
            bitboard |= BitBoard::from_square(square);
        }
    }
    bitboard
}

fn parse_pgn(pgn: &str) -> Vec<BitBoard> {
    let mut boards = Vec::new();
    let mut board = Board::default();
    boards.push(board.get_occupied());

    let pgn = pgn.trim();
    for mv in pgn.split_whitespace().filter_map(|s| s.parse::<ChessMove>().ok()) {
        board.make_move(mv);
        boards.push(board.get_occupied());
    }

    boards
}

fn parse_fen(fen: &str) -> BitBoard {
    let board = Board::from_fen(fen).unwrap();
    board.get_occupied()
}