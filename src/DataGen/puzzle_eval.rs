use std::io;
use std::env;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use chess::{Board, ChessMove};
use chess::pgn::Reader;
use serde::Deserialize;

use crate::engines::constants::ENGINE_BUILDERS;
use crate::engines::engine::Engine;

#[derive(Deserialize)]
struct Puzzle {
    pgn: String,
    moves: String,
    rating: u32,
}

fn evaluate_puzzle_from_pandas_row(puzzle: &Puzzle, engine: &mut dyn Engine) -> bool {
    let game = Reader::new(&puzzle.pgn).read_game().expect("Failed to read game from PGN");
    let board = game.end().board().clone();
    evaluate_puzzle_from_board(&board, &puzzle.moves.split(' ').collect::<Vec<_>>(), engine)
}

fn evaluate_puzzle_from_board(board: &Board, moves: &[&str], engine: &mut dyn Engine) -> bool {
    let mut board = board.clone();
    for (move_idx, move_str) in moves.iter().enumerate() {
        if move_idx % 2 == 1 {
            let predicted_move = engine.play(&board);
            if predicted_move.to_string() != *move_str {
                board.make_move(ChessMove::from_str(predicted_move.as_str()).unwrap());
                return board.checkmate();
            }
        }
        board.make_move(ChessMove::from_str(move_str).unwrap());
    }
    true
}