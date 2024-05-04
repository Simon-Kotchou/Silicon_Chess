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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <num_puzzles> <agent>", args[0]);
        std::process::exit(1);
    }

    let num_puzzles = args[1].parse::<usize>().expect("Invalid num_puzzles");
    let agent = &args[2];

    let puzzles_path = Path::new("../data/puzzles.csv");
    let puzzles_csv = fs::read_to_string(puzzles_path).expect("Failed to read puzzles.csv");
    let mut puzzles: Vec<Puzzle> = csv::Reader::from_reader(io::Cursor::new(puzzles_csv))
        .deserialize()
        .take(num_puzzles + 1)
        .map(|result| result.expect("Failed to parse puzzle"))
        .collect();

    let mut engine = ENGINE_BUILDERS[agent]();

    for (puzzle_id, puzzle) in puzzles.iter().enumerate() {
        let correct = evaluate_puzzle_from_pandas_row(puzzle, &mut engine);
        println!("{{\"puzzle_id\": {}, \"correct\": {}, \"rating\": {}}}", puzzle_id, correct, puzzle.rating);
    }
}