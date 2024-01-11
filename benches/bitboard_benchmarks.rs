use criterion::{black_box, criterion_group, criterion_main, Criterion};
use Silicon_Chess::Board::{BitBoard, MagicBitboards, Zobrist, MoveGenerator, ...};

fn bench_rook_moves(c: &mut Criterion) {
    let bit_board = BitBoard::new(); // setup
    let magic_bitboards = MagicBitboards::new();
    c.bench_function("rook_moves", |b| {
        b.iter(|| black_box(MagicBitboards::rook_attacks(&magic_bitboards, /* args */)))
    });
}

fn bench_zobrist_hashing(c: &mut Criterion) {
    let zobrist = Zobrist::new(/* seed */);
    let chess_board = /* setup board */;
    c.bench_function("zobrist_hashing", |b| {
        b.iter(|| black_box(zobrist.hash(&chess_board)))
    });
}

fn bench_set_and_remove_piece(c: &mut Criterion) {
    let mut bit_board = BitBoard::new();
    let piece = /* setup piece */;
    c.bench_function("set_and_remove_piece", |b| {
        b.iter(|| {
            bit_board.set_piece(piece, /* position */);
            bit_board.remove_piece(piece, /* position */);
        })
    });
}

fn bench_legal_move_generation(c: &mut Criterion) {
    let bit_board = BitBoard::new(); // setup
    let magic_bitboards = MagicBitboards::new();
    let move_generator = MoveGenerator::new(bit_board, magic_bitboards);
    c.bench_function("legal_move_generation", |b| {
        b.iter(|| black_box(move_generator.legal_moves(/* color */)))
    });
}

criterion_group!(benches, bench_rook_moves, bench_zobrist_hashing, bench_set_and_remove_piece, bench_legal_move_generation);
criterion_main!(benches);