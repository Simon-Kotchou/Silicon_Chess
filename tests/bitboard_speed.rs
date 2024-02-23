use criterion::{black_box, criterion_group, criterion_main, Criterion};
use your_crate_name::BitBoard; // Make sure to replace `your_crate_name` with the actual name of your crate

fn bitboard_operations_benchmark(c: &mut Criterion) {
    let bb1 = BitBoard::new(0xF0F0F0F0F0F0F0F0);
    let bb2 = BitBoard::new(0x0F0F0F0F0F0F0F0F);

    c.bench_function("BitBoard AND operation", |b| {
        b.iter(|| black_box(bb1) & black_box(bb2))
    });

    c.bench_function("BitBoard OR operation", |b| {
        b.iter(|| black_box(bb1) | black_box(bb2))
    });

    c.bench_function("BitBoard XOR operation", |b| {
        b.iter(|| black_box(bb1) ^ black_box(bb2))
    });

    c.bench_function("BitBoard NOT operation", |b| {
        b.iter(|| !black_box(bb1))
    });

    c.bench_function("BitBoard AND assignment", |b| {
        b.iter(|| {
            let mut bb = black_box(bb1);
            bb &= black_box(bb2);
            bb
        })
    });

    c.bench_function("BitBoard OR assignment", |b| {
        b.iter(|| {
            let mut bb = black_box(bb1);
            bb |= black_box(bb2);
            bb
        })
    });

    c.bench_function("BitBoard XOR assignment", |b| {
        b.iter(|| {
            let mut bb = black_box(bb1);
            bb ^= black_box(bb2);
            bb
        })
    });

    c.bench_function("BitBoard Multiplication", |b| {
        b.iter(|| black_box(bb1) * black_box(bb2))
    });

    c.bench_function("BitBoard Population Count", |b| {
        b.iter(|| black_box(bb1).popcnt())
    });

    c.bench_function("BitBoard Reverse Colors", |b| {
        b.iter(|| black_box(bb1).reverse_colors())
    });
}

criterion_group!(benches, bitboard_operations_benchmark);
criterion_main!(benches);