use criterion::{black_box, criterion_group, criterion_main, Criterion};
use your_crate_name::BitBoard; // Make sure to replace `your_crate_name` with the actual name of your crate

fn bitboard_operations_benchmark(c: &mut Criterion) {
    c.bench_function("BitBoard AND operation", |b| {
        let bb1 = BitBoard::new(0xFFFF); // Example BitBoard
        let bb2 = BitBoard::new(0x0F0F); // Another BitBoard for operation
        b.iter(|| black_box(bb1) & black_box(bb2))
    });

    c.bench_function("BitBoard OR operation", |b| {
        let bb1 = BitBoard::new(0xFFFF); // Example BitBoard
        let bb2 = BitBoard::new(0x0F0F); // Another BitBoard for operation
        b.iter(|| black_box(bb1) | black_box(bb2))
    });

    // Add more benchmarks for other operations like XOR, NOT, etc.
}

criterion_group!(benches, bitboard_operations_benchmark);
criterion_main!(benches);