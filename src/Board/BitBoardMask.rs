use crate::bitboard::BitBoard;
use crate::square::Square;

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