use std::collections::HashMap;

const CHARACTERS: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'p', 'n', 'r', 'k', 'q',
    'P', 'B', 'N', 'R', 'Q', 'K',
    'w', '.',
];

lazy_static! {
    static ref CHARACTERS_INDEX: HashMap<char, usize> = {
        let mut map = HashMap::new();
        for (index, &letter) in CHARACTERS.iter().enumerate() {
            map.insert(letter, index);
        }
        map
    };
}

const SPACES_CHARACTERS: &[char] = &['1', '2', '3', '4', '5', '6', '7', '8'];

pub const SEQUENCE_LENGTH: usize = 77;

pub fn tokenize(fen: &str) -> [u8; SEQUENCE_LENGTH] {
    let mut indices = Vec::new();

    let parts: Vec<&str> = fen.split(' ').collect();
    let (board, side, castling, en_passant, halfmoves_last, fullmoves) = (
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
    );

    let board = board.replace('/', "");
    let board = side.to_string() + &board;

    for char in board.chars() {
        if SPACES_CHARACTERS.contains(&char) {
            let count = char.to_digit(10).unwrap() as usize;
            indices.extend(std::iter::repeat(CHARACTERS_INDEX[&'.']).take(count));
        } else {
            indices.push(CHARACTERS_INDEX[&char]);
        }
    }

    if castling == "-" {
        indices.extend(std::iter::repeat(CHARACTERS_INDEX[&'.']).take(4));
    } else {
        for char in castling.chars() {
            indices.push(CHARACTERS_INDEX[&char]);
        }
        let padding_count = 4 - castling.len();
        indices.extend(std::iter::repeat(CHARACTERS_INDEX[&'.']).take(padding_count));
    }

    if en_passant == "-" {
        indices.extend(std::iter::repeat(CHARACTERS_INDEX[&'.']).take(2));
    } else {
        for char in en_passant.chars() {
            indices.push(CHARACTERS_INDEX[&char]);
        }
    }

    let halfmoves_last = format!("{:0<3}", halfmoves_last);
    for char in halfmoves_last.chars() {
        indices.push(CHARACTERS_INDEX[&char]);
    }

    let fullmoves = format!("{:0<3}", fullmoves);
    for char in fullmoves.chars() {
        indices.push(CHARACTERS_INDEX[&char]);
    }

    assert_eq!(indices.len(), SEQUENCE_LENGTH);

    indices.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let expected_tokens = [
            29, 23, 26, 28, 22, 26, 23, 29, 21, 21, 21, 21, 21, 21, 21, 21, 34, 34, 34, 34, 34, 34,
            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 25, 25, 25, 25, 25, 25, 25, 25, 27, 24, 25, 30,
            27, 24, 27, 32, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
            34, 34, 34, 34, 19, 34, 34, 34,
        ];

        let tokens = tokenize(fen);

        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    fn test_tokenize_no_castling() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
        let expected_tokens = [
            29, 23, 26, 28, 22, 26, 23, 29, 21, 21, 21, 21, 21, 21, 21, 21, 34, 34, 34, 34, 34, 34,
            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 25, 25, 25, 25, 25, 25, 25, 25, 34, 34, 34, 34,
            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 19, 34,
            34, 34,
        ];

        let tokens = tokenize(fen);

        assert_eq!(tokens, expected_tokens);
    }