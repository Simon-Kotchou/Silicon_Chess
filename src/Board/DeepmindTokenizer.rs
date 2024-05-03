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