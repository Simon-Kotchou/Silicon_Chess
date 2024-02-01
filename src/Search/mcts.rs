use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub struct MCTS {
    root: Arc<Mutex<Node>>,
    max_iterations: u64,
    exploration_constant: f64,
    zobrist: Zobrist,
    valuator: Arc<RwLock<TorchModel>>, // Add valuator to MCTS
}

#[derive(Debug, Clone)]
struct Node {
    state: GameState,
    parent: Option<usize>,
    children: Vec<Arc<Mutex<Node>>>,
    wins: u64,
    visits: u64,
}

impl MCTS {
    pub fn new(initial_state: GameState, max_iterations: u64, exploration_constant: f64, zobrist: Zobrist, valuator: TorchModel) -> Self {
        MCTS {
            root: Arc::new(Mutex::new(Node::new(initial_state))),
            max_iterations,
            exploration_constant,
            zobrist,
            valuator: Arc::new(RwLock::new(valuator)),
        }
    }

    pub fn search(&self) -> Option<GameState> {
        (0..self.max_iterations)
            .into_par_iter()
            .for_each(|_| {
                let selected_node_idx = self.select();
                let expanded_node_idx = self.expand(selected_node_idx);
                let playout_result = self.simulate(expanded_node_idx);
                self.backpropagate(expanded_node_idx, playout_result);
            });

        let root = self.root.lock().unwrap();
        root.best_child().map(|child| child.lock().unwrap().state.clone())
    }

    pub fn evaluate_board(&self, board: &ChessBoard) -> f64 {
        let input = board_to_tensor(board); // Convert the ChessBoard to a Tensor format that the model can understand
        let valuator = self.valuator.read().unwrap();
        let output = valuator.infer(&input).unwrap();
        output[0].double_value(&[]) // Get the evaluation score from the output tensor
    }

    fn select(&self) -> usize {
        let root = self.root.lock().unwrap();
        root.select_best_child(&self.exploration_constant)
    }

    fn expand(&self, node_idx: usize) -> usize {
        let mut root = self.root.lock().unwrap();
        let node = root.children[node_idx].clone();
        let new_state = node.lock().unwrap().state.random_child();
        let new_node = Arc::new(Mutex::new(Node::new(new_state)));
        node.lock().unwrap().children.push(new_node.clone());
        node.lock().unwrap().children.len() - 1
    }

    fn simulate(&self, node_idx: usize) -> f64 {
        let root = self.root.lock().unwrap();
        let node = &root.children[node_idx];
        node.lock().unwrap().state.random_playout()
    }

    fn backpropagate(&self, node_idx: usize, playout_result: f64) {
        let mut current_idx = node_idx;

        while let Some(parent_idx) = self.root.lock().unwrap().children[current_idx].lock().unwrap().parent {
            let current_node = self.root.lock().unwrap().children[current_idx].clone();
            let mut current_node_locked = current_node.lock().unwrap();
            current_node_locked.visits += 1;
            current_node_locked.wins += playout_result as u64;
            current_idx = parent_idx;
        }
    }
}

impl Node {
    fn new(state: GameState, zobrist: &Zobrist) -> Self {
        let hash = zobrist.hash(&state.chess_board);
        Node {
            state,
            parent: None,
            children: Vec::new(),
            wins: 0,
            visits: 0,
        }
    }

    fn select_best_child(&self, exploration_constant: &f64) -> usize {
        let mut best_score = f64::MIN;
        let mut best_child = 0;

        for (i, child) in self.children.iter().enumerate() {
            let child_score = child.score(exploration_constant);
            if child_score > best_score {
                best_score = child_score;
                best_child = i;
            }
        }

        best_child
    }

    fn best_child(&self) -> Option<usize> {
        if self.children.is_empty() {
            None
        } else {
            Some(self.children.iter().enumerate().max_by_key(|(_, child)| child.wins).unwrap().0)
        }
    }

    fn score(&self, exploration_constant: &f64) -> f64 {
        if self.visits == 0 {
            f64::MAX
        } else {
            (self.wins as f64 / self.visits as f64) + exploration_constant * (2.0 * (self.parent.unwrap().visits as f64).
            

fn board_to_tensor(board: &ChessBoard) -> Tensor {
    // Implement the logic to convert a ChessBoard into a Tensor format that the neural network can understand
}