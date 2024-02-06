use std::collections::HashMap;
use ndarray::Array1;

// Assuming Move, GameState, and other relevant structs are defined elsewhere.
struct AlphaGoNode {
    parent: Option<Box<AlphaGoNode>>,
    children: HashMap<Move, Box<AlphaGoNode>>,
    visit_count: u32,
    q_value: f64,
    prior_value: f64,
    u_value: f64,
}

impl AlphaGoNode {
    fn new(parent: Option<Box<AlphaGoNode>>, probability: f64) -> Self {
        AlphaGoNode {
            parent,
            children: HashMap::new(),
            visit_count: 0,
            q_value: 0.0,
            prior_value: probability,
            u_value: probability,
        }
    }

    fn select_child(&self) -> (&Move, &Box<AlphaGoNode>) {
        self.children.iter().max_by(|(_, a), (_, b)| {
            (a.q_value + a.u_value)
                .partial_cmp(&(b.q_value + b.u_value))
                .unwrap()
        }).unwrap()
    }

    fn expand_children(&mut self, moves: Vec<Move>, probabilities: Array1<f64>) {
        for (mov, &prob) in moves.iter().zip(probabilities.iter()) {
            self.children.entry(mov.clone()).or_insert_with(|| {
                Box::new(AlphaGoNode::new(Some(Box::new(self.clone())), prob))
            });
        }
    }

    fn update_values(&mut self, leaf_value: f64) {
        if let Some(ref mut parent) = self.parent {
            parent.update_values(leaf_value);
        }

        self.visit_count += 1;
        self.q_value += leaf_value / self.visit_count as f64;

        if let Some(ref parent) = self.parent {
            let c_u: f64 = 5.0;
            self.u_value = c_u * (parent.visit_count as f64).sqrt() * self.prior_value / (1.0 + self.visit_count as f64);
        }
    }
}

struct AlphaGoMCTS {
    // Assuming policy_agent, fast_policy_agent, and value_agent are defined elsewhere
    root: Box<AlphaGoNode>,
    lambda_value: f64,
    num_simulations: u32,
    depth: u32,
    rollout_limit: u32,
    // Other fields for agents...
}

impl AlphaGoMCTS {
    fn new(/* Parameters for policy_agent, fast_policy_agent, value_agent */) -> Self {
        AlphaGoMCTS {
            root: Box::new(AlphaGoNode::new(None, 1.0)),
            lambda_value: 0.5,
            num_simulations: 1000,
            depth: 50,
            rollout_limit: 100,
            // Initialize agents...
        }
    }

    fn select_move(&mut self, game_state: &GameState) -> Move {
        for _ in 0..self.num_simulations {
            let mut current_state = game_state.clone();
            let mut node = &mut *self.root;
            for _ in 0..self.depth {
                if node.children.is_empty() {
                    // Expand children, assuming policy_probabilities returns moves and their probabilities
                    let (moves, probabilities) = self.policy_probabilities(&current_state);
                    node.expand_children(moves, probabilities);
                }

                let (selected_move, next_node) = node.select_child();
                current_state = current_state.apply_move(selected_move); // Assuming apply_move clones the state and applies the move
                node = next_node;
            }

            // Assuming value.predict returns a value for the state
            let value = self.value_predict(&current_state);
            let rollout = self.policy_rollout(&current_state);

            let weighted_value = (1.0 - self.lambda_value) * value + self.lambda_value * rollout;
            node.update_values(weighted_value);
        }

        let &most_visited_move = self.root.children.iter().max_by_key(|(_, child)| child.visit_count).unwrap().0;
        most_visited_move.clone();

    // Assuming GameState, Move, and other relevant types are defined elsewhere.
    fn policy_probabilities(&self, game_state: &GameState) -> (Vec<Move>, Array1<f64>) {
        // Mockup: replace with actual policy network prediction logic.
        // This example assumes the policy network returns a vector of (Move, probability) pairs.
        let predictions = self.policy.predict(game_state); // Assuming this method exists and returns predictions as Vec<(Move, f64)>
        let legal_moves = game_state.legal_moves(); // Assuming this method exists and returns Vec<Move>
        
        let mut moves = Vec::new();
        let mut probabilities = Vec::new();
        for (mov, prob) in predictions.iter().filter(|(m, _)| legal_moves.contains(m)) {
            moves.push(mov.clone());
            probabilities.push(*prob);
        }

        let total_prob: f64 = probabilities.iter().sum();
        let normalized_probabilities = Array1::from(probabilities).mapv(|p| p / total_prob);
        
        (moves, normalized_probabilities)
    }

    fn policy_rollout(&self, game_state: &GameState) -> f64 {
        let mut current_state = game_state.clone();
        for _ in 0..self.rollout_limit {
            if current_state.is_over() {
                break;
            }
            
            // Mockup: replace with actual fast policy network logic.
            let move_probabilities = self.rollout_policy.predict(&current_state); // Assuming this returns Vec<(Move, f64)>
            if let Some((best_move, _)) = move_probabilities.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                current_state = current_state.apply_move(best_move); // Assuming this method exists
            }
        }

        match current_state.winner() {
            Some(winner) if winner == current_state.next_player() => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }
}