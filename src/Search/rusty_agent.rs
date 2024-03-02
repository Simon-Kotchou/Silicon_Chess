use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::Array1;
use rustc_hash::FxHashMap; // Faster hashing

// Node structure with optimized data handling
struct AlphaGoNode {
    parent: Option<Rc<RefCell<AlphaGoNode>>>,
    children: FxHashMap<Move, Rc<RefCell<AlphaGoNode>>>,
    visit_count: u32,
    q_value: f64,
    prior_value: f64,
    u_value: f64,
}

impl AlphaGoNode {
    fn new(parent: Option<Rc<RefCell<Self>>>, probability: f64) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(AlphaGoNode {
            parent,
            children: FxHashMap::default(),
            visit_count: 0,
            q_value: 0.0,
            prior_value: probability,
            u_value: probability,
        }))
    }

    fn select_child(&self) -> (&Move, &Rc<RefCell<AlphaGoNode>>) {
        self.children.iter().max_by(|(_, a), (_, b)| {
            let a_val = a.borrow();
            let b_val = b.borrow();
            (a_val.q_value + a_val.u_value).partial_cmp(&(b_val.q_value + b_val.u_value)).unwrap()
        }).unwrap()
    }

    fn expand_children(&mut self, moves: Vec<Move>, probabilities: Array1<f64>) {
        for (mov, &prob) in moves.iter().zip(probabilities.iter()) {
            self.children.entry(mov.clone()).or_insert_with(|| {
                AlphaGoNode::new(Some(Rc::clone(&Rc::new(RefCell::new(*self)))), prob)
            });
        }
    }

    fn update_values(&mut self, leaf_value: f64) {
        if let Some(parent) = &self.parent {
            parent.borrow_mut().update_values(leaf_value);
        }

        self.visit_count += 1;
        self.q_value += (leaf_value - self.q_value) / self.visit_count as f64;

        if let Some(parent) = &self.parent {
            let parent_val = parent.borrow();
            let c_u: f64 = 5.0; // Exploration constant, tune as needed
            self.u_value = c_u * ((1.0 + parent_val.visit_count as f64).ln() / (1.0 + self.visit_count as f64)).sqrt() * self.prior_value;
        }
    }
}

// AlphaGoMCTS with optimizations for efficient tree search and rollout
struct AlphaGoMCTS {
    root: Rc<RefCell<AlphaGoNode>>,
    lambda_value: f64,
    num_simulations: u32,
    depth: u32,
    rollout_limit: u32,
    // Assuming the existence of policy, fast_policy, and value networks
}

impl AlphaGoMCTS {
    fn new(/* Parameters for initializing policy, fast_policy, value networks */) -> Self {
        AlphaGoMCTS {
            root: AlphaGoNode::new(None, 1.0),
            lambda_value: 0.5,
            num_simulations: 1000,
            depth: 50,
            rollout_limit: 100,
            // Initialize networks...
        }
    }

    // Implementation of `select_move`, `policy_probabilities`, `policy_rollout`...
    // Similar to the provided pseudocode but adapted for Rc<RefCell<>> and parallel execution
}
