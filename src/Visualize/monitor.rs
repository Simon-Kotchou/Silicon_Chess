use piston_window::*;
use graphics::math::{Vec2d, add};
use std::time::Instant;

struct ChessMonitor {
    window: PistonWindow,
    search_tree: SearchTree,
    metrics: Metrics,
}

struct SearchTree {
    nodes: Vec<TreeNode>,
    // Additional fields to represent connections, depth, etc.
}

struct TreeNode {
    position: Vec2d<f64>, // Could represent board state, move, etc.
    evaluation: f64,      // Evaluation score for the node
    children: Vec<TreeNode>,
    // Other relevant fields
}

struct Metrics {
    win_rate: f64,
    elo_level: i32,
    // Other metrics like time per move, positions evaluated, etc.
}

impl ChessMonitor {
    fn new(width: f64, height: f64) -> ChessMonitor {
        let window: PistonWindow = WindowSettings::new("Chess Engine Monitor", [width, height])
            .exit_on_esc(true)
            .build()
            .unwrap_or_else(|e| panic!("Failed to build PistonWindow: {}", e));

        ChessMonitor {
            window,
            search_tree: SearchTree::new(), // Initialize an empty search tree
            metrics: Metrics::default(),    // Initialize default metrics
        }
    }

    fn update(&mut self) {
        // Update the search tree structure based on your engine's current exploration
        // This could involve adding new nodes, updating evaluations, etc.

        // Update metrics based on the latest engine data
        // e.g., self.metrics.win_rate = calculate_new_win_rate();

    }

    fn draw(&mut self, ctx: Context, g: &mut G2d) {
        // Clear the background
        clear([0.15, 0.17, 0.17, 1.0], g);

        // Draw the search tree
        self.search_tree.draw(ctx, g);

        // Draw metrics and other information
        self.metrics.draw(ctx, g);
    }

    // Additional methods for manipulating the search tree, adding metrics, etc.
}

impl SearchTree {
    fn new() -> Self {
        // Initialize an empty search tree
    }

    fn draw(&self, ctx: Context, g: &mut G2d) {
        // Logic to draw the search tree
        // Iterate through nodes and draw them with lines connecting to their children
    }
}

impl TreeNode {

    fn new(children: Vec<TreeNode>, evaluation: f64) -> Self {
        // Assuming each TreeNode has a list of children and associated evaluation scores
        children: Vec<TreeNode>;
        evaluation: f64; // Evaluation of the current node
    }

    fn draw(&self, ctx: Context, g: &mut G2d) {
        // Logic to draw a single tree node
        // This could involve drawing a circle or rectangle at the node's position,
        // and possibly adding text to display the node's evaluation score or other data
    }

    // Method to calculate the "sharpness" of the current node
    fn calculate_sharpness(&self) -> f64 {
        let scores: Vec<f64> = self.children.iter().map(|child| child.evaluation).collect();

        // Calculate the standard deviation of the scores
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance: f64 = scores.iter().map(|score| (*score - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_deviation: f64 = variance.sqrt();

        std_deviation // Higher values indicate "sharper" positions
    }

    // Method to evaluate the clarity of the top k moves
    fn evaluate_move_clarity(&self, k: usize) -> f64 {
        let mut top_moves = self.children.iter().map(|child| child.evaluation).collect::<Vec<f64>>();
        // Sort the moves in descending order based on evaluation
        top_moves.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Get the top k moves, ensuring we don't exceed the bounds
        let top_k_moves = top_moves.iter().take(k.min(top_moves.len()));

        // Calculate the difference between the top move and the k-th move
        if let Some(top_move) = top_k_moves.clone().next() {
            if let Some(kth_move) = top_k_moves.last() {
                return top_move - kth_move; // Larger values indicate more clarity in the position
            }
        }

        0.0 // Default to 0.0 if there are not enough moves to compare
    }
}
impl Metrics {
    // Initialize default metrics
    fn default() -> Self {
        Metrics {
            win_rate: 0.0,
            elo_level: 1500, // Example default ELO
            // Initialize other metrics as needed
        }
    }

    fn draw(&self, ctx: Context, g: &mut G2d) {
        // Logic to display metrics on the screen
        // This could involve drawing text or simple bar graphs to represent various metrics
    }
}

fn main() {
    let mut monitor = ChessMonitor::new();

    while let Some(event) = monitor.window.next() {
        monitor.update();

        monitor.window.draw_2d(&event, |ctx, g, _device| {
            clear([0.0, 0.0, 0.0, 1.0], g); // Clear the screen

            // Call the draw method to render the search tree and metrics
            monitor.draw();
        });
    }
}