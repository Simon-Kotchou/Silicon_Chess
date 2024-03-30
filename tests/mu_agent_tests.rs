#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_expand() {
        let state = Array1::zeros(BOARD_SIZE * BOARD_SIZE);
        let mut node = Node::new(state);

        let policy = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let reward = 1.0;
        let value = 0.5;

        node.expand(policy, reward, value);

        assert_eq!(node.reward, reward);
        assert_eq!(node.value, value);
        assert_eq!(node.policy.unwrap(), policy);
        assert_eq!(node.visit_count, 1);
        assert_eq!(node.children.len(), 4);
    }

    #[test]
    fn test_node_select_child() {
        let state = Array1::zeros(BOARD_SIZE * BOARD_SIZE);
        let mut node = Node::new(state);

        let policy = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let reward = 1.0;
        let value = 0.5;

        node.expand(policy, reward, value);

        let (action, child) = node.select_child(1.0);
        assert_eq!(action, 3);
        assert_eq!(child.borrow().visit_count, 0);
    }

    #[test]
    fn test_node_update_value() {
        let state = Array1::zeros(BOARD_SIZE * BOARD_SIZE);
        let mut node = Node::new(state);

        let value = 0.5;
        node.update_value(value);

        assert_eq!(node.value, value);
        assert_eq!(node.visit_count, 1);

        let new_value = 0.8;
        node.update_value(new_value);

        assert_eq!(node.value, (value + new_value) / 2.0);
        assert_eq!(node.visit_count, 2);
    }

    #[test]
    fn test_muzero_search() {
        let state = Array1::zeros(BOARD_SIZE * BOARD_SIZE);
        let cjepa_model = CJEPAModel::new();
        let agent = MuZeroAgent::new(&cjepa_model);

        let num_simulations = 100;
        let action = muzero_search(&agent, state, num_simulations);

        // Assert that the selected action is within the valid range
        assert!(action < BOARD_SIZE * BOARD_SIZE);
    }

    // Add more tests for other functions and scenarios
}