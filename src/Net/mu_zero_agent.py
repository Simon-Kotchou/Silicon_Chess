import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class Node:
    state: torch.Tensor
    action: int = None
    reward: float = 0.0
    value: float = 0.0
    policy: torch.Tensor = None
    visit_count: int = 0
    children: Dict[int, 'Node'] = field(default_factory=dict)

    def expand(self, policy, reward, value):
        self.reward = reward
        self.value = value
        self.policy = policy
        self.visit_count = 1
        for action in range(policy.shape[-1]):
            self.children[action] = Node(None)

    def select_child(self, c_puct=1.0):
        best_value = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            if child.visit_count == 0:
                q_value = 0.0
            else:
                q_value = child.reward + child.value

            u_value = c_puct * self.policy[action] * (self.visit_count ** 0.5) / (1 + child.visit_count)
            action_value = q_value + u_value

            if action_value > best_value:
                best_value = action_value
                best_action = action
                best_child = child

        return best_action, best_child

    def update_value(self, value):
        self.value = (self.visit_count * self.value + value) / (self.visit_count + 1)
        self.visit_count += 1

    def expanded(self):
        return len(self.children) > 0

board_size = 8  # Define the board size here

class MuZeroDynamics(nn.Module):
    def __init__(self, cjepa_model):
        super().__init__()
        self.cjepa = cjepa_model
        self.dynamics = nn.Sequential(
            nn.Linear(cjepa_model.embed_dim + board_size**2, cjepa_model.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cjepa_model.embed_dim, cjepa_model.embed_dim),
        )
        self.reward_head = nn.Linear(cjepa_model.embed_dim, 1)

    def forward(self, state, action):
        # Encode state and action
        state_embedding = self.cjepa.ijepa.context_encoder(state)
        action_embedding = torch.zeros(state.shape[0], board_size**2)
        action_embedding[torch.arange(state.shape[0]), action] = 1

        # Concatenate state and action embeddings
        state_action_embedding = torch.cat([state_embedding, action_embedding], dim=-1)

        # Predict next state and reward
        next_state_embedding = self.dynamics(state_action_embedding)
        reward = self.reward_head(next_state_embedding).squeeze(-1)

        return next_state_embedding, reward

class MuZeroAgent(nn.Module):
    def __init__(self, cjepa_model):
        super().__init__()
        self.dynamics = MuZeroDynamics(cjepa_model)
        self.prediction = MuZeroPrediction(cjepa_model)

    def forward(self, state, action):
        next_state_embedding, reward = self.dynamics(state, action)
        policy_logits, value = self.prediction(next_state_embedding)
        return next_state_embedding, reward, policy_logits, value

class MuZeroPrediction(nn.Module):
    def __init__(self, cjepa_model):
        super().__init__()
        self.policy_head = nn.Linear(cjepa_model.embed_dim, cjepa_model.board_size**4)
        self.value_head = nn.Linear(cjepa_model.embed_dim, 1)

    def forward(self, state_embedding):
        policy_logits = self.policy_head(state_embedding)
        value = self.value_head(state_embedding).squeeze(-1)
        return policy_logits, value

def muzero_search(agent, state, num_simulations):
    root = Node(state)
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = node.select_child()
            search_path.append(node)

        parent = search_path[-2]
        state = parent.state
        action = node.action

        # Expand the node using the prediction function
        next_state_embedding, reward, policy_logits, value = agent(state, action)
        node.expand(policy_logits, reward, value)

        # Backpropagate the value estimates
        for node in reversed(search_path):
            node.update_value()

    # Select the action with the highest visit count
    return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

def muzero_training(agent, replay_buffer, num_epochs, batch_size, lr):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch in replay_buffer.sample_batch(batch_size):
            obs_batch, action_batch, reward_batch, done_batch, policy_batch, value_batch = batch

            # Compute model predictions
            _, rewards, policy_logits, values = agent(obs_batch, action_batch)

            # Compute losses
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_batch)
            value_loss = torch.nn.functional.mse_loss(values, value_batch)
            reward_loss = torch.nn.functional.mse_loss(rewards, reward_batch)

            total_loss = policy_loss + value_loss + reward_loss

            # Update model parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()