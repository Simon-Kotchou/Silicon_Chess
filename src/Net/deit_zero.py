import torch
import torch.nn as nn
import torch.nn.functional as F
from deit_style_model import DeiTEncoder

class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, num_moves):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, 2, kernel_size=1)  # Assuming a 1x1 conv for simplicity
        self.bn = nn.BatchNorm2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2 * 8 * 8, num_moves)  # Adjust the size based on your input

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ValueHead(nn.Module):
    def __init__(self, hidden_dim):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 1x1 conv
        self.bn = nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8, 256)  # Adjust the size based on your input
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class DeiTChessBot(nn.Module):
    def __init__(self, deit_config, num_moves):
        super(DeiTChessBot, self).__init__()
        self.deit = DeiTEncoder(deit_config)
        self.policy_head = PolicyHead(deit_config.hidden_size, num_moves)
        self.value_head = ValueHead(deit_config.hidden_size)

    def forward(self, pixel_values):
        # Assuming pixel_values is the processed input suitable for DeiT
        features = self.deit(pixel_values)  # Get the features from the DeiT encoder

        # Assuming the class token is at the first position
        class_token = features[:, 0, :].unsqueeze(-1).unsqueeze(-1)  # Add dummy spatial dimensions

        # Pass the class token through the policy and value heads
        policy_logits = self.policy_head(class_token)
        value = self.value_head(class_token)

        return policy_logits, value