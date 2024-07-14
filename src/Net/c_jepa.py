import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random
from torch.utils.data import Dataset, DataLoader

class ChessTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(13, d_model)  # 12 piece types + 1 for empty square
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        return self.transformer(x)