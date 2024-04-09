import torch
import torch.nn as nn
import chess
import chess.pgn
import random
from transformers import PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments

class CJEPAConfig(PretrainedConfig):
    model_type = "cjepa"

    def __init__(
        self,
        board_size=8,
        embed_dim=128,
        num_heads=4,
        enc_layers=3,
        pred_layers=2,
        seq_length=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.board_size = board_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.enc_layers = enc_layers
        self.pred_layers = pred_layers
        self.seq_length = seq_length

class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.board_embed = nn.Linear(config.board_size**2 * 12, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.board_size**2, config.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(config.embed_dim, config.num_heads, dim_feedforward=config.embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.enc_layers)

    def forward(self, x):
        x = self.board_embed(x)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        return x

class ViTPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        predictor_layer = nn.TransformerEncoderLayer(config.embed_dim, config.num_heads, dim_feedforward=config.embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(predictor_layer, config.pred_layers)

    def forward(self, x, pos):
        pos_embed = self.pos_embed.expand(x.shape[0], pos.shape[1], -1)
        x = torch.cat([x, pos_embed], dim=1)
        x = self.transformer_encoder(x)
        return x[:, -pos.shape[1]:, :]

class IJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.board_size = config.board_size
        self.context_encoder = ViTEncoder(config)
        self.target_encoder = ViTEncoder(config)
        self.predictor = ViTPredictor(config)

    def forward(self, x, context_mask, target_masks):
        # Mask the board
        context_board = x * context_mask.flatten()
        target_boards = x.unsqueeze(1) * target_masks.view(x.shape[0], -1, self.board_size**2 * 12)

        # Encode
        context_reps = self.context_encoder(context_board)
        with torch.no_grad():
            self.target_encoder.eval()
            target_reps = self.target_encoder(target_boards.view(-1, self.board_size**2 * 12)).view(x.shape[0], -1, self.board_size**2, context_reps.shape[-1])

        # Get position embeddings for target squares
        target_pos = torch.nonzero(target_masks.flatten(1))[:, 1].view(x.shape[0], -1)

        # Predict targets
        pred_reps = self.predictor(context_reps, target_pos)

        # Compute L2 loss
        loss = torch.mean(torch.stack([torch.mean((pred_reps[i] - target_reps[i])**2) for i in range(x.shape[0])]))

        return loss

    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)