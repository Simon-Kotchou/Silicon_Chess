import torch
import torch.nn as nn
import chess
import chess.pgn
import chess.svg
from IPython.display import SVG, display

class ViTEncoder(nn.Module):
    def __init__(self, board_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.board_embed = nn.Linear(board_size**2 * 12, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, board_size**2, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.board_embed(x)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        return x

class ViTPredictor(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        predictor_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(predictor_layer, num_layers)

    def forward(self, x, pos):
        pos_embed = self.pos_embed.expand(x.shape[0], pos.shape[1], -1)
        x = torch.cat([x, pos_embed], dim=1)
        x = self.transformer_encoder(x)
        return x[:, -pos.shape[1]:, :]

class IJEPA(nn.Module):
    def __init__(self, board_size, embed_dim, num_heads, enc_layers, pred_layers):
        super().__init__()
        self.board_size = board_size
        self.context_encoder = ViTEncoder(board_size, embed_dim, num_heads, enc_layers)
        self.target_encoder = ViTEncoder(board_size, embed_dim, num_heads, enc_layers)
        self.predictor = ViTPredictor(embed_dim, num_heads, pred_layers)

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

class VJEPA(nn.Module):
    def __init__(self, board_size, embed_dim, num_heads, enc_layers, pred_layers, seq_length):
        super().__init__()
        self.board_size = board_size
        self.seq_length = seq_length
        self.context_encoder = ViTEncoder(board_size, embed_dim, num_heads, enc_layers)
        self.target_encoder = ViTEncoder(board_size, embed_dim, num_heads, enc_layers)
        self.predictor = ViTPredictor(embed_dim, num_heads, pred_layers)

    def forward(self, x, context_mask, target_masks):
        # Reshape the input to have sequence length dimension
        x = x.view(-1, self.seq_length, self.board_size**2 * 12)

        # Mask the board sequence
        context_boards = x * context_mask.view(-1, self.seq_length, self.board_size**2)
        target_boards = x.unsqueeze(2) * target_masks.view(-1, self.seq_length, 1, self.board_size**2 * 12)

        # Encode
        context_reps = self.context_encoder(context_boards.view(-1, self.board_size**2 * 12))
        context_reps = context_reps.view(-1, self.seq_length, self.board_size**2, context_reps.shape[-1])
        with torch.no_grad():
            self.target_encoder.eval()
            target_reps = self.target_encoder(target_boards.view(-1, self.board_size**2 * 12)).view(-1, self.seq_length, target_masks.shape[1], self.board_size**2, context_reps.shape[-1])

        # Get position embeddings for target squares
        target_pos = torch.nonzero(target_masks.flatten(1))[:, 1].view(-1, self.seq_length, target_masks.shape[1])

        # Predict targets
        pred_reps = self.predictor(context_reps.view(-1, self.board_size**2, context_reps.shape[-1]), target_pos.view(-1, target_masks.shape[1]))
        pred_reps = pred_reps.view(-1, self.seq_length, target_masks.shape[1], self.board_size**2, pred_reps.shape[-1])

        # Compute L2 loss
        loss = torch.mean(torch.stack([torch.mean((pred_reps[i] - target_reps[i])**2) for i in range(pred_reps.shape[0])]))

        return loss

    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

class CJEPA(nn.Module):
    def __init__(self, board_size, embed_dim, num_heads, enc_layers, pred_layers, seq_length):
        super().__init__()
        self.ijepa = IJEPA(board_size, embed_dim, num_heads, enc_layers, pred_layers)
        self.vjepa = VJEPA(board_size, embed_dim, num_heads, enc_layers, pred_layers, seq_length)

    def forward(self, x, context_mask, target_masks):
        ijepa_loss = self.ijepa(x[:, -1], context_mask, target_masks)
        vjepa_loss = self.vjepa(x, context_mask.unsqueeze(1).repeat(1, x.shape[1], 1, 1), target_masks.unsqueeze(1).repeat(1, x.shape[1], 1, 1))
        return ijepa_loss + vjepa_loss

    def update_target_encoders(self, momentum=0.996):
        self.ijepa.update_target_encoder(momentum)
        self.vjepa.update_target_encoder(momentum)

def square_masking(board_size, num_targets=4, context_scale=(0.85, 1.0), target_scale=(0.15, 0.2), min_target_size=3):
    num_squares = board_size ** 2
    context_mask = torch.ones(board_size, board_size)
    target_masks = []
    occupied_squares = torch.zeros(board_size, board_size)

    # Sample target squares
    for _ in range(num_targets):
        target_size = torch.randint(min_target_size, int(num_squares * target_scale[1]) + 1, (1,)).item()
        available_squares = (1 - occupied_squares).nonzero()

        if len(available_squares) < target_size:
            break

        target_indices = available_squares[torch.randperm(len(available_squares))[:target_size]]
        target_mask = torch.zeros(board_size, board_size)
        target_mask[target_indices[:, 0], target_indices[:, 1]] = 1
        target_masks.append(target_mask)

        # Update occupied squares and context mask
        occupied_squares += target_mask
        context_mask *= (1 - target_mask)

    target_masks = torch.stack(target_masks) if target_masks else torch.empty(0, board_size, board_size)
    return context_mask, target_masks

def parse_pgn(pgn_path):
    pgn = open(pgn_path)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    boards = []
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    return boards

def board_to_tensor(board):
    piece_to_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    tensor = torch.zeros(12, 8, 8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            tensor[piece_to_idx[piece.symbol()], i//8, i%8] = 1
    return tensor.flatten()

if __name__ == "__main__":
    # Load PGN
    pgn_path = "game.pgn"  # replace with your PGN path
    boards = parse_pgn(pgn_path)

    # Convert boards to tensors
    board_tensors = [board_to_tensor(board) for board in boards]
    board_tensors = torch.stack(board_tensors)

    # Initialize models
    board_size = 8
    embed_dim = 128
    num_heads = 4
    enc_layers = 3
    pred_layers = 2
    seq_length = 5

    ijepa = IJEPA(board_size, embed_dim, num_heads, enc_layers, pred_layers)
    vjepa = VJEPA(board_size, embed_dim, num_heads, enc_layers, pred_layers, seq_length)
    cjepa = CJEPA(board_size, embed_dim, num_heads, enc_layers, pred_layers, seq_length)

    # Generate masks
    context_mask, target_masks = square_masking(board_size)

    # Compute losses
    ijepa_loss = ijepa(board_tensors[-1], context_mask, target_masks)
    vjepa_loss = vjepa(board_tensors[-seq_length:], context_mask.unsqueeze(0).repeat(seq_length, 1, 1), target_masks.unsqueeze(0).repeat(seq_length, 1, 1))
    cjepa_loss = cjepa(board_tensors[-seq_length:], context_mask, target_masks)

    print("IJEPA Loss:", ijepa_loss.item())
    print("VJEPA Loss:", vjepa_loss.item())
    print("CJEPA Loss:", cjepa_loss.item())