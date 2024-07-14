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
    
class ChessJEPAPredictor(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, x, target_mask):
        B, N, D = x.shape
        target_pos = self.pos_encoding.expand(B, -1, -1)
        target_pos = target_pos[target_mask].view(B, -1, D)
        return self.transformer(target_pos, x)

class ChessJEPA(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_predictor_layers=6):
        super().__init__()
        self.context_encoder = ChessTransformerEncoder(d_model, nhead, num_encoder_layers)
        self.target_encoder = ChessTransformerEncoder(d_model, nhead, num_encoder_layers)
        self.predictor = ChessJEPAPredictor(d_model, nhead, num_predictor_layers)

    def forward(self, x, context_mask, target_mask):
        context_input = x * context_mask
        context_repr = self.context_encoder(context_input)
        
        with torch.no_grad():
            target_input = x * target_mask
            target_repr = self.target_encoder(target_input)

        pred_repr = self.predictor(context_repr, target_mask)
        
        return pred_repr, target_repr[target_mask]

    def update_target_encoder(self, momentum=0.99):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

class ChessPuzzleDataset(Dataset):
    def __init__(self, pgn_files):
        self.puzzles = []
        for pgn_file in pgn_files:
            self.puzzles.extend(self.load_puzzles(pgn_file))

    def load_puzzles(self, pgn_file):
        puzzles = []
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                moves = list(game.mainline_moves())
                puzzles.append((board, moves))
        return puzzles

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        board, _ = self.puzzles[idx]
        return self.board_to_tensor(board)

    def board_to_tensor(self, board):
        piece_to_idx = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
        }
        tensor = torch.zeros(64, dtype=torch.long)
        for square, piece in board.piece_map().items():
            tensor[square] = piece_to_idx[piece.symbol()]
        return tensor

def create_jepa_masks(board_size=8, num_targets=4, context_scale=(0.85, 1.0), target_scale=(0.15, 0.2)):
    num_squares = board_size ** 2
    context_mask = torch.ones(num_squares, dtype=torch.bool)
    target_masks = torch.zeros(num_targets, num_squares, dtype=torch.bool)
    
    for i in range(num_targets):
        target_size = int(num_squares * random.uniform(*target_scale))
        target_start = random.randint(0, num_squares - target_size)
        target_masks[i, target_start:target_start+target_size] = True
        context_mask[target_start:target_start+target_size] = False
    
    context_size = int(num_squares * random.uniform(*context_scale))
    context_start = random.randint(0, num_squares - context_size)
    context_mask[context_start:context_start+context_size] = True
    
    return context_mask, target_masks
