import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPieceEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.piece_embed = nn.Embedding(13, d_model)  # 6 piece types * 2 colors + empty
        self.position_embed = nn.Embedding(64, d_model)
        self.move_potential_encoder = nn.Linear(64, d_model)
        
    def forward(self, pieces, positions, move_potentials):
        return self.piece_embed(pieces) + self.position_embed(positions) + self.move_potential_encoder(move_potentials)

class ChessJEPAEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.piece_encoder = ChessPieceEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, pieces, positions, move_potentials, mask):
        x = self.piece_encoder(pieces, positions, move_potentials)
        x = x * (~mask).unsqueeze(-1).float()
        return self.transformer(x)
    
class ChessJEPAPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(d_model, 13)  # Predict piece type
        
    def forward(self, x, target_mask):
        x = self.transformer(target_mask.unsqueeze(-1).float(), x)
        return self.output(x)

class ChessJEPA(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_predictor_layers=3):
        super().__init__()
        self.context_encoder = ChessJEPAEncoder(d_model, nhead, num_encoder_layers)
        self.target_encoder = ChessJEPAEncoder(d_model, nhead, num_encoder_layers)
        self.predictor = ChessJEPAPredictor(d_model, nhead, num_predictor_layers)
        self.stockfish_predictor = nn.Linear(d_model, 1)  # Predict Stockfish evaluation
        
    def forward(self, pieces, positions, move_potentials, context_mask, target_mask):
        with torch.no_grad():
            target_repr = self.target_encoder(pieces, positions, move_potentials, target_mask)
        
        context_repr = self.context_encoder(pieces, positions, move_potentials, context_mask)
        pred_repr = self.predictor(context_repr, target_mask)
        stockfish_eval = self.stockfish_predictor(context_repr.mean(dim=1))
        
        return pred_repr, target_repr, stockfish_eval
    
    def update_target_encoder(self, momentum=0.99):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

def generate_chess_jepa_masks(batch_size, num_squares=64, num_targets=4, context_scale=(0.7, 0.9), target_scale=(0.1, 0.3)):
    context_mask = torch.ones(batch_size, num_squares, dtype=torch.bool)
    target_mask = torch.zeros(batch_size, num_squares, dtype=torch.bool)
    
    for i in range(batch_size):
        # Generate context mask
        context_size = int(num_squares * torch.empty(1).uniform_(*context_scale).item())
        context_start = torch.randint(0, num_squares - context_size + 1, (1,)).item()
        context_mask[i, context_start:context_start+context_size] = False
        
        # Generate target masks
        for _ in range(num_targets):
            target_size = int(num_squares * torch.empty(1).uniform_(*target_scale).item())
            target_start = torch.randint(0, num_squares - target_size + 1, (1,)).item()
            target_mask[i, target_start:target_start+target_size] = True
        
        # Ensure no overlap between context and target
        context_mask[i] |= target_mask[i]
    
    return context_mask, target_mask

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
