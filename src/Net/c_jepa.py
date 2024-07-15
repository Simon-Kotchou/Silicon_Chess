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
