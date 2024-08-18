import torch
import chess
import random
from transformers import ImageProcessingMixin
from typing import Dict, List, Optional, Union

class ChessImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values"]

    def __init__(self, do_augment=True):
        super().__init__()
        self.do_augment = do_augment
        self.piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                               'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    def __call__(self, chess_input: Union[str, chess.Board], return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if isinstance(chess_input, str):
            if len(chess_input.split()) > 1:  # Assume it's a PGN string
                board = chess.Board()
                moves = chess_input.split()
                for move in moves:
                    board.push_san(move)
            else:  # Assume it's a FEN string
                board = chess.Board(chess_input)
        elif isinstance(chess_input, chess.Board):
            board = chess_input
        else:
            raise ValueError("Input must be a FEN string, PGN string, or a chess.Board object")

        if self.do_augment:
            board = self.augment_board(board)

        tensor = self.board_to_tensor(board)
        
        data = {"pixel_values": tensor.unsqueeze(0)}  # Add batch dimension
        if return_tensors == "pt":
            return data
        elif return_tensors is None:
            return data
        else:
            raise ValueError(f"Unsupported return_tensors type: {return_tensors}")

    def augment_board(self, board: chess.Board) -> chess.Board:
        augmentations = [
            self.rotate_board,
            self.mirror_board,
            self.shift_board,
            self.add_random_pieces
        ]
        augmentation = random.choice(augmentations)
        return augmentation(board)

    def rotate_board(self, board: chess.Board) -> chess.Board:
        rotations = [0, 90, 180, 270]
        rotation = random.choice(rotations)
        new_board = chess.Board()
        new_board.clear()
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            if rotation == 90:
                new_square = chess.square(7 - rank, file)
            elif rotation == 180:
                new_square = chess.square(7 - file, 7 - rank)
            elif rotation == 270:
                new_square = chess.square(rank, 7 - file)
            else:
                new_square = square
            new_board.set_piece_at(new_square, piece)
        new_board.turn = board.turn
        new_board.castling_rights = board.castling_rights
        new_board.ep_square = board.ep_square
        new_board.halfmove_clock = board.halfmove_clock
        new_board.fullmove_number = board.fullmove_number
        return new_board

    def mirror_board(self, board: chess.Board) -> chess.Board:
        new_board = board.mirror()
        new_board.turn = not board.turn
        return new_board

    def shift_board(self, board: chess.Board) -> chess.Board:
        shift_x, shift_y = random.randint(-1, 1), random.randint(-1, 1)
        new_board = chess.Board()
        new_board.clear()
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            new_rank = (rank + shift_y) % 8
            new_file = (file + shift_x) % 8
            new_square = chess.square(new_file, new_rank)
            new_board.set_piece_at(new_square, piece)
        new_board.turn = board.turn
        new_board.castling_rights = board.castling_rights
        new_board.ep_square = board.ep_square
        new_board.halfmove_clock = board.halfmove_clock
        new_board.fullmove_number = board.fullmove_number
        return new_board

    def add_random_pieces(self, board: chess.Board) -> chess.Board:
        new_board = board.copy()
        empty_squares = list(set(chess.SQUARES) - set(board.piece_map().keys()))
        num_pieces_to_add = random.randint(1, 3)
        for _ in range(num_pieces_to_add):
            if empty_squares:
                square = random.choice(empty_squares)
                piece = random.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                color = random.choice([chess.WHITE, chess.BLACK])
                new_board.set_piece_at(square, chess.Piece(piece, color))
                empty_squares.remove(square)
        return new_board

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(16, 8, 8, dtype=torch.float32)
        
        # Set piece positions
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            tensor[self.piece_to_index[piece.symbol()]][7-rank][file] = 1
        
        # Add a channel for empty squares
        tensor[12] = 1 - tensor[:12].sum(dim=0)
        
        # Add color-agnostic piece positions
        tensor[13] = tensor[:6].sum(dim=0) + tensor[6:12].sum(dim=0)
        
        # Add turn channel
        tensor[14] = float(board.turn)
        
        # Add influence channel
        influence_channels = self.create_influence_channels(tensor)
        tensor[15] = influence_channels[12] - influence_channels[13]  # White influence - Black influence
        
        return tensor

    def create_influence_channels(self, board_tensor: torch.Tensor) -> torch.Tensor:
        influence_channels = torch.zeros(14, 8, 8, dtype=torch.float32)
        
        # Pawn influences (including en passant)
        white_pawn = board_tensor[0]
        black_pawn = board_tensor[6]
        influence_channels[0, :-1, 1:] += white_pawn[1:, :-1]  # Diagonal right
        influence_channels[0, :-1, :-1] += white_pawn[1:, 1:]  # Diagonal left
        influence_channels[6, 1:, 1:] += black_pawn[:-1, :-1]  # Diagonal right
        influence_channels[6, 1:, :-1] += black_pawn[:-1, 1:]  # Diagonal left
        
        # En passant
        if board_tensor[14, 0, 0] == 1:  # White to move
            ep_file = (board_tensor[12] == 0).float().sum(dim=0).argmax()
            if ep_file > 0:
                influence_channels[0, 2, ep_file-1] = 1
            if ep_file < 7:
                influence_channels[0, 2, ep_file+1] = 1
        else:
            ep_file = (board_tensor[12] == 0).float().sum(dim=0).argmax()
            if ep_file > 0:
                influence_channels[6, 5, ep_file-1] = 1
            if ep_file < 7:
                influence_channels[6, 5, ep_file+1] = 1
        
        # Knight influences
        knight_kernel = torch.tensor([
            [0,1,0,1,0],
            [1,0,0,0,1],
            [0,0,0,0,0],
            [1,0,0,0,1],
            [0,1,0,1,0]
        ], dtype=torch.float32)
        influence_channels[1] = torch.nn.functional.conv2d(board_tensor[1].unsqueeze(0).unsqueeze(0), knight_kernel.unsqueeze(0).unsqueeze(0), padding=2)[0, 0]
        influence_channels[7] = torch.nn.functional.conv2d(board_tensor[7].unsqueeze(0).unsqueeze(0), knight_kernel.unsqueeze(0).unsqueeze(0), padding=2)[0, 0]
        
        # Sliding piece influences (Bishop, Rook, Queen)
        for i, directions in enumerate([(1,1), (1,0), (1,1)]):  # Bishop, Rook, Queen
            white_influence = torch.zeros(8, 8, dtype=torch.float32)
            black_influence = torch.zeros(8, 8, dtype=torch.float32)
            for dx, dy in [directions, (-directions[0], directions[1]), (directions[0], -directions[1]), (-directions[0], -directions[1])]:
                ray = torch.zeros(8, 8, dtype=torch.float32)
                x, y = 3, 3
                while 0 <= x < 8 and 0 <= y < 8:
                    ray[x, y] = 1
                    x, y = x + dx, y + dy
                white_influence += torch.nn.functional.conv2d(board_tensor[i+2].unsqueeze(0).unsqueeze(0), ray.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
                black_influence += torch.nn.functional.conv2d(board_tensor[i+8].unsqueeze(0).unsqueeze(0), ray.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            influence_channels[i+2] = white_influence
            influence_channels[i+8] = black_influence
        
        # King influences
        king_kernel = torch.tensor([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ], dtype=torch.float32)
        influence_channels[5] = torch.nn.functional.conv2d(board_tensor[5].unsqueeze(0).unsqueeze(0), king_kernel.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
        influence_channels[11] = torch.nn.functional.conv2d(board_tensor[11].unsqueeze(0).unsqueeze(0), king_kernel.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
        
        # Add channels for total influence
        influence_channels[12] = influence_channels[:6].sum(dim=0)  # White total influence
        influence_channels[13] = influence_channels[6:12].sum(dim=0)  # Black total influence
        
        return influence_channels