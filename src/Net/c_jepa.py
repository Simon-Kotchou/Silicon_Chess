import torch
import chess
import chess.pgn
import chess.svg
from IPython.display import SVG, display

def visualize_masking(board, context_mask, target_masks):
    num_targets = target_masks.shape[0]

    display(SVG(chess.svg.board(board, size=400)))
    print("Original Board")

    context_board = board.copy()
    for i in range(64):
        if context_mask[i//8, i%8] == 0:
            piece = context_board.remove_piece_at(i)
    display(SVG(chess.svg.board(context_board, size=400)))
    print("Context Squares")

    for i in range(num_targets):
        mask_board = board.copy()
        for j in range(64):
            if target_masks[i, j//8, j%8] == 0:
                piece = mask_board.remove_piece_at(j)
        display(SVG(chess.svg.board(mask_board, size=400)))
        print(f"Target Mask {i+1}")

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

if __name__ == "__main__":
    # Load PGN
    pgn_path = "game2.pgn"  # replace with your PGN path
    boards = parse_pgn(pgn_path)

    # Select a random board position
    board_idx = torch.randint(len(boards), size=(1,)).item()
    board = boards[board_idx]

    # Generate masks
    context_mask, target_masks = square_masking(board_size=8)

    # Visualize masking
    visualize_masking(board, context_mask, target_masks)