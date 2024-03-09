import torch
import chess
import chess.pgn
import chess.svg
from IPython.display import SVG, display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_layers):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2) 
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
        return x[:, -pos.shape[1]:, :]  # return only the predicted targets

def multi_block_masking(img_size, patch_size, num_targets=4, context_scale=(0.85, 1.0), target_scale=(0.15, 0.2), target_aspect_ratio=(0.75, 1.5)):
    num_patches_per_dim = img_size // patch_size
    num_patches = num_patches_per_dim ** 2
    context_mask = torch.zeros(1, num_patches_per_dim, num_patches_per_dim)
    target_masks = []

    # Sample context block
    context_size = torch.rand(1) * (context_scale[1] - context_scale[0]) + context_scale[0]
    context_patches_per_dim = int(num_patches_per_dim * torch.sqrt(context_size))
    context_row = torch.randint(0, num_patches_per_dim - context_patches_per_dim + 1, (1,))
    context_col = torch.randint(0, num_patches_per_dim - context_patches_per_dim + 1, (1,))
    context_mask[0, context_row:context_row+context_patches_per_dim, context_col:context_col+context_patches_per_dim] = 1

    # Sample target blocks
    for _ in range(num_targets):
        target_size = torch.rand(1) * (target_scale[1] - target_scale[0]) + target_scale[0]
        aspect_ratio = torch.rand(1) * (target_aspect_ratio[1] - target_aspect_ratio[0]) + target_aspect_ratio[0]
        target_patches_per_dim = int(num_patches_per_dim * torch.sqrt(target_size))
        target_h = int(target_patches_per_dim * torch.sqrt(aspect_ratio))
        target_w = int(target_patches_per_dim / torch.sqrt(aspect_ratio))
        target_row = torch.randint(0, num_patches_per_dim - target_h + 1, (1,))
        target_col = torch.randint(0, num_patches_per_dim - target_w + 1, (1,))
        target_mask = torch.zeros(1, num_patches_per_dim, num_patches_per_dim)
        target_mask[0, target_row:target_row+target_h, target_col:target_col+target_w] = 1
        target_masks.append(target_mask)
        
        # Remove overlapping patches from context
        context_mask[0, target_row:target_row+target_h, target_col:target_col+target_w] = 0

    target_masks = torch.cat(target_masks, dim=0)
    return context_mask, target_masks

class IJEPA(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, 
                 num_heads, enc_layers, pred_layers):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.context_encoder = ViTEncoder(img_size, patch_size, embed_dim, num_heads, enc_layers)
        self.target_encoder = ViTEncoder(img_size, patch_size, embed_dim, num_heads, enc_layers)
        self.predictor = ViTPredictor(embed_dim, num_heads, pred_layers)
        
    def forward(self, x):
        # Sample masks
        context_masks, target_masks = multi_block_masking(self.img_size, self.patch_size)
        context_masks = context_masks.to(x.device).unsqueeze(0).repeat(x.shape[0], 1).view(x.shape[0], 1, self.img_size, self.img_size)
        target_masks = target_masks.to(x.device).unsqueeze(1).repeat(x.shape[0], 1, 1).view(x.shape[0], -1, self.img_size, self.img_size)

        # Mask images
        context_imgs = x * context_masks
        target_imgs = x.unsqueeze(1) * target_masks

        # Encode
        context_reps = self.context_encoder(context_imgs)
        with torch.no_grad():
            self.target_encoder.eval()
            target_reps = self.target_encoder(target_imgs.view(-1, 3, self.img_size, self.img_size)).view(x.shape[0], -1, self.img_size**2 // self.patch_size**2, context_reps.shape[-1])

        # Get position embeddings for target blocks
        target_pos = torch.nonzero(target_masks.flatten(2))[:, 2].view(x.shape[0], -1) 

        # Predict targets
        pred_reps = self.predictor(context_reps, target_pos)

        # Compute L2 loss
        loss = torch.mean(torch.stack([torch.mean((pred_reps[i] - target_reps[i])**2) for i in range(x.shape[0])]))

        return loss
    
    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

def visualize_masking(img, context_mask, target_masks, patch_size, alpha=0.35):
    num_targets = target_masks.shape[0]
    fig, axs = plt.subplots(1, 3 + num_targets, figsize=(20, 5))
    
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    context_img = np.zeros_like(img)
    context_mask = context_mask.squeeze().cpu().numpy()
    for i in range(context_mask.shape[0]):
        for j in range(context_mask.shape[1]):
            if context_mask[i, j] == 1:
                row = i
                col = j
                context_img[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = img[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
    axs[1].imshow(img, alpha=alpha)
    axs[1].imshow(context_img, alpha=1-alpha)
    axs[1].set_title("Context Block")
    axs[1].axis("off")

    target_img = np.zeros_like(img)
    for mask in target_masks:
        mask = mask.squeeze().cpu().numpy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    row = i
                    col = j
                    target_img[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = img[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
    axs[2].imshow(img, alpha=alpha)
    axs[2].imshow(target_img, alpha=1-alpha)
    axs[2].set_title("Target Blocks")
    axs[2].axis("off")

    for i in range(num_targets):
        target_block = np.zeros_like(img)
        mask = target_masks[i].squeeze().cpu().numpy()
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j, k] == 1:
                    row = j
                    col = k
                    target_block[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = img[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
        axs[3+i].imshow(img, alpha=alpha)
        axs[3+i].imshow(target_block, alpha=1-alpha)
        axs[3+i].set_title(f"Target Block {i+1}")
        axs[3+i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess images
    img_paths = ["test1.jpg", "test2.png", "test3.png"]  # replace with your image paths
    imgs = [Image.open(path).convert("RGB") for path in img_paths]
    imgs = [img.resize((224, 224)) for img in imgs]
    imgs = [torch.tensor(np.array(img)).permute(2, 0, 1).float().to(device) for img in imgs]
    imgs = torch.stack(imgs)
    print(imgs.shape)

    # Initialize model
    model = IJEPA(img_size=224, patch_size=16, embed_dim=768, num_heads=3, enc_layers=4, pred_layers=2).to(device)

    # Generate masks
    with torch.no_grad():
        context_masks, target_masks = multi_block_masking(model.img_size, model.patch_size)
        context_masks = context_masks.repeat(imgs.shape[0], 1, 1)
        target_masks = target_masks.repeat(imgs.shape[0], 1, 1, 1)

    # Visualize masking
    for i in range(imgs.shape[0]):
        img = imgs[i].permute(1, 2, 0).cpu().numpy() / 255.0
        context_mask = context_masks[i]
        target_mask = target_masks[i]
        visualize_masking(img, context_mask, target_mask, model.patch_size)

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