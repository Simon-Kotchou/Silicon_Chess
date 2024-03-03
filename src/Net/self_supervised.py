import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deit_style_model import DeiTEncoder

class ChessMIM(nn.Module):
    def __init__(self, deit_config, embedding_dim):
        super().__init__()
        self.encoder = DeiTEncoder(deit_config)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(deit_config.hidden_size, deit_config.hidden_size),
            nn.ReLU(),
            nn.Linear(deit_config.hidden_size, deit_config.hidden_size),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(deit_config.hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = self.apply_mask(x, mask)
        features = self.encoder(x)
        if mask is not None:
            masked_features = features[mask]
            reconstructed = self.decoder(masked_features)
            projected = self.projection_head(features)
            return reconstructed, projected, features
        return features

    def apply_mask(self, x, mask):
        masked_x = x.clone()
        masked_x[mask] = self.mask_token.expand(-1, mask.sum(), -1)
        return masked_x

def compute_loss(reconstructed, projected, features, mask, criterion_mse, criterion_info_nce):
    loss_mse = criterion_mse(reconstructed, features[mask])
    loss_info_nce = criterion_info_nce(projected)
    return loss_mse + loss_info_nce

def info_nce_loss(features, temperature=0.1):
    labels = torch.arange(features.size(0)).to(features.device)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    return F.cross_entropy(similarity_matrix, labels)

def random_masking(x, mask_prob=0.15):
    mask = torch.rand(x.size()[:-1], device=x.device) < mask_prob
    return mask

def train_mim(model, data_loader, optimizer, device, criterion_mse, criterion_info_nce):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        chess_boards = batch.to(device)
        mask = random_masking(chess_boards)
        reconstructed, projected, original_features = model(chess_boards, mask)
        loss = compute_loss(reconstructed, projected, original_features, mask, criterion_mse, criterion_info_nce)
        loss.backward()
        optimizer.step()

class SpatialEmbeddingLayer(nn.Module):
    def __init__(self, num_planes, embedding_dim):
        super().__init__()
        self.piece_embeddings = nn.Parameter(torch.randn(num_planes - 1, embedding_dim))  # Exclude the empty square plane
        self.position_embeddings = nn.Parameter(torch.randn(64, embedding_dim))  # 64 squares

    def forward(self, x):
        piecewise_embedding = torch.einsum('bchw,ed->bchwde', x[:, :-1, :, :], self.piece_embeddings)  # Exclude the empty square plane
        spatial_embedding = self.position_embeddings.view(1, 1, 8, 8, -1).expand_as(piecewise_embedding)
        return piecewise_embedding + spatial_embedding
    
class PieceRelationshipModule(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.relationship_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, spatial_embeddings):
        # Analyze pairwise relationships between pieces on the board
        relationships = torch.einsum('bchwde,bshwde->bcshwe', spatial_embeddings, spatial_embeddings)
        encoded_relationships = self.relationship_encoder(relationships)
        return encoded_relationships.sum(dim=[2, 3, 5])  # Sum over spatial dimensions and embedding dimension

class AnticipationMechanism(nn.Module):
    def __init__(self, num_planes, anticipation_dim):
        super().__init__()
        self.anticipation_head = nn.Sequential(
            nn.Conv2d(num_planes, anticipation_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(anticipation_dim, anticipation_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.anticipation_head(x)
    
class ChessSSEA(nn.Module):
    def __init__(self, num_planes, embedding_dim, anticipation_dim):
        super().__init__()
        self.spatial_embedding_layer = SpatialEmbeddingLayer(num_planes, embedding_dim)
        self.piece_relationship_module = PieceRelationshipModule(embedding_dim)
        self.anticipation_mechanism = AnticipationMechanism(num_planes, anticipation_dim)

    def forward(self, x):
        spatial_embeddings = self.spatial_embedding_layer(x)
        relationships = self.piece_relationship_module(spatial_embeddings)
        anticipation = self.anticipation_mechanism(x)
        return spatial_embeddings, relationships, anticipation