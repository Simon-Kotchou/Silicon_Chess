import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ChessPosition:
    board_state: str  # A string representation of the chess board state
    vector_representation: List[float] = None  # Vector representation of the board state

@dataclass
class ModelOutput:
    policy_logits: List[float]  # Move probabilities
    value: float  # Expected outcome from this position
    hidden_state: List[float]  # The internal representation (vector) of the position


class ChessVectorDatabase:
    def __init__(self, vector_dim, use_gpu=False):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)  # Using L2 distance for similarity
        if use_gpu:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

    def add_vectors(self, vectors):
        """Add vectors to the database."""
        self.index.add(vectors)

    def search_vectors(self, query_vector, k=10):
        """Search the database for the k most similar vectors."""
        distances, indices = self.index.search(np.array([query_vector]), k)
        return distances, indices

def find_similar_positions(query_position: ChessPosition, k=5):
    query_vector = np.array([query_position.vector_representation]).astype(np.float32)
    distances, indices = faiss_index.search(query_vector, k)
    return [(positions[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
