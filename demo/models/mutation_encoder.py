import torch
import torch.nn as nn

class MutationEncoder(nn.Module):
    """Encodes a simple mutation representation.
    For the demo we represent a mutation as a tuple (pos, type) where:
      - pos: integer position (0‑based) normalized by sequence length
      - type: one‑hot of substitution, insertion, deletion (3‑dim)
    The encoder concatenates the normalized position and type and projects
    to a fixed‑dimensional vector.
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.fc = nn.Linear(1 + 3, embed_dim)  # pos + one‑hot type
        self.activation = nn.ReLU()

    def forward(self, positions, types):
        """positions: Tensor of shape (batch, n_mutations, 1)
            types: Tensor of shape (batch, n_mutations, 3) one‑hot
            Returns: Tensor (batch, n_mutations, embed_dim)"""
        x = torch.cat([positions, types], dim=-1)
        return self.activation(self.fc(x))
