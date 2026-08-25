import torch
import yaml
from ..config import load_config
config = load_config()

# Simple default weights for mutation types
TYPE_WEIGHTS = {
    "substitution": 1.0,
    "insertion": 1.5,
    "deletion": 1.5,
}

def prioritize_mutations(mutation_positions: torch.Tensor, mutation_types: torch.Tensor) -> torch.Tensor:
    """Assign a priority score to each mutation.
    Args:
        mutation_positions (torch.Tensor): (n_mut, 1) normalized positions (0-1).
        mutation_types (torch.Tensor): (n_mut, 3) one‑hot mutation type.
    Returns:
        torch.Tensor: (n_mut,) priority scores (higher = more important).
    The score is a weighted sum of type weight and positional importance (e.g., central mutations get higher weight).
    """
    if mutation_positions.numel() == 0:
        return torch.tensor([])
    # Convert one‑hot to type string
    import torch
    type_idx = torch.argmax(mutation_types, dim=1).tolist()
    idx_to_type = {0: "substitution", 1: "insertion", 2: "deletion"}
    type_names = [idx_to_type[i] for i in type_idx]
    # Type weight vector
    type_weights = torch.tensor([TYPE_WEIGHTS.get(t, 1.0) for t in type_names], dtype=torch.float32)
    # Positional weight: give higher score to mutations near the middle of the sequence
    pos = mutation_positions.squeeze(1)
    positional_weight = 1.0 - torch.abs(pos - 0.5) * 2.0  # ranges from 0 at ends to 1 at centre
    # Final priority = type_weight * (0.7 + 0.3 * positional_weight)
    final_score = type_weights * (0.7 + 0.3 * positional_weight)
    return final_score
