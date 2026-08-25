import torch
from typing import List, Tuple
import yaml
from pathlib import Path

# Load config for potential settings (currently not used)
from ..config import load_config
config = load_config()

MUTATION_TYPES = {"substitution": 0, "insertion": 1, "deletion": 2}

def _one_hot_mutation(mut_type: str) -> List[int]:
    vec = [0, 0, 0]
    idx = MUTATION_TYPES.get(mut_type, 0)
    vec[idx] = 1
    return vec

def identify_mutations(query_seq: str, reference_seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identify simple mutations between `query_seq` and `reference_seq`.
    Returns two tensors:
        positions: (n_mutations, 1) normalized positions (float in [0,1])
        types: (n_mutations, 3) one‑hot mutation type
    The algorithm is a naïve diff based on `difflib.SequenceMatcher`.
    """
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, query_seq, reference_seq)
    ops = matcher.get_opcodes()
    positions = []
    types = []
    ref_len = len(reference_seq)
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        # For substitution, both i2-i1 and j2-j1 are 1
        if tag == "replace":
            for pos in range(i1, i2):
                norm = pos / ref_len if ref_len else 0.0
                positions.append([norm])
                types.append(_one_hot_mutation("substitution"))
        elif tag == "delete":
            for pos in range(i1, i2):
                norm = pos / ref_len if ref_len else 0.0
                positions.append([norm])
                types.append(_one_hot_mutation("deletion"))
        elif tag == "insert":
            # Insertions are reported at the position *after* the previous base
            pos = i1  # insertion point in reference
            norm = pos / ref_len if ref_len else 0.0
            # May be multiple inserted bases; treat as one mutation event
            positions.append([norm])
            types.append(_one_hot_mutation("insertion"))
    if not positions:
        # No mutations – return empty tensors
        return torch.empty((0, 1), dtype=torch.float32), torch.empty((0, 3), dtype=torch.float32)
    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    type_tensor = torch.tensor(types, dtype=torch.float32)
    return pos_tensor, type_tensor
