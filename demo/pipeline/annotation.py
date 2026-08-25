from ..config import load_config
config = load_config()

# Simple static mapping of mutation type index to annotation description
TYPE_TO_ANNOTATION = {
    "substitution": "Possible SNP (single nucleotide polymorphism)",
    "insertion": "Insertion event; may cause frameshift if length not multiple of 3",
    "deletion": "Deletion event; may cause frameshift if length not multiple of 3",
}

def annotate_mutations(mutation_types_tensor):
    """Convert a tensor of one‑hot mutation types to human‑readable annotations.
    Args:
        mutation_types_tensor (torch.Tensor): shape (n_mut, 3) one‑hot.
    Returns:
        List[str]: annotation strings for each mutation.
    """
    import torch
    if mutation_types_tensor.numel() == 0:
        return []
    # Convert one‑hot to index
    indices = torch.argmax(mutation_types_tensor, dim=1).tolist()
    index_to_type = {0: "substitution", 1: "insertion", 2: "deletion"}
    annotations = [TYPE_TO_ANNOTATION.get(index_to_type[i], "Unknown") for i in indices]
    return annotations
