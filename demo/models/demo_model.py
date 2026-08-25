import yaml
from pathlib import Path
import torch
import torch.nn as nn
from .sequence_encoder import KmerEncoder
from .feature_extractor import FeatureExtractor
from .mutation_encoder import MutationEncoder
from .cross_attention import CrossAttentionFusion
from .classifier import Classifier

# Load demo configuration
from ..config import load_config
config = load_config()

class DemoModel(nn.Module):
    """End‑to‑end lightweight demo model.
    It expects a DNA string (already QC‑cleaned) and a reference sequence.
    The pipeline:
        seq_emb = KmerEncoder -> FeatureExtractor
        mut_pos, mut_type = mutation identification (handled outside)
        mut_emb = MutationEncoder (may be empty)
        fused = CrossAttentionFusion(seq_feat, mut_emb)
        probs = Classifier(fused)
    """
    def __init__(self):
        super().__init__()
        embed_dim = config.get("embedding_dim", 64)
        self.seq_encoder = KmerEncoder(k=config.get("kmer_size", 3), embed_dim=embed_dim)
        self.feat_extractor = FeatureExtractor(input_dim=embed_dim, hidden_dim=config.get("feature_dim", 128))
        self.mut_encoder = MutationEncoder(embed_dim=embed_dim)
        self.cross_attn = CrossAttentionFusion(embed_dim=embed_dim, heads=config.get("attention_heads", 4))
        self.classifier = Classifier(
            embed_dim=embed_dim,
            hidden_dim=config.get("classifier_hidden_dim", 64),
            num_classes=config.get("num_classes", 4),
        )

    def forward(self, seq_str: str, mut_positions: torch.Tensor, mut_types: torch.Tensor):
        """Forward pass.
        Args:
            seq_str: raw DNA string.
            mut_positions: Tensor (batch, n_mut, 1) normalized positions.
            mut_types: Tensor (batch, n_mut, 3) one‑hot mutation types.
        Returns:
            probs: Tensor (batch, num_classes)
        """
        # Sequence encoding
        seq_emb = self.seq_encoder(seq_str).unsqueeze(0)  # (1, embed_dim)
        seq_feat = self.feat_extractor(seq_emb)  # (1, feature_dim)
        # Ensure seq_feat shape matches embed_dim for attention
        # If feature_dim differs, project back to embed_dim
        if seq_feat.shape[1] != self.cross_attn.attn.embed_dim:
            proj = nn.Linear(seq_feat.shape[1], self.cross_attn.attn.embed_dim).to(seq_feat.device)
            seq_feat = proj(seq_feat)
        # Mutation encoding (may be empty)
        if mut_positions.numel() == 0:
            # Create a dummy mutation tensor so attention receives at least one token
            # Use a zero tensor of shape (1, 1, embed_dim)
            mut_emb = torch.zeros((1, 1, self.cross_attn.attn.embed_dim), device=seq_feat.device)
        else:
            mut_emb = self.mut_encoder(mut_positions, mut_types)  # (batch, n_mut, embed_dim)
        # Cross‑attention fusion
        fused = self.cross_attn(seq_feat.unsqueeze(1), mut_emb)  # (batch, embed_dim)
        # Classification
        probs = self.classifier(fused)
        return probs
