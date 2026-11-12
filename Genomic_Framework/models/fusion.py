"""
models/fusion.py
================
Module 10 — Cross-Attention Feature Fusion

Purpose
-------
Execute dynamic multi-modal feature interaction, allowing the encoded
mutation vectors (derived from alignment and processing pipeline) to
query structural information from both DNABERT-2 and HyenaDNA
representations through a cross-attention mechanism.

Data-flow Summary
-----------------
::

    DNABERT-2 output  [B, L, 768]  ─┐
                                     ├─ Linear projections → [B, L, hidden_dim]
    HyenaDNA output   [B, L, D]   ─┘       ↓
                                     Sum/concat → context pool [B, L, hidden_dim]

    Mutation tensor [B, M, mut_dim] → Linear → [B, M, hidden_dim]
                                           ↓
                         Cross-Attention (Q=mutations, K=V=context)
                                           ↓
                            Output [B, M, hidden_dim]
                                           ↓
                         Global Average Pooling over M dim
                                           ↓
                            Fused vector [B, hidden_dim]

Output Tensor Shape
-------------------
``[batch_size, hidden_dim]``
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main fusion class (Module 10)
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Multi-modal cross-attention fusion module.

    Fuses per-position embeddings from DNABERT-2 and HyenaDNA with the
    encoded mutation feature tensor using a cross-attention mechanism where
    mutations act as queries and the joint sequence context acts as key-value
    memory.

    Parameters
    ----------
    dnabert2_dim : int
        Feature dimension of DNABERT-2 output (default 768).
    hyena_dim : int
        Feature dimension of HyenaDNA output.
    mutation_dim : int
        Feature dimension of each encoded mutation vector.
    hidden_dim : int
        Common projection dimension used throughout the fusion block.
        All sub-modules project to this dimension.
    num_heads : int
        Number of heads in the cross-attention layer.
    dropout : float
        Dropout probability inside projections and attention.
    context_merge : Literal["sum", "concat"]
        How to merge the two projected sequence representations.
        ``"sum"``    — element-wise addition (keeps shape ``[B, L, hidden_dim]``).
        ``"concat"`` — channel-wise concatenation followed by a linear
                       reduction back to ``hidden_dim``.
    pool_mode : Literal["mean", "max"]
        Global pooling strategy applied over the mutation dimension.

    Output Tensor Shape
    -------------------
    ``[batch_size, hidden_dim]``

    Examples
    --------
    >>> fusion = CrossAttentionFusion(hyena_dim=256, hidden_dim=512)
    >>> dna_feat  = torch.randn(2, 128, 768)   # DNABERT-2 output
    >>> hya_feat  = torch.randn(2, 128, 256)   # HyenaDNA output
    >>> mut_feat  = torch.randn(2, 32, 64)     # Mutation tensor
    >>> out = fusion(dna_feat, hya_feat, mut_feat)
    >>> out.shape
    torch.Size([2, 512])
    """

    def __init__(
        self,
        dnabert2_dim: int = 768,
        hyena_dim: int = 256,
        mutation_dim: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        context_merge: Literal["sum", "concat"] = "sum",
        pool_mode: Literal["mean", "max"] = "mean",
    ) -> None:
        super().__init__()

        # Validate head divisibility
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.context_merge = context_merge
        self.pool_mode = pool_mode

        # ------------------------------------------------------------------
        # 1. Sequence context projections
        # ------------------------------------------------------------------
        # Project DNABERT-2 features → hidden_dim
        self.dnabert2_proj = nn.Sequential(
            nn.Linear(dnabert2_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Project HyenaDNA features → hidden_dim
        self.hyena_proj = nn.Sequential(
            nn.Linear(hyena_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # If concat merge: reduce 2 * hidden_dim back to hidden_dim
        if context_merge == "concat":
            self.context_merge_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.context_merge_proj = nn.Identity()

        # ------------------------------------------------------------------
        # 2. Mutation feature projection
        # ------------------------------------------------------------------
        self.mutation_proj = nn.Sequential(
            nn.Linear(mutation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # 3. Cross-attention: Q = mutations, K = V = sequence context
        # ------------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # 4. Post-attention feed-forward refinement
        # ------------------------------------------------------------------
        self.post_attn_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # 5. Final output projection after pooling
        # ------------------------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._log_config(dnabert2_dim, hyena_dim, mutation_dim, num_heads)

    def _log_config(
        self,
        dnabert2_dim: int,
        hyena_dim: int,
        mutation_dim: int,
        num_heads: int,
    ) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CrossAttentionFusion — hidden_dim=%d, heads=%d, "
            "merge=%s, pool=%s | params=%.2fM",
            self.hidden_dim,
            num_heads,
            self.context_merge,
            self.pool_mode,
            total_params / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        dnabert2_features: torch.Tensor,
        hyena_features: torch.Tensor,
        mutation_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the fused genomic representation.

        Parameters
        ----------
        dnabert2_features : torch.Tensor
            DNABERT-2 contextual embeddings.
            Shape: ``[batch_size, seq_len, 768]``.
        hyena_features : torch.Tensor
            HyenaDNA long-range embeddings.
            Shape: ``[batch_size, seq_len, hyena_dim]``.
        mutation_features : torch.Tensor
            Encoded mutation tensor from the processing pipeline.
            Shape: ``[batch_size, max_mutations, mutation_dim]``.
        key_padding_mask : torch.Tensor | None
            Optional boolean mask for the context (sequence) positions.
            Shape: ``[batch_size, seq_len]``.  ``True`` marks padding.

        Returns
        -------
        torch.Tensor
            Fused representation. Shape: ``[batch_size, hidden_dim]``.
        """
        # --- 1. Project sequence representations to hidden_dim ---
        db2_ctx = self.dnabert2_proj(dnabert2_features)   # [B, L, H]
        hya_ctx = self.hyena_proj(hyena_features)          # [B, L, H]

        # --- 2. Merge sequence context streams ---
        if self.context_merge == "sum":
            context_pool = db2_ctx + hya_ctx               # [B, L, H]
        else:  # concat
            context_pool = self.context_merge_proj(
                torch.cat([db2_ctx, hya_ctx], dim=-1)
            )                                               # [B, L, H]

        # --- 3. Project mutation features ---
        mut_q = self.mutation_proj(mutation_features)      # [B, M, H]

        # --- 4. Cross-attention (Q=mutations, K=V=context) ---
        attn_out, _ = self.cross_attention(
            query=mut_q,
            key=context_pool,
            value=context_pool,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                   # [B, M, H]

        # Residual + norm
        mut_fused = self.attn_norm(mut_q + self.attn_dropout(attn_out))

        # --- 5. Post-attention FFN refinement ---
        mut_fused = self.ffn_norm(mut_fused + self.post_attn_ffn(mut_fused))
        # Shape still: [B, M, H]

        # --- 6. Global pooling over mutation dimension M → [B, H] ---
        if self.pool_mode == "mean":
            fused = mut_fused.mean(dim=1)
        else:  # max
            fused = mut_fused.max(dim=1).values

        # --- 7. Final output projection ---
        fused = self.output_proj(fused)                    # [B, H]

        logger.debug("CrossAttentionFusion output shape: %s", list(fused.shape))
        return fused

    def __repr__(self) -> str:  # noqa: D401
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"CrossAttentionFusion("
            f"hidden_dim={self.hidden_dim}, "
            f"params={total_params:,})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (module-level)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== CrossAttentionFusion Smoke Test ===")

    B, L, M = 2, 128, 32
    DNABERT2_DIM, HYENA_DIM, MUT_DIM, HIDDEN_DIM = 768, 256, 64, 512

    fusion = CrossAttentionFusion(
        dnabert2_dim=DNABERT2_DIM,
        hyena_dim=HYENA_DIM,
        mutation_dim=MUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=8,
    )
    print(f"  Model: {fusion}")

    dna_feat = torch.randn(B, L, DNABERT2_DIM)
    hya_feat = torch.randn(B, L, HYENA_DIM)
    mut_feat = torch.randn(B, M, MUT_DIM)

    print(f"  DNABERT-2 input shape  : {list(dna_feat.shape)}")
    print(f"  HyenaDNA  input shape  : {list(hya_feat.shape)}")
    print(f"  Mutation  input shape  : {list(mut_feat.shape)}")

    with torch.no_grad():
        out = fusion(dna_feat, hya_feat, mut_feat)

    print(f"  Fusion output shape    : {list(out.shape)}")
    assert out.shape == (B, HIDDEN_DIM), "Shape mismatch!"
    print("  ✓ Output shape verified: [batch_size, hidden_dim]")
