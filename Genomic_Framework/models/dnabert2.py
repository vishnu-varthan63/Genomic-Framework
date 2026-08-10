"""
models/dnabert2.py
==================
Module 4 — DNABERT-2 Feature Extractor

Purpose
-------
Extract contextual, short-to-medium-range semantic embeddings from DNA
sequences using a transformer-based representation.  Since no pre-trained
DNABERT-2 weights are loaded at this stage, the class faithfully simulates
the Hugging Face DNABERT-2 architecture using standard PyTorch primitives,
ensuring the full forward pass can execute against dummy inputs without any
external checkpoint.

Architecture Simulation
-----------------------
Real DNABERT-2 uses a character-level k-mer tokeniser feeding a BERT-style
bidirectional encoder.  This module reproduces the structural skeleton:

1. **Token Embedding Layer**  — maps integer token IDs → dense vectors of
   size ``embed_dim``.
2. **Positional Encoding**    — sinusoidal fixed PE added to embeddings.
3. **Transformer Encoder**    — ``num_layers`` identical self-attention blocks,
   each with ``num_heads`` attention heads and a feed-forward sub-layer.
4. **Output Projection**      — a linear layer that projects each position's
   hidden state to 768 dimensions (matching DNABERT-2's published hidden size).

Output Tensor Shape
-------------------
``[batch_size, seq_len, 768]``
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters that mirror the DNABERT-2 published configuration
# ---------------------------------------------------------------------------
DNABERT2_HIDDEN_SIZE: int = 768   # official output dimension
VOCAB_SIZE: int = 4096            # k-mer vocabulary (simulated)


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------

class _SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learnable) sinusoidal positional encoding as described in
    *Attention Is All You Need* (Vaswani et al., 2017).

    Parameters
    ----------
    embed_dim : int
        Embedding / model dimension.
    max_len : int
        Maximum supported sequence length.
    dropout : float
        Dropout probability applied after adding PE to the embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the PE table once and register as a non-parameter buffer
        pe = torch.zeros(max_len, embed_dim)                       # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embed_dim)
        )                                                           # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)               # even dims
        pe[:, 1::2] = torch.cos(position * div_term)               # odd dims
        pe = pe.unsqueeze(0)                                        # [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[batch_size, seq_len, embed_dim]``.

        Returns
        -------
        torch.Tensor
            Same shape as input, with PE added and dropout applied.
        """
        x = x + self.pe[:, : x.size(1), :]   # type: ignore[index]
        return self.dropout(x)


class _TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder layer mimicking the DNABERT-2 block:

    ``MultiHeadSelfAttention → Add & Norm → FeedForward → Add & Norm``

    Parameters
    ----------
    embed_dim : int
        Model / embedding dimension.
    num_heads : int
        Number of self-attention heads.
    ffn_dim : int
        Inner dimension of the position-wise feed-forward network.
    dropout : float
        Dropout probability inside attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[batch_size, seq_len, embed_dim]``.
        key_padding_mask : torch.Tensor | None
            Boolean mask of shape ``[batch_size, seq_len]``.  ``True``
            positions are ignored by attention.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        # -- Self-attention sub-layer with residual --
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + attn_out)

        # -- FFN sub-layer with residual --
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Main extractor class (Module 4)
# ---------------------------------------------------------------------------

class DNABERT2Extractor(nn.Module):
    """
    DNABERT-2 feature-extraction module.

    Accepts a batch of token-ID sequences (integers) **or** a pre-tokenised
    float tensor and produces contextual embeddings of shape
    ``[batch_size, seq_len, 768]``.

    Parameters
    ----------
    vocab_size : int
        Token-vocabulary size.  Defaults to ``VOCAB_SIZE`` (4096).
    embed_dim : int
        Internal embedding dimension used throughout the transformer.
        The final output is projected to 768 regardless of this value.
    num_heads : int
        Number of multi-head self-attention heads.
    num_layers : int
        Number of stacked transformer encoder layers.
    ffn_dim : int
        Feed-forward inner dimension in each encoder block.
    max_seq_len : int
        Maximum sequence length for positional encoding.
    dropout : float
        Uniform dropout probability.
    output_dim : int
        Final output projection dimension.  Fixed at 768 to match DNABERT-2.

    Examples
    --------
    >>> model = DNABERT2Extractor()
    >>> token_ids = torch.randint(0, 4096, (2, 128))   # batch=2, seq=128
    >>> out = model(token_ids)
    >>> out.shape
    torch.Size([2, 128, 768])
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        output_dim: int = DNABERT2_HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        # Validate divisibility for multi-head attention
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # --- 1. Token embedding ---
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # --- 2. Positional encoding ---
        self.positional_encoding = _SinusoidalPositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # --- 3. Transformer encoder stack ---
        self.encoder_layers = nn.ModuleList([
            _TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # --- 4. Output projection to 768 ---
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._log_config(vocab_size, num_layers, num_heads)

    def _log_config(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DNABERT2Extractor — embed_dim=%d, output_dim=%d, "
            "layers=%d, heads=%d, vocab=%d | params=%.2fM",
            self.embed_dim,
            self.output_dim,
            num_layers,
            num_heads,
            vocab_size,
            total_params / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the DNABERT-2 extraction pipeline.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer tensor of shape ``[batch_size, seq_len]`` containing
            token indices.  Values must be in ``[0, vocab_size)``.
        key_padding_mask : torch.Tensor | None
            Optional boolean mask ``[batch_size, seq_len]`` where ``True``
            marks padding positions to be ignored by attention.

        Returns
        -------
        torch.Tensor
            Contextual embeddings of shape ``[batch_size, seq_len, 768]``.
        """
        # --- Embed tokens ---
        x = self.token_embedding(token_ids)          # [B, L, embed_dim]

        # --- Add positional encoding ---
        x = self.positional_encoding(x)              # [B, L, embed_dim]

        # --- Pass through transformer encoder stack ---
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)  # [B, L, embed_dim]

        # --- Project to 768-dim output ---
        x = self.output_projection(x)                # [B, L, 768]

        logger.debug("DNABERT2Extractor output shape: %s", list(x.shape))
        return x

    def __repr__(self) -> str:  # noqa: D401
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"DNABERT2Extractor("
            f"embed_dim={self.embed_dim}, "
            f"output_dim={self.output_dim}, "
            f"params={total_params:,})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (module-level)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== DNABERT2Extractor Smoke Test ===")
    model = DNABERT2Extractor(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=128,
    )
    print(f"  Model: {model}")

    batch_size, seq_len = 2, 128
    dummy_tokens = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    print(f"  Input  shape: {list(dummy_tokens.shape)}")

    with torch.no_grad():
        output = model(dummy_tokens)
    print(f"  Output shape: {list(output.shape)}")
    assert output.shape == (batch_size, seq_len, 768), "Shape mismatch!"
    print("  ✓ Output shape verified: [batch_size, seq_len, 768]")
