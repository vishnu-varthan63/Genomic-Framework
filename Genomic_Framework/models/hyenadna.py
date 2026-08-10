"""
models/hyenadna.py
==================
Module 5 — HyenaDNA Feature Extractor

Purpose
-------
Capture long-range structural dependencies across extended genomic contexts
using implicit long-convolution architectures that operate in sub-quadratic
time — the defining property of the Hyena hierarchy.

Architecture Simulation
-----------------------
Real HyenaDNA (Nguyen et al., 2023) replaces the softmax attention of
transformers with **Hyena operators**: implicitly-parameterised, long
convolutions realised through structured state-space projections.  The core
recurrence is:

    y = H(u)  where  H is a depth-wise long convolution + element-wise gating

This module faithfully reproduces that structural skeleton using standard
PyTorch components so that the complete computational graph can be traced and
forward-passed against dummy inputs without external dependencies:

1. **Input Projection**     — maps ``[B, L, input_dim]`` → ``[B, L, d_model]``.
2. **Positional Filter Bank** — a learned bank of 1-D convolution filters
   simulating HyenaDNA's long-convolution sub-quadratic operators.
3. **Gating Mechanism**     — element-wise sigmoid gates that control
   information flow (multiplicative interaction, simulating Hyena's implicit
   filter modulation).
4. **Hyena Blocks**         — ``num_layers`` stacked operator blocks, each
   comprising filter convolution + gating + residual + layer norm.
5. **Output Projection**    — projects to ``d_model`` output dimension.

Output Tensor Shape
-------------------
``[batch_size, seq_len, d_model]``
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------

class _HyenaLongFilter(nn.Module):
    """
    Simulated Hyena long-convolution filter bank.

    In the real HyenaDNA implementation this is a structured SSM-based filter
    learnt via implicit neural representations.  Here we replicate its
    *interface* using a depth-wise 1-D causal convolution with a very long
    kernel — this preserves the inductive bias of long-range dependency
    modelling while remaining purely PyTorch.

    Parameters
    ----------
    channels : int
        Number of feature channels (= d_model for depth-wise operation).
    filter_order : int
        Length of the convolution kernel.  Higher values capture longer-range
        dependencies at the cost of memory.  Mirrors HyenaDNA's ``order``
        hyper-parameter.
    dropout : float
        Dropout applied to filter outputs.
    """

    def __init__(
        self,
        channels: int,
        filter_order: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.filter_order = filter_order

        # Depth-wise long convolution (groups=channels → channel-independent)
        self.long_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_order,
            padding=filter_order - 1,    # causal padding
            groups=channels,
            bias=False,
        )

        # Learnable decay envelope (mimics SSM-style exponential decay)
        self.decay_envelope = nn.Parameter(
            torch.linspace(1.0, 0.0, filter_order).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, filter_order]

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the long convolution filter.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[batch_size, seq_len, channels]``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        B, L, C = x.shape

        # Transpose to [B, C, L] for Conv1d
        x_t = x.transpose(1, 2)                # [B, C, L]

        # Apply long causal convolution
        out = self.long_conv(x_t)              # [B, C, L + filter_order - 1]
        out = out[:, :, :L]                    # Trim to original length [B, C, L]

        # Transpose back to [B, L, C]
        out = out.transpose(1, 2)              # [B, L, C]
        out = self.dropout(out)
        out = self.norm(out)
        return out


class _HyenaGate(nn.Module):
    """
    Element-wise gating mechanism that modulates the filtered signal.

    Computes:
        ``z = sigma(W_gate * x) ⊙ tanh(W_proj * x)``

    This is a structural analogue to HyenaDNA's multiplicative state
    interactions.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[batch_size, seq_len, d_model]``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        gate = torch.sigmoid(self.gate_proj(x))
        value = torch.tanh(self.value_proj(x))
        return gate * value


class _HyenaOperatorBlock(nn.Module):
    """
    A single Hyena operator block:

    ``x → LongFilter → Gate → x + residual → LayerNorm``

    Optionally preceded by a short depthwise convolution that simulates
    the local mixing present in HyenaDNA's short-range receptive field.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    filter_order : int
        Long-convolution kernel length.
    dropout : float
        Dropout inside filter and FFN.
    use_short_conv : bool
        Whether to prepend a short depthwise convolution (kernel=3) before
        the long filter for local feature extraction.
    """

    def __init__(
        self,
        d_model: int,
        filter_order: int = 64,
        dropout: float = 0.1,
        use_short_conv: bool = True,
    ) -> None:
        super().__init__()

        # Optional short convolution (local)
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.short_conv = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            )
            self.short_norm = nn.LayerNorm(d_model)

        # Long Hyena filter
        self.hyena_filter = _HyenaLongFilter(
            channels=d_model,
            filter_order=filter_order,
            dropout=dropout,
        )

        # Gating
        self.gate = _HyenaGate(d_model)

        # Post-block normalisation
        self.norm = nn.LayerNorm(d_model)

        # Feed-forward mixing layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[batch_size, seq_len, d_model]``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        # --- Optional short convolution ---
        if self.use_short_conv:
            x_sc = x.transpose(1, 2)                # [B, C, L]
            x_sc = self.short_conv(x_sc).transpose(1, 2)  # [B, L, C]
            x = self.short_norm(x + x_sc)

        # --- Long-range Hyena filter ---
        residual = x
        x = self.hyena_filter(x)

        # --- Gating ---
        x = self.gate(x)

        # --- Residual + norm ---
        x = self.norm(x + residual)

        # --- Position-wise FFN ---
        x = self.ffn_norm(x + self.ffn(x))

        return x


# ---------------------------------------------------------------------------
# Main extractor class (Module 5)
# ---------------------------------------------------------------------------

class HyenaDNAExtractor(nn.Module):
    """
    HyenaDNA feature-extraction module for long-range genomic dependency
    modelling.

    Parameters
    ----------
    input_dim : int
        Dimension of each input token vector.  Tokens can be integer IDs
        (triggering an internal embedding lookup) or continuous vectors
        (passed directly after projection).
    d_model : int
        Internal model dimension and final output dimension.
        Defaults to 256; use 512 for higher-capacity models.
    num_layers : int
        Number of stacked Hyena operator blocks.
    filter_order : int
        Long-convolution kernel length.  Higher values improve long-range
        modelling at the cost of memory.
    vocab_size : int
        Vocabulary size for the internal embedding lookup.  Set to 0 if
        the caller will supply continuous-valued input tensors directly.
    max_seq_len : int
        Maximum sequence length.
    dropout : float
        Uniform dropout probability.

    Output Tensor Shape
    -------------------
    ``[batch_size, seq_len, d_model]``

    Examples
    --------
    >>> extractor = HyenaDNAExtractor(d_model=256)
    >>> token_ids = torch.randint(0, 4096, (2, 128))
    >>> out = extractor(token_ids)
    >>> out.shape
    torch.Size([2, 128, 256])
    """

    def __init__(
        self,
        input_dim: int = 4096,           # vocabulary size when using int tokens
        d_model: int = 256,
        num_layers: int = 4,
        filter_order: int = 64,
        vocab_size: int = 4096,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.uses_embedding = vocab_size > 0

        # --- 1. Token embedding (integer inputs) ---
        if self.uses_embedding:
            self.token_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                padding_idx=0,
            )
        else:
            # Project continuous inputs to d_model
            self.input_proj = nn.Linear(input_dim, d_model)

        # --- 2. Learnable positional bias (HyenaDNA style — relative PE) ---
        self.pos_bias = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pos_bias, std=0.02)

        # --- 3. Hyena operator blocks ---
        self.blocks = nn.ModuleList([
            _HyenaOperatorBlock(
                d_model=d_model,
                filter_order=filter_order,
                dropout=dropout,
                use_short_conv=True,
            )
            for _ in range(num_layers)
        ])

        # --- 4. Final layer norm ---
        self.final_norm = nn.LayerNorm(d_model)

        self._log_config(num_layers, filter_order)

    def _log_config(self, num_layers: int, filter_order: int) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "HyenaDNAExtractor — d_model=%d, layers=%d, filter_order=%d | "
            "params=%.2fM",
            self.d_model,
            num_layers,
            filter_order,
            total_params / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the HyenaDNA extraction pipeline.

        Parameters
        ----------
        token_ids : torch.Tensor
            Integer tensor of shape ``[batch_size, seq_len]`` containing
            token indices, or a float tensor ``[batch_size, seq_len, C]``
            when ``vocab_size=0``.

        Returns
        -------
        torch.Tensor
            Long-range embeddings of shape ``[batch_size, seq_len, d_model]``.
        """
        # --- Embed tokens ---
        if self.uses_embedding and token_ids.dtype in (torch.int32, torch.int64):
            x = self.token_embedding(token_ids)    # [B, L, d_model]
        else:
            # Continuous input: project to d_model
            x = self.input_proj(token_ids.float()) # [B, L, d_model]

        # --- Add positional bias (truncated to actual seq length) ---
        L = x.size(1)
        x = x + self.pos_bias[:, :L, :]           # [B, L, d_model]

        # --- Pass through Hyena blocks ---
        for block in self.blocks:
            x = block(x)                           # [B, L, d_model]

        # --- Final normalisation ---
        x = self.final_norm(x)                     # [B, L, d_model]

        logger.debug("HyenaDNAExtractor output shape: %s", list(x.shape))
        return x

    def __repr__(self) -> str:  # noqa: D401
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"HyenaDNAExtractor("
            f"d_model={self.d_model}, "
            f"params={total_params:,})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (module-level)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== HyenaDNAExtractor Smoke Test ===")
    extractor = HyenaDNAExtractor(d_model=256, num_layers=4, filter_order=32)
    print(f"  Model: {extractor}")

    batch_size, seq_len = 2, 128
    dummy_tokens = torch.randint(0, 4096, (batch_size, seq_len))
    print(f"  Input  shape: {list(dummy_tokens.shape)}")

    with torch.no_grad():
        output = extractor(dummy_tokens)
    print(f"  Output shape: {list(output.shape)}")
    assert output.shape == (batch_size, seq_len, 256), "Shape mismatch!"
    print("  ✓ Output shape verified: [batch_size, seq_len, d_model]")
