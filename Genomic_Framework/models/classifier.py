"""
models/classifier.py
====================
Module 11 — MLP Classification Head

Purpose
-------
Map the integrated cross-attention feature vector produced by Module 10 to
final per-class logits using a multi-layer feed-forward network with
regularisation.

Architecture
------------
::

    Input [B, hidden_dim]
          ↓
    Linear(hidden_dim, hidden_dim // 2)
          ↓
    GELU
          ↓
    LayerNorm(hidden_dim // 2)    ← additional stabilisation
          ↓
    Dropout(0.3)
          ↓
    Linear(hidden_dim // 2, num_classes)
          ↓
    Output [B, num_classes]  ← raw logits (no softmax)

The raw logits are returned without a softmax so the module is compatible
with ``nn.CrossEntropyLoss`` during training and with ``torch.softmax`` in
:class:`GenomicReportGenerator` for inference.

Output Tensor Shape
-------------------
``[batch_size, num_classes]``
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main classifier class (Module 11)
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classification head.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input feature vector (output of fusion module).
    num_classes : int
        Number of target genomic-variant classes / phenotype categories.
    dropout : float
        Dropout probability in the hidden layer.  Defaults to 0.3 as per spec.
    extra_hidden_layers : list[int] | None
        Optional list of extra intermediate hidden-layer sizes to insert
        between the first hidden layer and the output layer.  When ``None``
        (default) the spec-required two-linear design is used.

    Output Tensor Shape
    -------------------
    ``[batch_size, num_classes]``

    Examples
    --------
    >>> clf = MLPClassifier(hidden_dim=512, num_classes=5)
    >>> x   = torch.randn(2, 512)   # output of CrossAttentionFusion
    >>> out = clf(x)
    >>> out.shape
    torch.Size([2, 5])
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_classes: int = 4,
        dropout: float = 0.3,
        extra_hidden_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if hidden_dim < 2:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be ≥ 2 so the intermediate "
                "layer hidden_dim // 2 is non-zero."
            )
        if num_classes < 2:
            raise ValueError(
                f"num_classes ({num_classes}) must be ≥ 2 for a valid "
                "classification task."
            )

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        intermediate_dim = hidden_dim // 2

        # ------------------------------------------------------------------
        # Build the classifier as a sequential stack
        # ------------------------------------------------------------------
        layers: List[nn.Module] = []

        # --- First hidden layer (spec-required) ---
        layers += [
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Dropout(p=dropout),
        ]

        # --- Optional extra hidden layers ---
        if extra_hidden_layers:
            in_dim = intermediate_dim
            for out_dim in extra_hidden_layers:
                layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.LayerNorm(out_dim),
                    nn.Dropout(p=dropout),
                ]
                in_dim = out_dim
            pre_output_dim = in_dim
        else:
            pre_output_dim = intermediate_dim

        # --- Output layer (raw logits) ---
        layers.append(nn.Linear(pre_output_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        self._init_weights()
        self._log_config()

    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform and zero bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_config(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MLPClassifier — hidden_dim=%d, intermediate=%d, "
            "num_classes=%d | params=%d",
            self.hidden_dim,
            self.hidden_dim // 2,
            self.num_classes,
            total_params,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map fused feature vector to class logits.

        Parameters
        ----------
        x : torch.Tensor
            Fused representation from :class:`CrossAttentionFusion`.
            Shape: ``[batch_size, hidden_dim]``.

        Returns
        -------
        torch.Tensor
            Raw class logits (no softmax applied).
            Shape: ``[batch_size, num_classes]``.
        """
        logits = self.classifier(x)               # [B, num_classes]
        logger.debug("MLPClassifier output shape: %s", list(logits.shape))
        return logits

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Softmax-normalised class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Shape: ``[batch_size, hidden_dim]``.

        Returns
        -------
        torch.Tensor
            Class probabilities summing to 1 per sample.
            Shape: ``[batch_size, num_classes]``.
        """
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return integer class indices with highest probability.

        Parameters
        ----------
        x : torch.Tensor
            Shape: ``[batch_size, hidden_dim]``.

        Returns
        -------
        torch.Tensor
            Predicted class indices.  Shape: ``[batch_size]``.
        """
        return self.forward(x).argmax(dim=-1)

    def __repr__(self) -> str:  # noqa: D401
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"MLPClassifier("
            f"hidden_dim={self.hidden_dim}, "
            f"num_classes={self.num_classes}, "
            f"params={total_params})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (module-level)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== MLPClassifier Smoke Test ===")

    clf = MLPClassifier(hidden_dim=512, num_classes=5, dropout=0.3)
    print(f"  Model: {clf}")

    batch_size = 4
    dummy_fused = torch.randn(batch_size, 512)
    print(f"  Input  shape: {list(dummy_fused.shape)}")

    with torch.no_grad():
        logits = clf(dummy_fused)
        probs  = clf.predict_proba(dummy_fused)
        preds  = clf.predict(dummy_fused)

    print(f"  Logits shape : {list(logits.shape)}")
    print(f"  Probs  shape : {list(probs.shape)}")
    print(f"  Predictions  : {preds.tolist()}")
    assert logits.shape == (batch_size, 5), "Shape mismatch!"
    print("  ✓ Output shape verified: [batch_size, num_classes]")
