"""
Genomic Framework — Models Package
====================================
Exposes all PyTorch feature-extraction, fusion, and classification modules.
"""

from .dnabert2 import DNABERT2Extractor
from .hyenadna import HyenaDNAExtractor
from .fusion import CrossAttentionFusion
from .classifier import MLPClassifier

__all__ = [
    "DNABERT2Extractor",
    "HyenaDNAExtractor",
    "CrossAttentionFusion",
    "MLPClassifier",
]
