"""
Genomic_Framework — Top-Level Package
======================================
Production-grade Genomic Sequence Classification Framework

Modules
-------
pipeline.input_qc          — Modules 1 & 2: QC and ingestion
pipeline.mutation_pipeline — Modules 3 & 6-9: Alignment + mutation processing
models.dnabert2            — Module 4: DNABERT-2 feature extractor
models.hyenadna            — Module 5: HyenaDNA feature extractor
models.fusion              — Module 10: Cross-attention feature fusion
models.classifier          — Module 11: MLP classification head
engine                     — Modules 12 & 13 + master orchestrator
"""

from .engine import (
    GenomicClassificationEngine,
    ExplainabilityEngine,
    GenomicReportGenerator,
)
from .models import (
    DNABERT2Extractor,
    HyenaDNAExtractor,
    CrossAttentionFusion,
    MLPClassifier,
)
from .pipeline import (
    GenomicSequenceIngester,
    SequenceAligner,
    MutationProcessor,
)

__version__ = "1.0.0"
__author__ = "Genomic Framework Engineering Team"

__all__ = [
    # Engine
    "GenomicClassificationEngine",
    "ExplainabilityEngine",
    "GenomicReportGenerator",
    # Models
    "DNABERT2Extractor",
    "HyenaDNAExtractor",
    "CrossAttentionFusion",
    "MLPClassifier",
    # Pipeline
    "GenomicSequenceIngester",
    "SequenceAligner",
    "MutationProcessor",
]
