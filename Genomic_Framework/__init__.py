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

import sys
import types

# Compatibility guard for Windows AppLocker/WDAC policy blocking optree._C.pyd
if "torch._dynamo" not in sys.modules:
    try:
        import optree  # noqa: F401
    except Exception:
        class _DummyDynamo(types.ModuleType):
            __file__ = "dummy.py"
            def disable(self, *args, **kwargs):
                if args and callable(args[0]):
                    return args[0]
                return lambda f: f
            def graph_break(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return lambda *args, **kwargs: None
        sys.modules["torch._dynamo"] = _DummyDynamo("torch._dynamo")

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
