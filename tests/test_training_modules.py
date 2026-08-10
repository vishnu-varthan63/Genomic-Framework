"""Smoke tests for all new modules."""
import json
import pathlib
import shutil
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

# ---- 1. Tokeniser ----
from Genomic_Framework.pipeline.preprocessor import (
    CLASS_LABELS,
    SUBTYPE_TO_CLASS,
    GenomicDataset,
    tokenise_sequence,
)

t = tokenise_sequence("ACGTACGTACGT", seq_len=10, vocab_size=4096)
assert t.shape == (10,)
assert t.dtype == torch.long
assert (t[:9] > 0).all()
# Determinism check
t2 = tokenise_sequence("ACGTACGTACGT", seq_len=10, vocab_size=4096)
assert t.tolist() == t2.tolist()
print("tokenise_sequence()        OK  shape=%s" % list(t.shape))

# ---- 2. Label table ----
assert len(CLASS_LABELS) == 4
assert SUBTYPE_TO_CLASS["H5N1"] == 3
assert SUBTYPE_TO_CLASS["H9N2"] == 1
assert SUBTYPE_TO_CLASS["healthy"] == 0
print("SUBTYPE_TO_CLASS / CLASS_LABELS OK")

# ---- 3. MetricsTracker ----
from Genomic_Framework.training.trainer import MetricsTracker

tracker = MetricsTracker(num_classes=4, class_labels=CLASS_LABELS)
logits = torch.tensor([[2.0, 0.5, 0.1, 0.0], [0.0, 0.1, 2.0, 0.0]])
labels = torch.tensor([0, 2])
tracker.update(logits, labels, loss=0.5)
m = tracker.compute()
assert m["accuracy"] == 1.0, "Expected 1.0 acc"
assert "macro_f1" in m
assert "confusion_matrix" in m
assert len(m["confusion_matrix"]) == 4
print(
    "MetricsTracker             OK  acc=%.4f  macro_f1=%.4f"
    % (m["accuracy"], m["macro_f1"])
)

# ---- 4. GenomicDataset with synthetic FASTA ----
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(
        ">seq1\nACGTACGTACGTACGTACGTACGTACGTACGT\n"
        ">seq2\nTTTTAAAACCCCGGGGTTTTAAAACCCCGGGG\n"
        ">seq3\nGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\n",
        encoding="utf-8",
    )
    ds = GenomicDataset(
        fasta_paths=[(fasta_file, 0)],
        seq_len=16,
        vocab_size=256,
        max_mutations=8,
        mutation_dim=64,
        max_seq_len_qc=512,
        cache_tokens=True,
    )
    assert len(ds) == 3, "Expected 3 samples"
    tok, mut, lbl = ds[0]
    assert tok.shape == (16,)
    assert mut.shape == (8, 64)
    assert lbl.item() == 0
    assert tok.dtype == torch.long
    assert mut.dtype == torch.float32
    assert 0 in ds._cache
    print(
        "GenomicDataset             OK  len=%d  tok=%s  mut=%s"
        % (len(ds), list(tok.shape), list(mut.shape))
    )

# ---- 5. GenomicTrainer instantiation ----
from Genomic_Framework.engine import GenomicClassificationEngine
from Genomic_Framework.training.trainer import GenomicTrainer


def _dummy_loader():
    toks = torch.randint(0, 256, (4, 16))
    muts = torch.zeros(4, 8, 64)
    lbls = torch.randint(0, 4, (4,))
    ds_d = TensorDataset(toks, muts, lbls)
    return DataLoader(ds_d, batch_size=2)


engine = GenomicClassificationEngine(
    seq_len=16, vocab_size=256, hyena_dim=64, hidden_dim=128,
    num_classes=4, max_mutations=8, mutation_dim=64,
)
ckpt_tmp = pathlib.Path("checkpoints_test_tmp")
trainer = GenomicTrainer(
    model=engine,
    train_loader=_dummy_loader(),
    val_loader=_dummy_loader(),
    test_loader=_dummy_loader(),
    num_epochs=1,
    patience=2,
    checkpoint_dir=ckpt_tmp,
)
print("GenomicTrainer             OK  %r" % trainer)

# ---- 6. train_epoch + validate_epoch ----
trainer._current_epoch = 1
train_m = trainer.train_epoch()
val_m   = trainer.validate_epoch(trainer.val_loader)
print(
    "train_epoch()              OK  loss=%.4f  acc=%.4f"
    % (train_m["loss"], train_m["accuracy"])
)
print(
    "validate_epoch()           OK  loss=%.4f  acc=%.4f  f1=%.4f"
    % (val_m["loss"], val_m["accuracy"], val_m["macro_f1"])
)

# ---- 7. save / load checkpoint ----
trainer.save_checkpoint(epoch=1, val_loss=val_m["loss"], is_best=True)
ckpt_path = ckpt_tmp / "best_model.pt"
assert ckpt_path.exists(), "best_model.pt not found"
loaded_epoch = trainer.load_checkpoint(ckpt_path)
assert loaded_epoch == 1
print("save/load checkpoint       OK  epoch=%d" % loaded_epoch)

# ---- 8. evaluate ----
test_metrics = trainer.evaluate(trainer.test_loader)
assert "accuracy" in test_metrics
assert "macro_f1" in test_metrics
assert "confusion_matrix" in test_metrics
print("evaluate()                 OK  acc=%.4f" % test_metrics["accuracy"])

# ---- Cleanup ----
shutil.rmtree(str(ckpt_tmp), ignore_errors=True)

print()
print("All smoke tests PASSED.")
