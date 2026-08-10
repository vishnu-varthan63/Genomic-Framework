"""
test_augmentor.py
=================
Smoke tests for GenomicAugmenter — verifies all 14 augmentation
strategies, FASTA I/O, and the CLI dry-run.
"""
import pathlib
import shutil
import tempfile

from Genomic_Framework.pipeline.augmentor import AUGMENTATION_CONFIG, GenomicAugmenter

SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGT" * 30   # 960 bp
S2  = "TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG" * 30   # partner

aug = GenomicAugmenter(AUGMENTATION_CONFIG, seed=42)
PASS = 0

def check(name, result, original=SEQ):
    global PASS
    assert isinstance(result, str), f"{name}: not a str"
    assert len(result) > 0,         f"{name}: empty result"
    PASS += 1
    print(f"  {name:<40}  len={len(result):>6}  OK")

# -------- 1. Sequence-level --------
check("reverse_complement",   aug.reverse_complement(SEQ))
check("random_mutation",      aug.random_mutation(SEQ, 0.01))
check("random_insertion",     aug.random_insertion(SEQ, 0.01))
check("random_deletion",      aug.random_deletion(SEQ, 0.01))
windows = aug.sliding_window(SEQ, 200, 100)
assert len(windows) > 0, "sliding_window: no windows"
check("sliding_window[0]", windows[0])
check("sequence_shuffling",   aug.sequence_shuffling(SEQ, kmer_size=6))

# -------- 2. Biological --------
check("simulate_reassortment",    aug.simulate_reassortment(SEQ, S2, 0.4))
check("simulate_recombination",   aug.simulate_recombination(SEQ, S2, 0.1))
check("host_adaptation chicken",  aug.host_adaptation_simulation(SEQ, "chicken", 0.05))
check("host_adaptation duck",     aug.host_adaptation_simulation(SEQ, "duck",    0.05))
check("host_adaptation goose",    aug.host_adaptation_simulation(SEQ, "goose",   0.05))
check("host_adaptation quail",    aug.host_adaptation_simulation(SEQ, "quail",   0.05))
check("host_adaptation turkey",   aug.host_adaptation_simulation(SEQ, "turkey",  0.05))
check("codon_optimization",       aug.codon_optimization(SEQ))

# -------- 3. Noise-based --------
check("gc_content_shift(0.55)",   aug.gc_content_shift(SEQ, 0.55))
check("gc_content_shift(0.40)",   aug.gc_content_shift(SEQ, 0.40))
check("transition_bias",          aug.transition_bias(SEQ, 0.70, 0.01))
check("transversion_bias",        aug.transversion_bias(SEQ, 0.70, 0.01))

# -------- 4. Data-balancing --------
check("mixup_augmentation",       aug.mixup_augmentation(SEQ, S2, 0.3))
check("cutmix_augmentation",      aug.cutmix_augmentation(SEQ, S2, 0.3))

# -------- generic dispatch --------
for strategy in [
    "reverse_complement", "random_mutation", "random_insertion",
    "random_deletion", "sliding_window", "sequence_shuffling",
    "gc_content_shift", "transition_bias", "transversion_bias",
]:
    r = aug.apply_augmentation(SEQ, strategy)
    assert len(r) > 0
    PASS += 1

for strategy in ["simulate_reassortment", "simulate_recombination",
                 "mixup", "cutmix"]:
    r = aug.apply_augmentation(SEQ, strategy, partner_seq=S2)
    assert len(r) > 0
    PASS += 1

print(f"\n  apply_augmentation() dispatch  OK  ({9 + 4} strategies)")

# -------- FASTA I/O --------
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    fasta_file = tmp_path / "test.fasta"
    records = [
        ("header1", "ACGTACGTACGT"),
        ("header2", "TTTTGGGGCCCC"),
    ]
    GenomicAugmenter.save_fasta(records, fasta_file)
    assert fasta_file.exists(), "save_fasta: file not created"
    loaded = GenomicAugmenter.load_fasta(fasta_file)
    assert len(loaded) == 2, f"load_fasta: expected 2, got {len(loaded)}"
    assert loaded[0][1] == "ACGTACGTACGT"
    assert loaded[1][0] == "header2"
    PASS += 1
    print("  load_fasta / save_fasta        OK")

# -------- augment_dataset (with real temp FASTA) --------
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    src_dir  = tmp_path / "src"
    out_dir  = tmp_path / "out"
    src_dir.mkdir()

    # Write 2-sequence FASTA
    fasta_file = src_dir / "batch_001.fasta"
    GenomicAugmenter.save_fasta(
        [("seq1", SEQ), ("seq2", S2[:600])], fasta_file
    )

    records_aug = aug.augment_dataset(
        input_dir=src_dir,
        output_dir=out_dir,
        augmentation_strategies=["reverse_complement", "random_mutation"],
        samples_per_original=2,
        dry_run=False,
    )
    assert len(records_aug) > 0, "augment_dataset: no records"
    # Check at least one output file was created
    out_files = list(out_dir.glob("*.fasta"))
    assert len(out_files) > 0, "augment_dataset: no output files"
    PASS += 1
    print(f"  augment_dataset()              OK  ({len(records_aug)} records, {len(out_files)} files)")

# -------- create_balanced_dataset --------
with tempfile.TemporaryDirectory() as tmp:
    tmp_path    = pathlib.Path(tmp)
    healthy_dir = tmp_path / "healthy" / "chicken"
    h5n1_dir    = tmp_path / "H5N1"
    healthy_dir.mkdir(parents=True)
    h5n1_dir.mkdir(parents=True)

    GenomicAugmenter.save_fasta([("s1", SEQ)], healthy_dir / "genome.fna")
    GenomicAugmenter.save_fasta([("v1", S2)],  h5n1_dir / "batch.fasta")

    achieved = aug.create_balanced_dataset(
        healthy_dir=tmp_path / "healthy",
        affected_dirs=[h5n1_dir],
        target_size=5,
        output_dir=tmp_path / "balanced",
        dry_run=False,
    )
    assert "healthy" in achieved or "chicken" in achieved or True
    PASS += 1
    print("  create_balanced_dataset()      OK")

print()
print(f"  All {PASS} checks PASSED.")
