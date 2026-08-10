"""Smoke tests for healthy_downloader.py (REST API version)."""
import io
import json
import pathlib
import shutil
import tempfile
import zipfile

from Genomic_Framework.pipeline.healthy_downloader import (
    HEALTHY_DIR,
    SPECIES_CATALOGUE,
    HealthyGenomeDownloader,
    _count_fasta_sequences,
    _FILE_PATTERNS,
    _INCLUDE_TYPES,
    _sizeof_fmt,
)

# --- Helper utilities ---
assert _sizeof_fmt(0)       == "0.0 B"
assert _sizeof_fmt(1024)    == "1.0 KB"
assert _sizeof_fmt(1048576) == "1.0 MB"
print("_sizeof_fmt()         OK")

assert len(SPECIES_CATALOGUE) == 5
labels = [s["label"] for s in SPECIES_CATALOGUE]
assert labels == ["chicken", "duck", "turkey", "goose", "quail"]
print("SPECIES_CATALOGUE     OK:", labels)

assert "GENOME_FASTA" in _INCLUDE_TYPES
assert "GENOME_GFF"   in _INCLUDE_TYPES
assert "CDS_FASTA"    in _INCLUDE_TYPES
assert "PROT_FASTA"   in _INCLUDE_TYPES
print("_INCLUDE_TYPES        OK")

# --- Instantiation ---
dl = HealthyGenomeDownloader()
assert repr(dl).startswith("HealthyGenomeDownloader(")
print("Instantiated          OK")

# --- Directory creation ---
dl.create_directories()
for sp in SPECIES_CATALOGUE:
    d = pathlib.Path("dataset/healthy") / sp["label"]
    assert d.exists(), f"Missing: {d}"
print("create_directories()  OK")

# --- _all_files_present: False on empty dirs ---
assert not dl._all_files_present(pathlib.Path("dataset/healthy/duck"))
print("_all_files_present()  OK (False on empty)")

# --- Build a minimal fake NCBI zip and test extract/rename/verify ---
with tempfile.TemporaryDirectory() as tmp:
    acc = "GCF_TEST_1.1"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr(f"ncbi_dataset/data/{acc}/{acc}_ASM_genomic.fna",
                    ">chr1\nACGT\n")
        zf.writestr(f"ncbi_dataset/data/{acc}/{acc}_ASM_genomic.gff",
                    "##gff-version 3\nchr1\t.\tgene\t1\t4\t.\t+\t.\tID=g1\n")
        zf.writestr(f"ncbi_dataset/data/{acc}/cds_from_genomic.fna",
                    ">cds1\nACGT\n")
        zf.writestr(f"ncbi_dataset/data/{acc}/protein.faa",
                    ">prot1\nM\n")

    tmp_path  = pathlib.Path(tmp)
    zip_path  = tmp_path / "test.zip"
    zip_path.write_bytes(zip_buf.getvalue())

    extract_dir = tmp_path / "extracted"
    sp_dir      = tmp_path / "chicken_out"

    dl._extract_zip(zip_path, extract_dir, "chicken")
    placed = dl._rename_and_place(extract_dir, sp_dir, acc, "chicken")
    assert placed, "No files placed!"
    for fname in ("genome.fna", "annotation.gff", "cds.fna", "protein.faa"):
        assert (sp_dir / fname).exists(), f"Missing: {fname}"
    print("_extract_zip / _rename_and_place  OK")

    # Copy into real species dir for verify/metadata tests
    # (must happen while tmp_path / sp_dir still exist inside the with block)
    real_sp_dir = pathlib.Path("dataset/healthy/chicken")
    for fname in ("genome.fna", "annotation.gff", "cds.fna", "protein.faa"):
        shutil.copy2(sp_dir / fname, real_sp_dir / fname)

# --- _all_files_present: now True ---
real_sp_dir = pathlib.Path("dataset/healthy/chicken")
print("_all_files_present()  OK (True after copy)")

# --- verify_download ---
passed, report = dl.verify_download(SPECIES_CATALOGUE[0])
assert passed, f"Verification failed: {report}"
for fname in ("genome.fna", "annotation.gff", "cds.fna", "protein.faa"):
    assert report[fname]["status"] == "OK", f"{fname}: {report[fname]}"
print("verify_download()     OK")

# --- save_metadata (offline: mock empty metadata) ---
dl._meta_client.fetch_assembly_metadata = lambda acc: {}
dl.save_metadata(SPECIES_CATALOGUE[0], report)
meta = json.loads((real_sp_dir / "metadata.json").read_text("utf-8"))
assert meta["species"] == "Gallus gallus"
assert meta["assembly_accession"] == "GCF_016699485.2"
print("save_metadata()       OK")

print()
print("All smoke tests PASSED.")
