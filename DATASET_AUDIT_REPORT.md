# Forensic Dataset Audit Report
**Project:** Avian Influenza Genomic Intelligence Framework  
**Workspace:** `F:\vishnu\f\Projects\GEnomic-framework`  
**Audit Date:** 2026-10-15  
**Audit Gate:** Gate A (Forensic Inventory & Discrepancy Reconciliation)  
**Status:** COMPLETE — EMPIRICALLY MEASURED  

---

## Executive Summary

A comprehensive, record-level forensic audit was conducted across all existing genomic datasets in the project workspace, backup repositories, and historical logs. Every file, FASTA header, nucleotide string, accession number, and sequence hash was inspected and measured directly using automated Python forensic scripts (`scripts/audit_existing_datasets.py`, `scripts/audit_raw_viral.py`, `scripts/audit_balanced_classes.py`, `scripts/audit_leakage_and_cleavage.py`).

### Critical Audit Findings
1. **Host Genome Conflation Resolved:** The historical claim of `1,048,666 sequences` and `4,782,973,902 base pairs` in `dataset/` arose from recursively concatenating eukaryotic avian host genomes (Turkey, Duck, Quail, Goose assemblies totaling 5.12 GB and ~1.03M scaffolds/CDS) with the viral files. The actual viral surveillance dataset (`dataset/raw/`) contains **134 FASTA files, 13,389 sequences, and 42,012,236 bp**.
2. **Severe Taxonomic Contamination in Raw Viral Sequences:** Within `dataset/raw/`, **242 records (comprising 20,487,787 bp, or 48.8% of the total base pairs)** are non-influenza contaminants resulting from unconstrained historical keyword searches. These contaminants include whole-genome shotgun scaffolds of *Batillaria attramentaria* (Asian mud snail, up to 5.2M bp contigs in `H7N7`), *Acinetobacter baumannii* (bacterium in `H7N9`), *Rattus norvegicus* / *Mus musculus* karyopherin mRNA, synthetic patent vaccine constructs, and bacteriophages.
3. **Flawed "Benign" Class Definition:** In `dataset_augmented/balanced/Benign/`, **94.8% (1,129 out of 1,191 sequences)** are eukaryotic Mallard duck (*Anas platyrhynchos*) mRNA/CDS (e.g., guanylyl cyclase-activating protein 1-like), not non-pathogenic viral genomes. The model was trained to differentiate host bird DNA from viral RNA rather than learning viral pathogenicity.
4. **Subtype-to-Class Pathogenicity Conflation:** Pathogenicity was hardcoded by viral subtype rather than biological determinants. Empirical translation of the 1,991 Hemagglutinin (Segment 4) sequences showed that **68.8% of H7N2 sequences harbor HPAI polybasic cleavage sites** (despite being hardcoded as Class 1 "Likely Benign"), while **79.2% of H5N9 sequences** and **73.4% of H7N9 sequences** harbor monobasic LPAI cleavage sites (despite being hardcoded as Class 3 "Pathogenic").
5. **Massive Cross-Split Data Leakage:** In the historical 4-class balanced benchmark (35,211 sequences across 72 batch files), augmentation was performed *prior* to splitting. Simulating the file-level split revealed **1,028 parent accessions overlapping between Train and Test (88.6% of all Test set accessions)** and **237 exact identical sequences shared between Train and Test**. This explains the artificially inflated test accuracy (>98%) in historical logs.

---

## 1. Physical Dataset Inventory & Directory Structure

| Directory Path | Role / Content | Physical Files | Disk Size | Status / Findings |
| :--- | :--- | :---: | :---: | :--- |
| `dataset/raw/` | Historical raw viral surveillance | 134 FASTA | 43.01 MB | 14 subtypes; contains 13,147 viral & 242 contaminant records |
| `dataset/healthy/` | Avian host reference assemblies | 27 files | 5,122.01 MB | 5 host species (chicken, duck, goose, quail, turkey); 5.12 GB |
| `dataset_augmented/balanced/` | 4-class balanced training pool | 72 FASTA | 51.21 MB | 35,211 sequences across 4 classes; 12.6x augmented |
| `dataset_augmented/raw/` | Full augmented viral pool | 405 FASTA | 45.42 MB | Tiered augmentations of raw viral sequences |
| `dataset_augmented/healthy/` | Augmented host sequences | 63 FASTA | 1.53 MB | Augmented duck host CDS sequences |
| `D:\dataset\` | External untouched backup pool | 161 files | 5,165.01 MB | Bit-for-bit identical to `F:\...\dataset\` |

---

## 2. Forensic Reconciliation of Historical Claims vs. Empirical Ground Truth

| Metric | Historical Claim (`dataset_statistics.json`) | Measured Reality (This Audit) | Forensic Explanation / Root Cause |
| :--- | :---: | :---: | :--- |
| **Raw Repository Pool Sequences** | 1,048,666 | **13,389 (Viral)** + ~1,035,000 (Host) | Historical generator concatenated turkey genome (185,175 scaffolds) and duck/quail assemblies with viral files. |
| **Raw Repository Pool Base Pairs** | 4,782,973,902 bp (~4.78 Gb) | **42,012,236 bp (Viral)** + 4.74 Gb (Host) | 4.74 Gb of the total base pairs belonged to eukaryotic poultry host DNA, not influenza virus. |
| **Raw Viral FASTA Files** | 149 files claimed | **134 viral + 15 host FASTA files** | Exactly 149 FASTA files exist when adding the 15 host FASTA files in `dataset/healthy/`. |
| **Augmented Pool Sequences** | 67,482 sequences | **67,482 sequences** | Confirmed present in `dataset_augmented/raw/` and `dataset_augmented/healthy/`. |
| **Balanced Benchmark Sequences** | 35,211 sequences | **35,211 sequences** | Confirmed across 72 files in `dataset_augmented/balanced/`. |
| **Unique Biological Records in Balanced** | Presumed 35,211 unique | **2,796 unique accessions** | 35,211 sequences derive from only 2,796 distinct biological accessions (12.6x expansion). |
| **Train / Val / Test Split** | 23,520 / 5,000 / 6,691 | **23,520 / 5,000 / 6,691** | File-level batch allocation: 48 files (Train), 10 files (Val), 14 files (Test). |
| **Cross-Split Test Set Leakage** | Claimed independent (0%) | **88.6% Parent Leakage** | 1,028 of the 1,160 accessions in Test also occur in Train; 237 exact sequence duplicates. |

---

## 3. Detailed Audit of Raw Surveillance Dataset (`dataset/raw/`)

### Subtype Breakdown & Completeness
The 134 FASTA files in `dataset/raw/` represent 14 subtype directories:

| Subtype Folder | Files | Total Seqs | Total Bases (bp) | Mean Len (bp) | Complete Seqs | Partial Seqs | Contaminants Detected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **H10N7** | 10 | 1,000 | 1,815,259 | 1,815.3 | 995 | 5 | 1 snail contig (*Batillaria*) |
| **H10N8** | 10 | 1,000 | 1,628,433 | 1,628.4 | 971 | 29 | None |
| **H5N1** | 10 | 1,000 | 1,634,432 | 1,634.4 | 986 | 14 | None |
| **H5N2** | 10 | 1,000 | 1,644,409 | 1,644.4 | 997 | 3 | None |
| **H5N3** | 10 | 1,000 | 1,620,877 | 1,620.9 | 957 | 43 | 8 patent vaccine constructs |
| **H5N6** | 10 | 1,000 | 1,636,096 | 1,636.1 | 992 | 8 | 1 synthetic consensus construct |
| **H5N8** | 10 | 1,000 | 1,562,222 | 1,562.2 | 902 | 98 | 1 synthetic consensus construct |
| **H5N9** | 6 | 591 | 964,076 | 1,631.3 | 565 | 26 | 8 patent recombinant constructs |
| **H7N1** | 10 | 1,000 | 1,611,358 | 1,611.4 | 917 | 83 | None |
| **H7N2** | 10 | 1,000 | 1,749,377 | 1,749.4 | 962 | 38 | None |
| **H7N3** | 10 | 1,000 | 1,652,270 | 1,652.3 | 988 | 12 | None |
| **H7N7** | 8 | 800 | 14,105,281 | 17,631.6 | 780 | 20 | **4 snail contigs (12.8M bp total)** |
| **H7N9** | 10 | 998 | 8,787,234 | 8,804.8 | 883 | 115 | **Bacterial, rodent mRNA & phage** |
| **H9N2** | 10 | 1,000 | 1,600,912 | 1,600.9 | 979 | 21 | None |
| **TOTAL** | **134** | **13,389** | **42,012,236** | **3,137.8** | **12,494** | **895** | **242 non-influenza records** |

### Genome Segment Distribution
Viral Influenza A has an 8-segment negative-sense single-stranded RNA genome:

| Segment | Protein Product | Expected Native Length | Count in Raw | Complete (>=90%) | Partial (<90%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Polymerase PB2 | ~2,341 bp | 1,613 | 1,597 (99.0%) | 16 (1.0%) |
| **2** | Polymerase PB1 | ~2,341 bp | 1,614 | 1,598 (99.0%) | 16 (1.0%) |
| **3** | Polymerase PA | ~2,233 bp | 1,580 | 1,513 (95.8%) | 67 (4.2%) |
| **4** | Hemagglutinin (HA) | ~1,778 bp | 1,965 | 1,809 (92.1%) | 156 (7.9%) |
| **5** | Nucleoprotein (NP) | ~1,565 bp | 1,551 | 1,528 (98.5%) | 23 (1.5%) |
| **6** | Neuraminidase (NA) | ~1,413 bp | 1,664 | 1,609 (96.7%) | 55 (3.3%) |
| **7** | Matrix (M1/M2) | ~1,027 bp | 1,611 | 1,574 (97.7%) | 37 (2.3%) |
| **8** | Non-structural (NS1/NEP) | ~890 bp | 1,553 | 1,514 (97.5%) | 39 (2.5%) |
| **Unassigned** | Contaminants / Chimeras | Variable | 238 | 132 | 106 |

---

## 4. Detailed Audit of 4-Class Balanced Benchmark (`dataset_augmented/balanced/`)

| Metric | Class 0: Benign | Class 1: Likely Benign | Class 2: Likely Pathogenic | Class 3: Pathogenic | Entire Benchmark |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Physical Files** | 3 files | 37 files | 22 files | 10 files | **72 files** |
| **Total Sequences** | 1,191 | 18,185 | 10,855 | 4,980 | **35,211** |
| **Percentage of Dataset** | 3.38% | 51.65% | 30.83% | 14.14% | **100.0%** |
| **Unique Sequences** | 1,078 | 17,708 | 10,256 | 4,678 | **33,719** |
| **Exact Duplicate Sequences** | 113 | 477 | 599 | 302 | **1,492** |
| **Unique Source Accessions** | 81 | 1,024 | 1,171 | 520 | **2,796** |
| **Total Base Pairs** | 1,385,010 bp | 24,653,531 bp | 13,546,658 bp | 6,324,231 bp | **45,909,430 bp** |
| **Mean Sequence Length** | 1,162.9 bp | 1,355.7 bp | 1,248.0 bp | 1,269.9 bp | **1,303.8 bp** |
| **Median Sequence Length** | 1,500 bp | 1,500 bp | 1,500 bp | 1,500 bp | **1,500 bp** |
| **Min / Max Length** | 189 / 1,500 bp | 145 / 1,500 bp | 175 / 1,500 bp | 51 / 1,500 bp | **51 / 1,500 bp** |
| **Eukaryotic Host Sequences** | **1,129 (94.8%)** | 180 (1.0%) | 86 (0.8%) | 0 (0.0%) | **1,395 (4.0%)** |
| **Viral Sequences** | 62 (5.2%) | 18,005 (99.0%) | 10,769 (99.2%) | 4,980 (100%) | **33,816 (96.0%)** |

---

## 5. Molecular Cleavage Site Motifs vs. Hardcoded Subtype Labels

In biological reality (WOAH Terrestrial Manual), Avian Influenza pathogenicity is determined by the **cleavage site of Hemagglutinin (HA)**:
- **LPAI (Low Pathogenic):** Single basic amino acid before the fusion peptide (monobasic: e.g., $P-Q-R-E-T-R*G-L-F$).
- **HPAI (High Pathogenic):** Multiple basic amino acids before the fusion peptide (polybasic motif: e.g., $R-X-K/R-R*G-L-F$ or $\ge 3$ basic residues).

Auditing all 1,991 Hemagglutinin sequences in `dataset/raw/` by 6-frame translation and cleavage motif matching revealed:

```
Total HA Sequences Examined: 1,991
Cleavage Junctions Detected: 1,908 (95.8%)
Overall Biological Calls:   LPAI = 1,130 (59.2%),  HPAI = 778 (40.8%)
```

### Discordance Between Historical Hardcoded Labels and Biological Cleavage Motifs

| Subtype Folder | Historical Hardcoded Class | HPAI Motif (Polybasic) | LPAI Motif (Monobasic) | Discordance Analysis |
| :--- | :--- | :---: | :---: | :--- |
| **H5N1** | Class 3 (Pathogenic) | 125 | 0 | 100% concordant with HPAI |
| **H5N2** | Class 1 (Likely Benign) | **42** | 74 | **36.2% of H5N2 sequences are actually HPAI!** |
| **H5N3** | Class 1 (Likely Benign) | **12** | 123 | **8.9% are HPAI** |
| **H5N6** | Class 2 (Likely Pathogenic) | 129 | 6 | 95.6% HPAI clade 2.3.4.4 |
| **H5N8** | Class 2 (Likely Pathogenic) | 186 | 10 | 94.9% HPAI clade 2.3.4.4b |
| **H5N9** | Class 3 (Pathogenic) | 16 | **61** | **79.2% are LPAI (Mislabeled as Pathogenic!)** |
| **H7N1** | Class 2 (Likely Pathogenic) | 16 | **142** | **89.9% are LPAI (Mislabeled as Pathogenic!)** |
| **H7N2** | Class 1 (Likely Benign) | **154** | 70 | **68.8% are HPAI (Mislabeled as Benign!)** |
| **H7N3** | Class 1 (Likely Benign) | **39** | 83 | **32.0% are HPAI (Mislabeled as Benign!)** |
| **H7N7** | Class 2 (Likely Pathogenic) | 6 | **107** | **94.7% are LPAI (Mislabeled as Pathogenic!)** |
| **H7N9** | Class 3 (Pathogenic) | 29 | **80** | **73.4% are LPAI (Wave 1-4 poultry isolates)** |
| **H10N7** | Class 2 (Likely Pathogenic) | 0 | 129 | 100% LPAI in poultry (Mislabeled as Pathogenic!) |
| **H10N8** | Class 2 (Likely Pathogenic) | 6 | 115 | 95.0% LPAI in poultry |
| **H9N2** | Class 1 (Likely Benign) | 18 | 130 | 87.8% LPAI in poultry |

> [!CAUTION]
> Hardcoding pathogenicity by subtype alone is scientifically invalid and defenseless in peer review. An H7N2 isolate with a polybasic cleavage site is high-pathogenic, while an H7N9 isolate without the 2017 insertion is low-pathogenic in poultry.

---

## 6. Historical Augmentation & Split Leakage Audit

### Audit of `dataset_augmented/` Transformations
Inspection of `data_augmentation_pipeline.py` identified the following methods:
1. **Tier 1 (Sliding Window & Truncation):** Hard-clipped all sequences longer than 1,500 bp to 1,500 bp. Truncated PB2/PB1/PA/HA sequences and discarded 3' coding and non-coding termini.
2. **Tier 2 (Fragment Shuffling & Reverse Complement):** Shuffled internal fragments, destroying codon reading frames, ORF integrity, and cleavage motifs. Reverse complement of negative-sense vRNA produced non-sense positive strands without annotating strand polarity.
3. **Tier 3 (Codon Bias Substitution):** Replaced codons with host preferences without regard to viral secondary RNA structures (e.g., packaging signals at segment ends).
4. **Tier 4 (GC Content Shift):** Induced artificial GC bias up to $\pm 10\%$.
5. **Tier 5 (SequenceSMOTE):** Chimerized sequences by coin-flip selection between pairs of sequences character by character, producing biologically invalid synthetic polymers.

### Split Leakage Audit
Simulating the file-level split implemented in `preprocessor.py` (seed=42):
- **Train Split:** 48 files, 23,520 sequences, 2,441 unique parent accessions.
- **Validation Split:** 10 files, 5,000 sequences, 1,069 unique parent accessions.
- **Test Split:** 14 files, 6,691 sequences, 1,160 unique parent accessions.

```
Accession-Level Leakage:
  Train & Val Overlap:   779 accessions (72.9% of Val accessions present in Train)
  Train & Test Overlap:  1,028 accessions (88.6% of Test accessions present in Train)
  Val & Test Overlap:    427 accessions

Exact Sequence Identity Leakage:
  Train & Val Identical Sequences:  207 sequences
  Train & Test Identical Sequences: 237 sequences
```

> [!WARNING]
> Because augmentation was conducted before splitting, 88.6% of the test set consists of direct augmented derivatives or identical copies of sequences in the training set. This leakage invalidates all historical test set performance claims.

---

## 7. Audit Conclusion & Gate A Recommendation

1. **Rejection of Historical Balanced Set for Research Benchmark:** The historical `dataset_augmented/balanced` directory cannot be defended for scientific publication due to 88.6% test set leakage and host genome contamination in the Benign class.
2. **Preservation of Raw Files:** All 134 raw files and host references are preserved untouched.
3. **Rebuild Required:** A clean, versioned, isolate-aware dataset must be constructed following the policies defined in `DATASET_LABELING_POLICY.md` and acquired via `DATA_SOURCE_REGISTER.md`.
