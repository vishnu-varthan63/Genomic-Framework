# Dataset Quality Control & Validation Specification
**Project:** Avian Influenza Genomic Intelligence Framework  
**Workspace:** `F:\vishnu\f\Projects\GEnomic-framework`  
**Date:** 2026-10-15  
**Audit Gate:** Gate A Deliverable  
**Specification Version:** 2.0.0-BiologicalStandard  

---

## 1. Scope & Objective

This document defines the strict, scientifically validated Quality Control (QC) protocols and acceptance thresholds for the Avian Influenza Genomic Intelligence Framework.

Past iterations suffered from two severe QC defects:
1. **Arbitrary Truncation:** All sequences were blindly clipped to 1,500 bp, truncating segments 1, 2, 3, and 4 and discarding biological termini.
2. **Taxonomic Infiltration:** Keyword-based downloads allowed bacterial genomes (*Acinetobacter*), marine molluscs (*Batillaria attramentaria*), and mammalian mRNA (*Rattus norvegicus*) to pass into the viral raw directories because no organism verification was enforced.

This specification establishes biologically grounded, segment-aware quality filters.

---

## 2. Biological Validation Filters

### 2.1 Taxonomic Identity Verification
Every candidate sequence must have an unambiguous taxonomic lineage:
- **Kingdom:** Orthornavirae
- **Phylum:** Negarnaviricota
- **Class:** Insthoviricetes
- **Order:** Articulavirales
- **Family:** Orthomyxoviridae
- **Genus:** Alphainfluenzavirus
- **Species:** Influenza A virus (**NCBI Taxonomy ID: 11320**)

> [!CRITICAL]
> Any sequence whose NCBI TaxID is not `11320` (or subordinate sublineage), or whose header contains non-influenza organisms (e.g., host genome contigs, bacterial scaffolds, phage constructs, mammalian transcripts), is **IMMEDIATELY EXCLUDED**.

### 2.2 Recombinant, Patent & Synthetic Exclusions
All records containing patent document identifiers (e.g., `KR 102019...`), vector backbones, cloning linkers, or designated `Synthetic construct` designations are quarantined into `dataset/excluded/synthetic_and_patents/`. Only naturally occurring or officially verified surveillance isolates are admitted to the primary research benchmark.

---

## 3. Segment-Aware Completeness Thresholds

Influenza A virus possesses 8 single-stranded negative-sense RNA segments. Each segment has a biologically constrained length range. Applying a single arbitrary length cutoff (such as 1,500 bp) is biologically invalid.

| Segment | Encoded Protein(s) | Typical Native Length (bp) | Minimum Complete Threshold | Maximum Threshold | Minimum Partial Threshold (Flagged) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | PB2 (Polymerase basic protein 2) | 2,341 bp | 2,150 bp | 2,450 bp | 1,000 bp |
| **2** | PB1, PB1-F2, PB1-N40 | 2,341 bp | 2,150 bp | 2,450 bp | 1,000 bp |
| **3** | PA, PA-X, PA-N155, PA-N182 | 2,233 bp | 2,050 bp | 2,350 bp | 1,000 bp |
| **4** | Hemagglutinin (HA) | 1,778 bp | 1,650 bp | 1,880 bp | 800 bp |
| **5** | Nucleoprotein (NP) | 1,565 bp | 1,450 bp | 1,650 bp | 750 bp |
| **6** | Neuraminidase (NA) | 1,413 bp | 1,320 bp | 1,500 bp | 700 bp |
| **7** | Matrix (M1, M2) | 1,027 bp | 950 bp | 1,120 bp | 500 bp |
| **8** | Non-structural (NS1, NEP/NS2) | 890 bp | 820 bp | 980 bp | 450 bp |

*Criteria:*
- **Complete:** Sequence length falls within $[\text{Minimum Complete Threshold}, \text{Maximum Threshold}]$ and covers $\ge 90\%$ of the coding sequence.
- **Partial (Usable):** Sequence length is below complete threshold but $\ge \text{Minimum Partial Threshold}$. Allowed only for segment-specific sub-analyses; marked with `completeness_flag = "partial"`.
- **Rejected (< Partial):** Any fragment below the minimum partial threshold is excluded as an uninformative short read.

---

## 4. Nucleotide Alphabet & Sequence Integrity Checks

### 4.1 Allowed Alphabet
- **Standard Canonical Bases:** `A`, `C`, `G`, `T` (or `U` for direct RNA records, normalized to `T`).
- **Allowed Ambiguity:** `N` (unknown base).
- **IUPAC Degenerate Codes:** `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V` are tolerated only if representing verified sequencing polymorphism, and must not exceed 0.2% of the sequence length.

### 4.2 Ambiguity Threshold ($N$-Fraction)
$$\text{Fraction}_N = \frac{\text{Count}(N)}{\text{Length}(\text{Seq})}$$
- **Threshold:** $\text{Fraction}_N \le 0.01$ (Maximum 1.0% ambiguous bases).
- Sequences with $\text{Fraction}_N > 0.01$ are rejected.
- Sequences with continuous runs of $N \ge 15$ are rejected (indicates assembly gaps).

### 4.3 Homopolymer Runs
$$\text{MaxHomopolymer}(\text{Seq}) \le 10$$
Runs of 11 or more consecutive identical nucleotides indicate sequencing artifact (e.g., homopolymer error in pyrosequencing or nanopore sequencing without proper polishing).

### 4.4 GC Content Distribution
Avian influenza viral genome segments exhibit a characteristic GC content range:
- Native influenza A GC content: **41.0% to 46.5%** across segments.
- Permitted QC window: **36.0% to 52.0%**.
- Sequences outside this range are flagged for manual review or excluded as foreign sequence. (E.g., *Batillaria attramentaria* contaminants exhibited GC content > 58%).

---

## 5. QC Pass/Fail Audit of Existing Raw Surveillance Pool

Applying these biological QC filters to the existing 13,389 records in `dataset/raw/`:

```
Total Sequences Audited:             13,389
  1. Passed All Viral QC Filters:    13,147 (98.19%)
  2. Failed - Taxonomic Infiltration:   242 (1.81%)
       - Batillaria attramentaria:         5 records (12,839,341 bp)
       - Acinetobacter baumannii:         67 records (6,892,104 bp)
       - Rodent Kpna6 mRNA:                5 records (28,309 bp)
       - Bacteriophage contigs:            1 record (182,550 bp)
       - Synthetic/Patent Constructs:     164 records (545,483 bp)
  3. Total Base Pairs Retained:      21,524,449 bp (from 42,012,236 bp)
  4. Contaminant Base Pairs Purged:  20,487,787 bp (48.8% of raw bases!)
```

### Segment Representation of the 13,147 Valid Viral Records

| Segment | Protein | Valid Records | Complete (>=90%) | Partial |
| :---: | :---: | :---: | :---: | :---: |
| **1** | PB2 | 1,613 | 1,597 | 16 |
| **2** | PB1 | 1,614 | 1,598 | 16 |
| **3** | PA | 1,580 | 1,513 | 67 |
| **4** | HA | 1,965 | 1,809 | 156 |
| **5** | NP | 1,551 | 1,528 | 23 |
| **6** | NA | 1,664 | 1,609 | 55 |
| **7** | MP | 1,611 | 1,574 | 37 |
| **8** | NS | 1,553 | 1,514 | 39 |
| **Total** | — | **13,147** | **12,742 (96.9%)** | **405 (3.1%)** |

---

## 6. Host Genome Quarantine Protocol

The 5 poultry host genome assemblies located in `dataset/healthy/` are quarantined from the viral training sets:
- `chicken` (*Gallus gallus*, GCF_016699485.2)
- `duck` (*Anas platyrhynchos*)
- `turkey` (*Meleagris gallopavo*, GCF_000146605.3)
- `goose` (*Anser anser*)
- `quail` (*Coturnix japonica*)

**Protocol:**
1. These assemblies are relocated to a dedicated host repository `dataset/host_references/`.
2. They are indexed solely for:
   - Host factor interaction studies (e.g., host receptor binding mapping).
   - Decontaminating host reads from raw viral sequencing runs.
3. They must **NEVER** be ingested into any viral pathogenicity training, validation, or test split.
