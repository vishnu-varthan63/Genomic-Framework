# Comprehensive Data Source Register & Acquisition Plan
**Project:** Avian Influenza Genomic Intelligence Framework  
**Workspace:** `F:\vishnu\f\Projects\GEnomic-framework`  
**Date:** 2026-10-15  
**Audit Gate:** Gate A Deliverable  
**Plan Version:** 2.0.0-ReproducibleIngestion  

---

## 1. Overview & Strategy

This document establishes the official source register, retrieval protocols, API interfaces, authentication guidelines, and reconciliation strategies for acquiring primary sequence data and epidemiological surveillance records.

### Core Principles
1. **Zero-Guessing Provenance:** Every sequence ingested must have a stable accession, source repository, organism TaxID, collection date, geographic origin, and host taxonomy.
2. **Decoupled Storage Architecture:** Because drive `F:` has **15.4 GB** free while drive `D:` has **131.8 GB** free, all raw bulk downloads in Gate B will be directed to `D:\genomic_framework_downloads\` (or staged via symlink) to prevent workspace disk exhaustion.
3. **No Redundant Mirror Ingestion:** NCBI GenBank, ENA (EMBL-EBI), and DDBJ share records via the International Nucleotide Sequence Database Collaboration (INSDC). Mirrored accessions will be resolved to a single canonical INSDC record to prevent duplicate counting.
4. **Surveillance vs. Sequence Separation:** Surveillance platforms (WOAH WAHIS, FAO EMPRES-i, WHO FluNet, ICAR-NIHSAD, DAHD) provide outbreak corroboration and disease event metadata. Their incident case counts will **never** be conflated with nucleotide sequences.

---

## 2. Primary Sequence Repositories

### 2.1 NCBI Virus / NCBI Datasets
- **Official URLs:** 
  - NCBI Virus: [https://www.ncbi.nlm.nih.gov/labs/virus/](https://www.ncbi.nlm.nih.gov/labs/virus/)
  - NCBI Datasets: [https://www.ncbi.nlm.nih.gov/datasets/](https://www.ncbi.nlm.nih.gov/datasets/)
- **Data Available:** Complete genomes, individual segments (1–8), CDS records, protein translations, structured isolate metadata.
- **Taxonomic Criteria:**
  - Organism: *Influenza A virus* (**Taxonomy ID: 11320**)
  - Host: *Aves* (**Taxonomy ID: 8782**)
- **Target Segments:** Segments 1–8 (PB2, PB1, PA, HA, NP, NA, M, NS).
- **Target Subtypes:** Priority surveillance subtypes (H5N1, H5N2, H5N3, H5N5, H5N6, H5N8, H5N9, H7N1, H7N2, H7N3, H7N7, H7N8, H7N9, H9N2, H10N7, H10N8).
- **Date Range:** Configurable interval `2015-01-01` to `2025-12-31` (10-year surveillance span) + verified 2026 panzootic records.
- **Retrieval Mechanism:**
  - Tool: `ncbi-datasets-cli` (v16+) or REST API v2:
    ```bash
    datasets download virus genome taxon 11320 \
      --host aves \
      --geo-location "all" \
      --released-after 2015-01-01 \
      --include genome,cds,protein,annotation \
      --filename ncbi_avian_flu.zip
    ```
- **Rate Limits & Auth:** 3 requests/sec without API key; 10 requests/sec with `NCBI_API_KEY`.
- **License / Terms:** Public Domain (US Government work).

---

### 2.2 GenBank Nucleotide (NCBI Nuccore)
- **Official URL:** [https://www.ncbi.nlm.nih.gov/nuccore/](https://www.ncbi.nlm.nih.gov/nuccore/)
- **Data Available:** Full GenBank flat files (`.gb`, `.gbk`) containing detailed feature tables (cleavage site annotations, CDS spans, signal peptides, citations).
- **Retrieval Mechanism:** NCBI Entrez Direct (`esearch` / `efetch`):
  ```bash
  esearch -db nuccore -query "txid11320[Organism:exp] AND aves[Host] AND 2015/01/01:2025/12/31[PDAT]" | \
  efetch -format gbc -mode xml > genbank_avian_flu.xml
  ```
- **Rate Limits & Auth:** Same as NCBI Virus.

---

### 2.3 European Nucleotide Archive (ENA / EMBL-EBI)
- **Official URL:** [https://www.ebi.ac.uk/ena/browser/home](https://www.ebi.ac.uk/ena/browser/home)
- **Data Available:** European surveillance accessions, raw sequencing reads (FASTQ/SRA), assembled consensus sequences, European reference isolates.
- **Retrieval Mechanism:** ENA Portal REST API:
  - Endpoint: `https://www.ebi.ac.uk/ena/portal/api/search`
  - Query: `tax_tree(11320) AND host="aves" AND first_public >= 2015-01-01`
- **Rate Limits & Auth:** Unauthenticated; 10–20 requests/sec with polite backoff.
- **License / Terms:** Open access (EMBL-EBI open data terms).

---

### 2.4 DDBJ (DNA Data Bank of Japan)
- **Official URL:** [https://www.ddbj.nig.ac.jp/](https://www.ddbj.nig.ac.jp/)
- **Data Available:** East Asian and Pacific surveillance isolates, migratory flyway wild-bird sequencing records.
- **Retrieval Mechanism:** DDBJ Search REST API / REST-WSA or INSDC synchronized feeds.
- **License / Terms:** Open access.

---

### 2.5 Influenza Research Database / BV-BRC
- **Official URL:** [https://www.fludb.org/](https://www.fludb.org/) (transitioned to [https://www.bv-brc.org/](https://www.bv-brc.org/))
- **Data Available:** Pre-curated Influenza A metadata including curated HA cleavage sites, internal protein markers, IVPI assays where available, host annotations, and lineage designations.
- **Retrieval Mechanism:** BV-BRC Command Line Interface (CLI) / Data API:
  ```bash
  p3-all-genomes --eq taxon_id,11320 --eq host_group,Avian | \
  p3-get-genome-features --eq feature_type,CDS > bvbrc_features.tsv
  ```
- **Rate Limits & Auth:** Free registration; API token supported.

---

### 2.6 GISAID EpiFlu Database
- **Official URL:** [https://gisaid.org/](https://gisaid.org/)
- **Data Available:** The primary global surveillance database for emerging Avian Influenza clades (including detailed clade 2.3.4.4b H5N1 global panzootic tracking, segment alignments, patient and poultry metadata).
- **Access Policy & Compliance (CRITICAL):**
  - GISAID sequences cannot be bulk-downloaded via unauthorized scraping.
  - Ingestion must occur exclusively via authorized user credentials in compliance with the **GISAID Database Access Agreement (DAA)**.
  - Raw sequences downloaded from GISAID must **NOT** be committed to public GitHub repositories.
  - Authorship and contributing laboratories must be attributed in an accompanying acknowledgment table.
- **Protocol:** The framework provides an authorized GISAID ingest script (`scripts/ingest_gisaid.py`) that reads user-exported GISAID FASTA and metadata packages directly from a local staging directory.

---

## 3. Official Surveillance & Corroboration Sources

Surveillance databases do not distribute raw nucleotide FASTA files; they distribute epidemiological disease event reports, outbreak locations, affected poultry species, and official determinations of High Pathogenicity (HPAI) vs. Low Pathogenicity (LPAI).

| Source | Official URL | Surveillance Scope | Role in Framework |
| :--- | :--- | :--- | :--- |
| **WOAH WAHIS** | [https://wahis.woah.org/](https://wahis.woah.org/) | World Organisation for Animal Health global official disease alerts | Ground-truth outbreak classification (HPAI vs. LPAI status in affected flocks) |
| **FAO EMPRES-i** | [https://data.apps.fao.org/empres-i/](https://data.apps.fao.org/empres-i/) | Food and Agriculture Organization Global Animal Disease Information System | Epidemiological event mapping, wild-bird mortality event corroboration |
| **WHO FluNet** | [https://www.who.int/tools/flunet/](https://www.who.int/tools/flunet/) | Global Influenza Surveillance and Response System (GISRS) | Human zoonotic spillover tracking, seasonal and circulating subtype trends |
| **ICAR-NIHSAD** | [https://nihsad.nic.in/](https://nihsad.nic.in/) | National Institute of High Security Animal Diseases (Bhopal, India) | Indian subcontinent HPAI surveillance reports, national reference testing |
| **ICAR** | [https://icar.org.in/](https://icar.org.in/) | Indian Council of Agricultural Research | National agricultural surveillance bulletins and poultry health advisories |
| **DAHD India** | [https://dahd.gov.in/](https://dahd.gov.in/) | Department of Animal Husbandry and Dairying, Government of India | Official notifications of Avian Influenza outbreaks across Indian states |

---

## 4. INSDC Deduplication & Reconciliation Pipeline

```
[NCBI Virus / Datasets]     [ENA Browser]     [DDBJ Accessions]     [GISAID Authorized Export]
           \                       |                  /                         |
            \                      |                 /                          |
             v                     v                v                           v
     +----------------------------------------------------+                     |
     |         INSDC Cross-Reference Engine               |                     |
     |    (Match accession.version & identical sequence)  |                     |
     +----------------------------------------------------+                     |
                               |                                                |
                               v                                                v
            +----------------------------------------------------------------------+
            |                   Canonical Record Harmonizer                        |
            |   1. Assign internal UUID (`REC_XXXXXXXXXXXX`)                       |
            |   2. Extract & verify TaxID 11320 (Purge non-influenza)              |
            |   3. Compute SHA-256 sequence hash                                   |
            |   4. Map Segment (1-8) & Subtype (H5N1, etc.)                        |
            |   5. Extract HA Cleavage Site Motif (HPAI vs LPAI)                   |
            |   6. Corroborate with WOAH/WAHIS & FAO surveillance metadata         |
            +----------------------------------------------------------------------+
                               |
                               v
            +----------------------------------------------------------------------+
            |                   Traceable Directory Staging                        |
            |   - D:\genomic_framework_downloads\raw_sources\ (Untouched Raw)      |
            |   - dataset/validated/ (QC-Passed Viral Sequences)                   |
            |   - dataset/host_references/ (Poultry Host Genomes)                  |
            |   - dataset/excluded/ (Contaminants, Patented, Duplicates)           |
            |   - dataset/manifests/ (DATASET_MANIFEST.csv)                        |
            +----------------------------------------------------------------------+
```

---

## 5. Storage & Bandwidth Management

- **Storage Allocation:**
  - `F:\vishnu\f\Projects\GEnomic-framework`: Reserved for curated datasets, code, configurations, test suites, reports, and manifests.
  - `D:\genomic_framework_downloads`: Staging path for heavy raw archives and uncompressed INSDC sequence packages.
- **Estimated Ingestion Volume:**
  - Permitted public NCBI/ENA/DDBJ Avian Influenza complete sequences (2015–2025): ~65,000 to ~85,000 segment records.
  - Uncompressed FASTA size: ~150 MB to ~250 MB (readily accommodated without impacting disk limits).
  - Full GenBank feature XML/tables: ~1.5 GB.
- **Rate Limit Compliance:** All download scripts implement exponential backoff ($2^n$ seconds) with maximum retry limits ($n=5$) and honor NCBI `api_key` environment variables.
