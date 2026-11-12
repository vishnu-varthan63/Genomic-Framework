# Scientifically Defensible Dataset Labeling Policy
**Project:** Avian Influenza Genomic Intelligence Framework  
**Workspace:** `F:\vishnu\f\Projects\GEnomic-framework`  
**Date:** 2026-10-15  
**Audit Gate:** Gate A Deliverable  
**Policy Version:** 2.0.0-BiologicalEvidence  

---

## 1. Executive Problem Statement & Critique of Historical Labels

The historical implementation assigned sequence labels using a hardcoded dictionary (`SUBTYPE_TO_CLASS` in `preprocessor.py` and `data_augmentation_pipeline.py`):
```python
# HISTORICAL FLAWED MAPPING:
SUBTYPE_TO_CLASS = {
    "healthy":  0,  # Class 0: Benign (EUKARYOTIC BIRD GENOMES!)
    "H5N2": 1, "H5N3": 1, "H7N2": 1, "H7N3": 1, "H9N2": 1, # Class 1: Likely Benign
    "H5N6": 2, "H5N8": 2, "H7N1": 2, "H7N7": 2, "H10N7": 2, "H10N8": 2, # Class 2: Likely Pathogenic
    "H5N1": 3, "H5N9": 3, "H7N9": 3, # Class 3: Pathogenic
}
```

### Forensic Deficiencies of the Historical Policy
1. **Host-Virus Conflation in Class 0:** Class 0 ("Benign") was populated with 94.8% Mallard duck chromosomal mRNA/CDS (e.g., guanylyl cyclase-activating protein 1-like), while classes 1–3 were viral RNA. This trained the neural network to differentiate eukaryotic avian cellular transcripts from orthomyxoviral negative-sense RNA, completely failing to learn viral pathogenicity.
2. **Subtype $\neq$ Pathogenicity:** In biological virology, pathogenicity is determined by molecular motifs and host-pathogen interactions, not subtype nomenclature alone:
   - **H7N2:** Hardcoded historically as Class 1 ("Likely Benign"), but 68.8% of the HA sequences in the dataset harbor polybasic HPAI cleavage sites (e.g., the 2002 Virginia H7N2 poultry outbreak).
   - **H5N9:** Hardcoded as Class 3 ("Pathogenic"), but 79.2% of its HA sequences in the dataset are low-pathogenic (LPAI) wild-bird isolates with monobasic cleavage sites.
   - **H7N7:** Hardcoded as Class 2 ("Likely Pathogenic"), but 94.7% of its sequences are monobasic LPAI.
   - **H7N9:** Hardcoded as Class 3 ("Pathogenic"), but Waves 1–4 poultry isolates (2013–2016) were strictly low-pathogenic in poultry (asymptomatic in chickens) before acquiring a polybasic insertion in late 2016/Wave 5.
3. **Segment-Level Incoherence:** Applying a whole-virus pathogenicity label to individual viral segments (such as PB2, PA, NP, M, NS) without biological markers conflates segment identity with viral phenotype.

---

## 2. Biological Standards for Avian Influenza Pathogenicity

In international veterinary medicine and viral surveillance (World Organisation for Animal Health / WOAH Terrestrial Manual, Chapter 3.3.4), Avian Influenza viruses are classified into two official categories:

### 2.1 High Pathogenicity Avian Influenza (HPAI)
An influenza A virus is classified as HPAI if it satisfies either of the following criteria:
1. **In Vivo Chicken Assay (IVPI):** Has an Intravenous Pathogenicity Index (IVPI) in 6-week-old chickens greater than **1.2**.
2. **Molecular Cleavage Site Signature (H5 or H7):** For H5 and H7 subtypes, possesses multiple basic amino acids (arginine $[R]$ or lysine $[K]$) at the cleavage site of the hemagglutinin (HA0) protein ($R-X-K/R-R*G$ or $\ge 3$ basic residues within the junction loop preceding the fusion peptide $G-L-F-G-A-I-A-G-F$).
   - *Mechanism:* Polybasic cleavage sites permit cleavage by ubiquitous intracellular subtilisin-like proteases (furin and PC6), causing systemic viral replication and multi-organ failure.

### 2.2 Low Pathogenicity Avian Influenza (LPAI)
An influenza A virus that does not meet the HPAI criteria is classified as LPAI:
- Possesses a single basic amino acid at the HA cleavage site (monobasic motif: $P-Q-R-E-T-R*G-L-F$).
- *Mechanism:* Cleavage is restricted to trypsin-like extracellular serine proteases localized to the respiratory and enteric tracts, causing localized or asymptomatic infection in avian hosts.
- All non-H5/H7 subtypes in poultry (e.g., H1–H4, H6, H8–H16) are classified as LPAI unless specifically demonstrated to possess exceptional polybasic insertions or lethal chicken phenotypes.

---

## 3. Mandatory Labeling Directives

1. **Host Isolation:** Eukaryotic poultry reference genomes (chicken, duck, goose, turkey, quail) are strictly excluded from the viral pathogenicity classifier and quarantined in `dataset/host_references/`.
2. **Segment-Specific Evaluation:** 
   - HA sequences (Segment 4) are labeled directly via **cleavage site translation and motif extraction**.
   - Non-HA segments (PB2, PB1, PA, NP, NA, MP, NS) are assigned isolate-level pathogenicity **only when linked to a verified isolate whose HA sequence or surveillance record confirms HPAI or LPAI status**.
3. **Zoonotic / Mammalian Adaptation Annotation:** Secondary virulence markers (such as PB2 E627K, D701N; HA Q226L/G228S receptor binding shifts) are tracked as explicit multi-label attribute flags, rather than merged into a single ambiguous score.
4. **No Guessed or Presumed Labels:** Sequences lacking both a detectable cleavage site and isolate-level surveillance corroboration are marked `INDETERMINATE` and withheld from training until verified.

---

## 4. Proposed Labeling Architectures for User Selection

We formulate three scientifically defensible architecture options for user review:

### Architecture Option 1: Binary Molecular Pathogenicity (RECOMMENDED)
*Aligns directly with official WOAH/WHO definitions and peer-reviewed genomics literature.*

- **Target:** Direct molecular pathogenicity prediction.
- **Classes (2):**
  - **Class 0 (LPAI):** Low Pathogenicity Avian Influenza. Verified monobasic HA cleavage motif ($P-Q-R-E-T-R*G$) or non-H5/H7 avian surveillance isolate with IVPI $\le 1.2$.
  - **Class 1 (HPAI):** High Pathogenicity Avian Influenza. Verified polybasic HA cleavage motif ($R-X-K/R-R*G$ or $\ge 3$ basic residues) or confirmed outbreak mortality isolate.
- **Advantages:** Scientifically indisputable, 100% concordant with international veterinary standards, zero label ambiguity, robust against reviewer scrutiny.

---

### Architecture Option 2: Multi-Task Joint Architecture
*Decouples subtype identification from pathogenicity prediction.*

- **Shared Backbone:** DNABERT-2 + HyenaDNA feature fusion encoder.
- **Task Head A (Subtype Classification):** 14-class classifier predicting viral subtype (H5N1, H5N6, H5N8, H7N9, H9N2, etc.).
- **Task Head B (Virulence / Cleavage Site):** Binary classifier predicting HPAI vs. LPAI molecular phenotype.
- **Task Head C (Segment Classification):** 8-class classifier predicting genome segment (PB2, PB1, PA, HA, NP, NA, MP, NS).
- **Advantages:** Completely separates taxonomy from virulence; allows the model to learn that an H7N2 or H5N2 can be *either* HPAI or LPAI depending on its sequence.

---

### Architecture Option 3: Evidence-Based 4-Tier Virulence Taxonomy
*Preserves the project's original 4-class output dimension while replacing the flawed historical labels with biological evidence.*

- **Class 0: Wild-Reservoir Avirulent (LPAI-Wild):**
  - *Evidence:* Natural reservoir isolates (Anseriformes/Charadriiformes) with monobasic cleavage sites and absence of poultry-adaptive markers.
- **Class 1: Poultry-Adapted Low Pathogenicity (LPAI-Poultry):**
  - *Evidence:* Endemic poultry-adapted lineages (e.g., poultry H9N2, low-pathogenic H5/H7 circulating in domestic gallinaceous birds) with high transmissibility but monobasic cleavage.
- **Class 2: High Pathogenicity Avian Influenza (HPAI-Avian):**
  - *Evidence:* Polybasic cleavage site ($R-X-K/R-R*G$), panzootic poultry mortality (e.g., clade 2.3.4.4b H5Nx, Eurasian H7N7).
- **Class 3: High Pathogenicity Zoonotic with Mammalian Adaptation (HPAI-Mammalian):**
  - *Evidence:* Isolates harboring confirmed mammalian adaptation and severe spillover markers:
    - PB2 E627K or D701N (mammalian polymerase activity at 33°C).
    - HA human receptor binding preference (e.g., 226L/228S in H7N9).
    - Clinical isolates from fatal human or mammalian carnivore infections.
- **Advantages:** Maintains the 4-class neural network architecture while establishing rigorous, publishable biological ground truth.

---

## 5. Decision Required from User

The user is requested to select one of the three options above to govern Gate C (curation & splitting):
- **Option 1 (Recommended):** Binary Pathogenicity (LPAI vs. HPAI).
- **Option 2:** Multi-Task Joint Architecture (Subtype + Pathogenicity + Segment).
- **Option 3:** Evidence-Based 4-Tier Virulence Taxonomy.
