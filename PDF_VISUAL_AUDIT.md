# PDF Visual & Typographical Page-by-Page Audit

**Target Paper**: *Avian Influenza Genomic Intelligence: A Hybrid Deep Learning Architecture for Pathogenicity Category Prediction and Mutation-Aware Sequence Modeling*  
**Format**: Standard IEEE Two-Column Conference Format  
**Audit Date**: August 2026  

---

## Page-by-Page Verification Grid

| Page | Layout State | Components Inspected | Visual Quality Checks | Pass/Fail |
|:---:|:---:|---|---|:---:|
| **Page 1** | Two-Column | Title, Author Block, Abstract, Keywords, Section I (Intro), Fig. 1 (Architecture schematic spanning full width) | Title centered; affiliation clean; no stray markdown; abstract contained within margins; Fig. 1 spans both columns at top without clipping | **PASS** |
| **Page 2** | Two-Column | Section II (Related Work), Table I (Literature Comparison spanning 2 columns), Fig. 2 (Pipeline flow), Section III (Methodology: QC & Alignment) | Table I text wraps naturally across all 7 columns; Fig. 2 fits Column 1 cleanly; Eq. (1)-(4) aligned with standard equation numbering | **PASS** |
| **Page 3** | Two-Column | Section III cont. (DNABERT-2, HyenaDNA, Mutation Prioritization, Cross-Attention), Fig. 3 (Model Graph), Table II (Subtype Mapping) | Fig. 3 crisp inside Column 1; all tensor dimensions ($[B, 128, 768]$, $[B, 32, 64]$) visible; Table II wraps subtype lists; Eq. (5)-(24) verified | **PASS** |
| **Page 4** | Two-Column | Section IV (Dataset & Setup), Table III (Dataset Composition), Table IV (Augmentation Multipliers), Table V (Split Stats), Table VI (Hyperparameters) | Table III, IV, and V fit Column 1 and 2 perfectly with zero second-column overlap; all sequence counts ($35,211$) and base pair totals verified | **PASS** |
| **Page 5** | Two-Column | Section V (Results), Table VII (Parameter Distribution), Table VIII (Benchmark Metrics), Table IX (Per-Class Performance), Fig. 4 (Curves), Fig. 5 (Confusion Matrix) | Table VII module names wrapped without spilling into Column 2; Table VIII & IX clean; Fig. 4 & 5 rendered at 300 DPI with clear labels | **PASS** |
| **Page 6** | Two-Column | Section VI (Explainability), Fig. 6 (Class Performance), Fig. 7 (Feature Attribution), Section VII (Discussion), Section VIII (Ablation Protocol), Table X | Fig. 6 & 7 fit column width without obstructing adjacent text; Table X variant matrix clean; all attributions properly referenced | **PASS** |
| **Page 7** | Two-Column | Section IX (Limitations), Section X (Conclusion & Future Work), References (28 citations) | Conclusion summarizes results accurately; References block formatted cleanly in two IEEE columns with zero overlapping floats | **PASS** |

---

## Key Visual Verification Checks Completed:
- [x] **Margin Symmetry**: Visually equal left and right margins across all pages ($0.625\text{ in}$).
- [x] **Zero Float Spillover**: No tables or figures extend past their respective column boundaries.
- [x] **Reference Legibility**: References column is completely unobstructed by figures or floating tables.
- [x] **Equation Centering**: All 29 mathematical equations aligned and numbered cleanly.
- [x] **Table Typography**: Professional `booktabs` rules used exclusively (`\toprule`, `\midrule`, `\bottomrule`) with zero vertical table borders.
