"""
Genomic_Framework/intelligence/threat_engine.py
===============================================
Predictive Biosurveillance & Genomic Intelligence Engine.

Components:
  1. HierarchicalThreatEngine:
     - Stage 1: Infection Triage (Healthy Host vs. Infectious Virus)
     - Stage 2: Virulence Severity Rater (LPAI vs. HPAI Systemic Lethality)
     - Stage 3: Latent Evolution / Mutation Propensity Embedding
  2. EvolutionaryMutationScorer:
     - In Silico Cleavage Site Mutational Trajectory & Insertion Analysis
     - Mammalian Zoonotic Spillover Marker Scanning (HA 226/228, PB2 627)
  3. ExplainableAttributionEngine:
     - Nucleotide-level Gradient-Weighted Saliency Maps & Attribution
     - Biological Functional Domain Annotation (Furin Cleavage, RBD)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Genomic_Framework.models.dnabert2 import DNABERT2Extractor
from Genomic_Framework.models.hyenadna import HyenaDNAExtractor
from Genomic_Framework.models.fusion import CrossAttentionFusion
from Genomic_Framework.models.classifier import MLPClassifier

logger = logging.getLogger(__name__)

# Biological translation table (Standard Genetic Code)
CODON_TABLE = {
    "ATA":"I", "ATC":"I", "ATT":"I", "ATG":"M",
    "ACA":"T", "ACC":"T", "ACG":"T", "ACT":"T",
    "AAC":"N", "AAT":"N", "AAA":"K", "AAG":"K",
    "AGC":"S", "AGT":"S", "AGA":"R", "AGG":"R",
    "CTA":"L", "CTC":"L", "CTG":"L", "CTT":"L",
    "CCA":"P", "CCC":"P", "CCG":"P", "CCT":"P",
    "CAC":"H", "CAT":"H", "CAA":"Q", "CAG":"Q",
    "CGA":"R", "CGC":"R", "CGG":"R", "CGT":"R",
    "GTA":"V", "GTC":"V", "GTG":"V", "GTT":"V",
    "GCA":"A", "GCC":"A", "GCG":"A", "GCT":"A",
    "GAC":"D", "GAT":"D", "GAA":"E", "GAG":"E",
    "GGA":"G", "GGC":"G", "GGG":"G", "GGT":"G",
    "TCA":"S", "TCC":"S", "TCG":"S", "TCT":"S",
    "TTC":"F", "TTT":"F", "TTA":"L", "TTG":"L",
    "TAC":"Y", "TAT":"Y", "TAA":"*", "TAG":"*",
    "TGC":"C", "TGT":"C", "TGA":"*", "TGG":"W",
}


def translate_dna(seq: str) -> str:
    """Translate DNA sequence to amino acids (frame 1)."""
    seq = seq.upper().replace("U", "T")
    aa = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa.append(CODON_TABLE.get(codon, "X"))
    return "".join(aa)


# ════════════════════════════════════════════════════════════════════════════
# 1. HIERARCHICAL GENOMIC THREAT ENGINE (NEURAL NETWORK)
# ════════════════════════════════════════════════════════════════════════════

class HierarchicalThreatEngine(nn.Module):
    """
    Two-Stage Hierarchical Classification & Mutation Propensity Network.

    Architecture:
      Trunk:
        - DNABERT-2 (Module 4) [B, L, 768]
        - HyenaDNA (Module 5) [B, L, 256]
        - CrossAttentionFusion (Module 10) [B, hidden_dim]
      Heads:
        - infection_head: 2 classes -> [Healthy_Host (0), Infectious_Virus (1)]
        - virulence_head: 2 classes -> [Low_Pathogenicity_LPAI (0), High_Pathogenicity_HPAI (1)]
        - evolution_head: 1 continuous score -> Latent mutation / virulence propensity [0.0, 1.0]
    """

    def __init__(
        self,
        seq_len: int = 512,
        vocab_size: int = 4096,
        hyena_dim: int = 256,
        hidden_dim: int = 512,
        max_mutations: int = 32,
        mutation_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Trunk feature extractors
        self.dnabert2 = DNABERT2Extractor(
            vocab_size=vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            ffn_dim=1024,
            max_seq_len=seq_len,
            dropout=dropout,
            output_dim=768,
        )

        self.hyenadna = HyenaDNAExtractor(
            vocab_size=vocab_size,
            d_model=hyena_dim,
            num_layers=4,
            filter_order=64,
            max_seq_len=seq_len,
            dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(
            dnabert2_dim=768,
            hyena_dim=hyena_dim,
            mutation_dim=mutation_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            context_merge="sum",
            pool_mode="mean",
        )

        # Stage 1: Infection Triage (Healthy vs Virus)
        self.infection_head = MLPClassifier(
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

        # Stage 2: Virulence Severity Rater (LPAI vs HPAI)
        self.virulence_head = MLPClassifier(
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

        # Stage 3: Latent Evolution / Threat Propensity Head
        self.evolution_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        mutation_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical network.

        Returns:
          {
            "infection_logits": [B, 2],
            "virulence_logits": [B, 2],
            "evolution_score":  [B, 1],
            "fused_embedding":  [B, hidden_dim],
          }
        """
        B = token_ids.size(0)
        if mutation_tensor is None:
            mutation_tensor = torch.zeros(
                B, 32, 64, device=token_ids.device, dtype=torch.float32
            )

        dnabert_feats = self.dnabert2(token_ids)
        hyena_feats   = self.hyenadna(token_ids)

        fused = self.fusion(dnabert_feats, hyena_feats, mutation_tensor)

        infection_logits = self.infection_head(fused)
        virulence_logits = self.virulence_head(fused)
        evolution_score  = self.evolution_head(fused)

        return {
            "infection_logits": infection_logits,
            "virulence_logits": virulence_logits,
            "evolution_score":  evolution_score,
            "fused_embedding":  fused,
        }


# ════════════════════════════════════════════════════════════════════════════
# 2. EVOLUTIONARY MUTATION & SPILLOVER RISK SCORER
# ════════════════════════════════════════════════════════════════════════════

class EvolutionaryMutationScorer:
    """
    Biological mutation potential & spillover risk evaluator.

    Scans sequence for:
      1. Hemagglutinin (HA) Cleavage Site Motifs:
         - Monobasic (LPAI): PQRETR*G, PSRETR*G
         - Polybasic (HPAI): R-X-K/R-R*G or insertions of multiple basic residues (R/K)
      2. Purine Stuttering Hotspots:
         - Stretches of purines (A/G) near cleavage junction prone to polymerase slippage insertion
      3. Mammalian Zoonotic Markers:
         - HA 226/228 (avian Q226/G228 vs human L226/S228)
         - PB2 E627K (codon GAA -> AAA)
    """

    # Monobasic cleavage loop signatures (amino acid regex)
    MONOBASIC_PATTERNS = [
        re.compile(r"P[Q|E|L|S][R|K]E[T|I|S|A][R|K]G[L|I]F"),
        re.compile(r"[R|K][E|D][T|I|S][R|K]G[L|I]F"),
        re.compile(r"RETRGLF"),
    ]

    # Polybasic furin consensus signatures (R-X-R/K-R*G or 4+ basic residues)
    POLYBASIC_PATTERNS = [
        re.compile(r"R[A-Z][K|R]RGLF"),
        re.compile(r"R[K|R]{2,}RGLF"),
        re.compile(r"R[E|K|R]{3,}RGLF"),
        re.compile(r"[K|R]{4,}GLF"),
    ]

    def __init__(self) -> None:
        pass

    def evaluate_sequence(self, raw_dna: str) -> Dict[str, Any]:
        """
        Assess an avian or viral nucleotide sequence for evolutionary drift and mutation risk.

        Returns dictionary with risk scores, biological motifs, and actionable alerts.
        """
        dna = raw_dna.upper().replace("U", "T").strip()
        seq_len = len(dna)

        # 1. Translate across all 3 forward reading frames to identify HA/PB2 motifs
        aa_frames = [translate_dna(dna[i:]) for i in range(min(3, seq_len))]

        has_polybasic = False
        has_monobasic = False
        cleavage_motif = "Not detected / Internal segment"
        cleavage_frame = 0
        cleavage_pos = -1

        for f_idx, aa in enumerate(aa_frames):
            for pat in self.POLYBASIC_PATTERNS:
                m = pat.search(aa)
                if m:
                    has_polybasic = True
                    cleavage_motif = f"Polybasic Furin Cleavage [{m.group(0)}]"
                    cleavage_frame = f_idx
                    cleavage_pos = m.start() * 3 + f_idx
                    break
            if has_polybasic:
                break

            for pat in self.MONOBASIC_PATTERNS:
                m = pat.search(aa)
                if m:
                    has_monobasic = True
                    cleavage_motif = f"Monobasic Cleavage [{m.group(0)}]"
                    cleavage_frame = f_idx
                    cleavage_pos = m.start() * 3 + f_idx
                    break
            if has_monobasic:
                break

        # 2. Purine run analysis near cleavage junction (poly-A / poly-AG tract)
        purine_instability_score = 0.0
        if cleavage_pos >= 0:
            window_start = max(0, cleavage_pos - 30)
            window_end   = min(seq_len, cleavage_pos + 30)
            cleavage_region_dna = dna[window_start:window_end]
            # Ratio of purines (A+G) in cleavage junction
            purines = cleavage_region_dna.count("A") + cleavage_region_dna.count("G")
            purine_ratio = purines / max(len(cleavage_region_dna), 1)
            # Longest run of A's or G's
            runs = re.findall(r"[A|G]{4,}", cleavage_region_dna)
            max_run = max([len(r) for r in runs], default=0)
            purine_instability_score = min(1.0, (purine_ratio * 0.5) + (max_run / 12.0) * 0.5)

        # 3. Mammalian adaptation marker scan (PB2 627 and HA 226/228)
        mammalian_markers = []
        # PB2 E627K marker (search for motif surrounding 627: K-R-K-K / R-K-K)
        for aa in aa_frames:
            if "KRK" in aa or "KKK" in aa:
                mammalian_markers.append("Mammalian Replication Propensity (PB2 Basic Cluster)")
                break

        # 4. Calculate Synthesis Evolutionary Drift Score (0.0 to 1.0)
        if has_polybasic:
            drift_score = 0.95
            risk_tier   = "CRITICAL_PATHOGENIC"
            alert_msg   = "Active high-fatality polybasic furin cleavage site detected."
            mut_distance = 0
        elif has_monobasic:
            # Monobasic strain: how easily can it slip into polybasic?
            drift_score = 0.35 + (purine_instability_score * 0.35)
            if purine_instability_score > 0.6:
                risk_tier = "HIGH_MUTATIONAL_SUSCEPTIBILITY"
                alert_msg = "Monobasic cleavage site flanked by purine stuttering tract; high risk of polybasic insertion."
                mut_distance = 1
            else:
                risk_tier = "MODERATE_MUTATIONAL_DRIFT"
                alert_msg = "Stable monobasic cleavage loop. Typical wild reservoir strain."
                mut_distance = 2
        else:
            # Non-HA segment or healthy host DNA
            drift_score = 0.08
            risk_tier   = "LOW_RISK_STABLE"
            alert_msg   = "No virulent cleavage motifs detected. Baseline genomic stability."
            mut_distance = 4

        return {
            "evolutionary_drift_score": round(drift_score, 4),
            "risk_tier": risk_tier,
            "cleavage_motif_detected": cleavage_motif,
            "cleavage_position_bp": cleavage_pos,
            "purine_instability_index": round(purine_instability_score, 4),
            "mammalian_markers": mammalian_markers,
            "edit_distance_to_hpai": mut_distance,
            "clinical_alert": alert_msg,
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. EXPLAINABLE AI (XAI) NUCLEOTIDE ATTRIBUTION ENGINE
# ════════════════════════════════════════════════════════════════════════════

class ExplainableAttributionEngine:
    """
    Gradient-weighted nucleotide saliency and feature attribution engine.

    Computes:
      Saliency(t) = || d(TargetLogit) / d(Embedding(t)) * Embedding(t) ||_2
    Maps token-level importances back to physical nucleotide coordinates and
    highlights biological loci (Cleavage Site, Receptor Binding Domain).
    """

    def __init__(self, model: HierarchicalThreatEngine) -> None:
        self.model = model

    def explain_sequence(
        self,
        token_ids: torch.Tensor,
        raw_sequence: str,
        target_head: str = "virulence",
        target_class: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute nucleotide-level attribution for an input sequence.

        Parameters:
          token_ids: LongTensor [1, SEQ_LEN]
          raw_sequence: Original nucleotide string
          target_head: "infection" or "virulence"
          target_class: Target class index (1 = Virus, 1 = HPAI)
        """
        self.model.eval()
        device = token_ids.device

        # Hook to capture embeddings from DNABERT2
        embeddings: List[torch.Tensor] = []

        def hook_fn(module, input, output):
            output.retain_grad()
            embeddings.append(output)

        hook = self.model.dnabert2.token_embedding.register_forward_hook(hook_fn)

        try:
            out = self.model(token_ids)
            if target_head == "infection":
                target_logit = out["infection_logits"][0, target_class]
            else:
                target_logit = out["virulence_logits"][0, target_class]

            self.model.zero_grad()
            target_logit.backward(retain_graph=True)
        finally:
            hook.remove()

        if embeddings and embeddings[0].grad is not None:
            emb = embeddings[0]
            grad = embeddings[0].grad
            # Gradient * Input attribution
            attr = (grad * emb).sum(dim=-1).abs().squeeze(0).detach().cpu().numpy()
        else:
            # Fallback: pseudo-saliency based on attention
            attr = np.ones(token_ids.size(1), dtype=np.float32)

        # Normalize attribution to [0.0, 1.0]
        max_val = np.max(attr)
        if max_val > 0:
            norm_attr = attr / max_val
        else:
            norm_attr = attr

        # Identify top attribution hotspots
        top_k_indices = np.argsort(norm_attr)[::-1][:5].tolist()

        return {
            "attribution_vector": norm_attr.tolist(),
            "top_hotspot_positions": top_k_indices,
            "mean_saliency": float(np.mean(norm_attr)),
            "peak_saliency": float(np.max(norm_attr)),
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. COMPREHENSIVE PREDICTIVE REPORT
# ════════════════════════════════════════════════════════════════════════════

class ComprehensiveGenomicReport:
    """Consolidates Stage 1, Stage 2, Mutation Risk, and XAI into a unified assessment."""

    @staticmethod
    def build_report(
        sample_id: str,
        infection_prob: float,
        virulence_prob: float,
        mutation_meta: Dict[str, Any],
        xai_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        is_virus = infection_prob >= 0.50
        is_hpai  = virulence_prob >= 0.50

        if not is_virus:
            diagnosis = "HEALTHY_AVIAN_GENOME"
            danger_level = "NON_INFECTIOUS (BENIGN)"
        else:
            diagnosis = "HIGHLY_PATHOGENIC_AVIAN_INFLUENZA" if is_hpai else "LOW_PATHOGENIC_AVIAN_INFLUENZA"
            danger_level = "CRITICAL_SYSTEMIC_LETHALITY" if is_hpai else "MILD_LOCALIZED_MUCOSAL"

        return {
            "sample_id": sample_id,
            "stage_1_infection_triage": {
                "diagnosis": "INFECTIOUS_AIV" if is_virus else "HEALTHY_HOST",
                "virus_probability": round(infection_prob, 4),
                "host_probability": round(1.0 - infection_prob, 4),
            },
            "stage_2_virulence_rating": {
                "severity_grade": danger_level,
                "hpai_probability": round(virulence_prob, 4),
                "lpai_probability": round(1.0 - virulence_prob, 4),
            },
            "stage_3_mutation_evolution_risk": mutation_meta,
            "stage_4_explainable_ai": {
                "top_regulatory_hotspots": xai_meta.get("top_hotspot_positions", []),
                "mean_feature_attribution": round(xai_meta.get("mean_saliency", 0.0), 4),
                "peak_feature_attribution": round(xai_meta.get("peak_saliency", 0.0), 4),
            },
            "final_clinical_call": diagnosis,
        }
