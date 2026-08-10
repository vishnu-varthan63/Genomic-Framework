"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { motion } from "framer-motion";
import { Layers, Dna, Cpu, GitBranch, ShieldCheck, CheckCircle2, ArrowRight } from "lucide-react";
import { metrics } from "@/lib/content";

export default function ArchitecturePage() {
  const modules = [
    {
      name: "GenomicClassificationEngine",
      role: "Main Engine",
      params: "10.21M Parameters",
      desc: "Unified PyTorch Module (`nn.Module`) wrapping HyenaDNA sequence backbone, mutation aligner, multi-head self-attention transformer layers, and 4-class linear projection head.",
      inputs: "token_ids: IntTensor [B, 128], mutation_vector: FloatTensor [B, 32, 64]",
      outputs: "logits: FloatTensor [B, 4] -> Softmax Probabilities [B, 4]",
      icon: Layers,
      color: "border-sky-500/30 text-sky-400 bg-sky-500/10",
    },
    {
      name: "k-mer Tokenizer & Embedding",
      role: "Sequence Representation",
      params: "1.05M Parameters",
      desc: "Maps 6-mer nucleotide windows across vocabulary size 4,096 into dense 256-dimensional embeddings. Includes positional encoding.",
      inputs: "Raw DNA string -> token_ids [B, 128]",
      outputs: "embedded_seq: FloatTensor [B, 128, 256]",
      icon: Dna,
      color: "border-violet-500/30 text-violet-400 bg-violet-500/10",
    },
    {
      name: "HyenaDNA Long-Context Block",
      role: "Long-Range Sequence Backbone",
      params: "4.82M Parameters",
      desc: "Sub-quadratic implicit long convolution filter captures global genomic context across sequence segments without full quadratic self-attention matrix cost.",
      inputs: "embedded_seq [B, 128, 256]",
      outputs: "hyena_features [B, 128, 256]",
      icon: Cpu,
      color: "border-emerald-500/30 text-emerald-400 bg-emerald-500/10",
    },
    {
      name: "Pairwise Mutation Alignment Processor",
      role: "Mutation Feature Extractor",
      params: "0.45M Parameters",
      desc: "Smith-Waterman alignment engine extracts position-wise nucleotide substitutions and insertion/deletion events relative to class reference strains.",
      inputs: "FASTA sequence + Reference Strains",
      outputs: "mutation_vector [B, 32, 64]",
      icon: GitBranch,
      color: "border-amber-500/30 text-amber-400 bg-amber-500/10",
    },
    {
      name: "Transformer Encoder Layer (4 Heads)",
      role: "Self-Attention Mechanism",
      params: "3.14M Parameters",
      desc: "2-layer TransformerEncoder with 4-head MultiheadAttention and FeedForward network (hidden_dim=512) for refine-step feature contextualization.",
      inputs: "hyena_features [B, 128, 256]",
      outputs: "transformer_features [B, 128, 512]",
      icon: Layers,
      color: "border-cyan-500/30 text-cyan-400 bg-cyan-500/10",
    },
    {
      name: "Feature Fusion & Classifier Head",
      role: "Classification Projection",
      params: "0.75M Parameters",
      desc: "Concatenates mean-pooled sequence features [B, 512] with flattened mutation features [B, 2048] -> Linear(2560 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 4).",
      inputs: "fused_vector [B, 2560]",
      outputs: "class_logits [B, 4]",
      icon: ShieldCheck,
      color: "border-purple-500/30 text-purple-400 bg-purple-500/10",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              System Architecture & Module Design
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Hybrid Neural Network Engine
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Combining sub-quadratic Hyena DNA implicit convolutions, 4-head Transformer attention, and pairwise mutation feature fusion.
            </p>
          </div>

          {/* Parameter Breakdown Banner */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-16">
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 text-center">
              <div className="text-xs text-slate-400 font-semibold uppercase mb-1">Total Parameters</div>
              <div className="text-3xl font-extrabold text-white">{metrics.totalParams}</div>
              <div className="text-[11px] text-sky-400 mt-1">Trainable PyTorch Params</div>
            </div>
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 text-center">
              <div className="text-xs text-slate-400 font-semibold uppercase mb-1">Vocabulary Size</div>
              <div className="text-3xl font-extrabold text-white">4,096</div>
              <div className="text-[11px] text-violet-400 mt-1">6-mer Hexamer Tokens</div>
            </div>
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 text-center">
              <div className="text-xs text-slate-400 font-semibold uppercase mb-1">Sequence Length</div>
              <div className="text-3xl font-extrabold text-white">128 Tokens</div>
              <div className="text-[11px] text-emerald-400 mt-1">Window Length</div>
            </div>
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 text-center">
              <div className="text-xs text-slate-400 font-semibold uppercase mb-1">Classes</div>
              <div className="text-3xl font-extrabold text-white">4 Tiers</div>
              <div className="text-[11px] text-amber-400 mt-1">Pathogenicity Labels</div>
            </div>
          </div>

          {/* Interactive ASCII / Data Flow Diagram */}
          <div className="p-8 sm:p-10 rounded-3xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl mb-16">
            <h2 className="text-xl font-bold text-white mb-6">Data & Tensor Flow Architecture</h2>
            
            <div className="overflow-x-auto">
              <div className="min-w-[700px] space-y-4 font-mono text-xs text-slate-300">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                  <span className="text-sky-400 font-bold">1. INPUT FASTA</span>
                  <span className="text-slate-500">ACGTACGT... (DNA String)</span>
                  <span className="text-emerald-400">Length ~1000 bp</span>
                </div>
                <div className="text-center text-sky-500">↓ sliding window tokenization (k=6)</div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                  <span className="text-violet-400 font-bold">2. TOKEN IDENTIFIERS</span>
                  <span className="text-slate-300">[1787, 2341, 401, ..., 3375]</span>
                  <span className="text-emerald-400">Tensor: [B, 128]</span>
                </div>
                <div className="text-center text-sky-500">↓ parallel path 1: HyenaDNA Backbone | parallel path 2: Mutation Alignment</div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <span className="text-sky-400 font-bold block">Path 1: HyenaDNA + Transformer</span>
                    <span className="text-slate-400 block">Hyena implicit convolutions (256d) → TransformerEncoder 4-heads (512d)</span>
                    <span className="text-emerald-400 font-bold block">Output: [B, 128, 512] → Mean Pool [B, 512]</span>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <span className="text-amber-400 font-bold block">Path 2: Mutation Alignment</span>
                    <span className="text-slate-400 block">Smith-Waterman pairwise reference alignment → max 32 features x 64 dims</span>
                    <span className="text-emerald-400 font-bold block">Output: [B, 32, 64] → Flatten [B, 2048]</span>
                  </div>
                </div>
                
                <div className="text-center text-sky-500">↓ Feature Fusion Concatenation: [B, 512 + 2048] = [B, 2560]</div>

                <div className="p-4 rounded-xl bg-slate-950 border border-sky-500/40 flex items-center justify-between">
                  <span className="text-purple-400 font-bold">3. CLASSIFICATION HEAD</span>
                  <span className="text-slate-300">Linear(2560→256) → ReLU → Dropout(0.2) → Linear(256→4)</span>
                  <span className="text-emerald-400 font-bold">Logits [B, 4]</span>
                </div>
              </div>
            </div>
          </div>

          {/* Modules Detail Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
            {modules.map((mod) => {
              const Icon = mod.icon;
              return (
                <div key={mod.name} className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-sky-500/30 transition-all flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className={`p-3 rounded-xl border ${mod.color}`}>
                        <Icon className="w-5 h-5" />
                      </div>
                      <span className="text-xs font-mono font-bold text-slate-400 bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800">
                        {mod.params}
                      </span>
                    </div>

                    <h3 className="text-lg font-bold text-white mb-1">{mod.name}</h3>
                    <span className="text-xs font-semibold text-sky-400 block mb-3 uppercase tracking-wider">{mod.role}</span>
                    <p className="text-xs text-slate-300 leading-relaxed mb-4">{mod.desc}</p>
                  </div>

                  <div className="space-y-2 pt-4 border-t border-slate-800 text-[11px] font-mono">
                    <div className="flex justify-between text-slate-400">
                      <span>IN:</span>
                      <span className="text-slate-200">{mod.inputs}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>OUT:</span>
                      <span className="text-emerald-400 font-bold">{mod.outputs}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
