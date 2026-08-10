"use client";

import React, { useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { motion, AnimatePresence } from "framer-motion";
import { FileCode, Dna, GitBranch, Layers, Cpu, ShieldCheck, ArrowDown, ChevronRight, Sparkles, CheckCircle2 } from "lucide-react";

export default function PipelinePage() {
  const [activeStage, setActiveStage] = useState(0);

  const stages = [
    {
      id: 1,
      name: "1. FASTA File Discovery & Input QC",
      icon: FileCode,
      tag: "Ingestion",
      color: "border-sky-500/40 bg-sky-500/10 text-sky-400",
      purpose: "Locate raw and augmented FASTA sequence files across class folders and validate encoding integrity.",
      input: "FASTA sequence files (.fasta, .fa, .fna) in class subdirectories",
      output: "Validated list of (file_path, class_index) tuples",
      algorithm: "Subtype-to-Class Regex Matcher & Corrupt File Handler",
      advantages: "Prevents pipeline crashes due to invalid nucleotides or malformed FASTA headers.",
      nextInteraction: "Passes cleaned file paths to Preprocessing & Tokenization Engine.",
      details: [
        "Checks header formatting and IUPAC nucleotide characters",
        "Applies fallback parsing for multi-FASTA files",
        "Supports class-balanced directory layouts"
      ]
    },
    {
      id: 2,
      name: "2. k-mer Tokenization & Preprocessing",
      icon: Dna,
      tag: "Tokenization",
      color: "border-violet-500/40 bg-violet-500/10 text-violet-400",
      purpose: "Convert raw ACGT nucleotide strings into fixed-length token ID sequences matching biological k-mer patterns.",
      input: "Raw DNA nucleotide string (length ~500-2000 bp)",
      output: "PyTorch LongTensor of shape [128] token IDs in range [0, 4095]",
      algorithm: "Sliding Window Hexamer Tokenizer (k=6, stride=1, vocab_size=4096)",
      advantages: "Matches biological hexamer motifs while reducing sequence length by factor of 6.",
      nextInteraction: "Supplies tokenized tensor sequence to DataLoader & Sequence Encoder.",
      details: [
        "vocab_size = 4^6 = 4096 unique hexamer combinations",
        "Truncates or pads to uniform seq_len=128",
        "Configurable k-mer window size via config YAML"
      ]
    },
    {
      id: 3,
      name: "3. Pairwise Alignment & Mutation Matrix",
      icon: GitBranch,
      tag: "Mutation Alignment",
      color: "border-amber-500/40 bg-amber-500/10 text-amber-400",
      purpose: "Align input sequence against class reference sequences to extract explicit mutation vectors.",
      input: "Input nucleotide sequence + Reference FASTA sequences",
      output: "Float32 Tensor of shape [32, 64] representing mutation positions and edit distances",
      algorithm: "Pairwise Needleman-Wunsch / Smith-Waterman Alignment Engine",
      advantages: "Explicitly encodes single nucleotide polymorphisms (SNPs) and insertion/deletion events.",
      nextInteraction: "Passed directly to Feature Fusion Module alongside Hyena embeddings.",
      details: [
        "Extracts up to max_mutations=32 position features",
        "Encodes mutation dimension=64 per feature vector",
        "Calculates edit distance and mismatch count metrics"
      ]
    },
    {
      id: 4,
      name: "4. Stratified Split & Duplicate Auditing",
      icon: ShieldCheck,
      tag: "Data Integrity",
      color: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
      purpose: "Partition dataset into 70% Train, 15% Val, and 15% Test while guaranteeing representation for all 4 classes.",
      input: "Deduplicated sample pool (72 FASTA files across 4 classes)",
      output: "Train (48), Val (10), and Test (14) file list partitions",
      algorithm: "Per-Class Stratified Partitioning with Small-Class Reserve-Test First logic",
      advantages: "Eliminates class absence in test set (fixing 0 Pathogenic / 0 Benign bug). Zero file leakage.",
      nextInteraction: "Feeds partitioned file lists into GenomicDataset instances.",
      details: [
        "Deduplicates file paths before splitting",
        "For small classes (n<=4), reserves test sample first",
        "Performs automated set-intersection leakage check"
      ]
    },
    {
      id: 5,
      name: "5. Class-Weighted DataLoader Construction",
      icon: Cpu,
      tag: "Batching",
      color: "border-cyan-500/40 bg-cyan-500/10 text-cyan-400",
      purpose: "Prepare PyTorch DataLoaders with inverse-frequency class weighting and weighted random sampling.",
      input: "GenomicDataset instances for Train, Val, Test",
      output: "PyTorch DataLoaders (batch_size=16, num_workers=0, pin_memory=True)",
      algorithm: "WeightedRandomSampler + Inverse-Frequency CrossEntropy Loss Weights",
      advantages: "Prevents majority class (Likely_Benign) from dominating gradient updates.",
      nextInteraction: "Streams mini-batches to model during training & inference loop.",
      details: [
        "Calculates class weights: [11.98, 0.48, 0.80, 1.71]",
        "Exposes class_weights property for loss calculation",
        "Supports pin_memory for fast CUDA host-to-device transfers"
      ]
    },
    {
      id: 6,
      name: "6. HyenaDNA + Transformer Classification Engine",
      icon: Layers,
      tag: "Model Inference",
      color: "border-purple-500/40 bg-purple-500/10 text-purple-400",
      purpose: "Pass tokenized sequence and mutation tensor through hybrid neural architecture to compute class logits.",
      input: "token_ids [B, 128], mutation_vectors [B, 32, 64]",
      output: "Softmax probability distribution over 4 classes [B, 4]",
      algorithm: "HyenaBlock → TransformerEncoder → Feature Fusion Concatenation → Classifier Head",
      advantages: "10.21M parameters capture sub-quadratic long-range motifs and explicit mutation vectors.",
      nextInteraction: "Sends predicted probabilities to Metric Tracker & Evaluator.",
      details: [
        "Hyena embedding dimension: 256",
        "Transformer hidden dimension: 512, 4 heads",
        "Dropout=0.2, linear classification projection"
      ]
    },
    {
      id: 7,
      name: "7. Metrics Computation & Automated Reporting",
      icon: Sparkles,
      tag: "Evaluation",
      color: "border-rose-500/40 bg-rose-500/10 text-rose-400",
      purpose: "Calculate 8 scientific performance metrics and export complete evaluation artifacts.",
      input: "y_true ground truth labels + y_pred probabilities [B, 4]",
      output: "Metrics JSON, CSV classification report, ROC curves, Markdown & HTML reports",
      algorithm: "AdvancedMetricsTracker & AdvancedEvaluator pipeline",
      advantages: "Computes MCC (0.7821), Cohen's Kappa (0.7795), Balanced Acc (75.29%) and saves CSV/HTML reports.",
      nextInteraction: "Generates final deployable research artifacts.",
      details: [
        "Guards per-class ROC-AUC against single-class NaN bugs",
        "Exports PNG plots for confusion matrix, ROC, PR, confidence",
        "Generates interactive stand-alone HTML summary report"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Page Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              End-to-End Pipeline
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Data Processing & Inference Pipeline
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Explore every stage of the genomic AI framework from raw FASTA ingestion to multi-class evaluation reporting.
            </p>
          </div>

          {/* Stage Selector Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 mb-12">
            {stages.map((st, idx) => {
              const Icon = st.icon;
              const isCurrent = activeStage === idx;
              return (
                <button
                  key={st.id}
                  onClick={() => setActiveStage(idx)}
                  className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between ${
                    isCurrent
                      ? "bg-slate-900 border-sky-500 shadow-lg shadow-sky-500/20"
                      : "bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400"
                  }`}
                >
                  <Icon className={`w-5 h-5 mb-2 ${isCurrent ? "text-sky-400" : "text-slate-500"}`} />
                  <div>
                    <span className="text-[10px] font-mono text-slate-500 block">STAGE 0{st.id}</span>
                    <span className={`text-xs font-bold truncate block ${isCurrent ? "text-white" : "text-slate-300"}`}>
                      {st.tag}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Active Stage Detailed Card */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeStage}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="p-8 sm:p-10 rounded-3xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl mb-16 shadow-2xl relative overflow-hidden"
            >
              <div className="flex flex-wrap items-center justify-between gap-4 mb-8 pb-6 border-b border-slate-800">
                <div className="flex items-center gap-4">
                  <div className={`p-4 rounded-2xl border ${stages[activeStage].color}`}>
                    {React.createElement(stages[activeStage].icon, { className: "w-8 h-8" })}
                  </div>
                  <div>
                    <span className="text-xs font-mono font-semibold text-sky-400 uppercase tracking-wider">
                      {stages[activeStage].tag}
                    </span>
                    <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
                      {stages[activeStage].name}
                    </h2>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    disabled={activeStage === 0}
                    onClick={() => setActiveStage((p) => p - 1)}
                    className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-semibold disabled:opacity-30 text-slate-300"
                  >
                    Previous
                  </button>
                  <button
                    disabled={activeStage === stages.length - 1}
                    onClick={() => setActiveStage((p) => p + 1)}
                    className="px-3 py-1.5 rounded-lg bg-sky-500 text-slate-950 text-xs font-bold disabled:opacity-30"
                  >
                    Next Stage
                  </button>
                </div>
              </div>

              {/* Purpose */}
              <div className="mb-8">
                <h4 className="text-xs font-bold uppercase text-slate-400 tracking-wider mb-2">Stage Purpose</h4>
                <p className="text-slate-200 text-base leading-relaxed">{stages[activeStage].purpose}</p>
              </div>

              {/* Inputs & Outputs Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800">
                  <div className="text-xs font-bold uppercase text-sky-400 tracking-wider mb-2">Expected Input</div>
                  <div className="text-xs font-mono text-slate-300 bg-slate-900 p-3 rounded-xl border border-slate-800">
                    {stages[activeStage].input}
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800">
                  <div className="text-xs font-bold uppercase text-emerald-400 tracking-wider mb-2">Stage Output</div>
                  <div className="text-xs font-mono text-slate-300 bg-slate-900 p-3 rounded-xl border border-slate-800">
                    {stages[activeStage].output}
                  </div>
                </div>
              </div>

              {/* Algorithm & Advantages */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
                  <div className="text-xs font-bold uppercase text-violet-400 tracking-wider mb-2">Algorithms & Methods</div>
                  <p className="text-xs text-slate-300 leading-relaxed">{stages[activeStage].algorithm}</p>
                </div>

                <div className="p-5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
                  <div className="text-xs font-bold uppercase text-amber-400 tracking-wider mb-2">Key Advantage</div>
                  <p className="text-xs text-slate-300 leading-relaxed">{stages[activeStage].advantages}</p>
                </div>
              </div>

              {/* Implementation Details Bullet Points */}
              <div className="p-5 rounded-2xl bg-slate-950/40 border border-slate-800/60">
                <div className="text-xs font-bold uppercase text-slate-400 tracking-wider mb-3">Technical Highlights</div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {stages[activeStage].details.map((dt) => (
                    <div key={dt} className="flex items-center gap-2 text-xs text-slate-300">
                      <CheckCircle2 className="w-4 h-4 text-sky-400 shrink-0" />
                      <span>{dt}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Interaction with Next Module */}
              <div className="mt-8 pt-6 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300">Downstream Flow:</span>
                <span className="text-sky-400 font-medium flex items-center gap-1">
                  {stages[activeStage].nextInteraction}
                  <ChevronRight className="w-4 h-4" />
                </span>
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Vertical Flowchart View */}
          <div className="space-y-6">
            <h3 className="text-xl font-bold text-white mb-6 text-center">Complete Pipeline Flowchart</h3>
            <div className="space-y-4">
              {stages.map((st, idx) => (
                <div key={st.id} className="flex flex-col items-center">
                  <div className="w-full p-5 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono font-bold text-sky-400 bg-sky-500/10 px-2.5 py-1 rounded-lg border border-sky-500/20">
                        STAGE 0{st.id}
                      </span>
                      <span className="text-sm font-bold text-white">{st.name}</span>
                    </div>
                    <span className="text-xs text-slate-400 hidden sm:inline-block">{st.algorithm}</span>
                  </div>
                  {idx < stages.length - 1 && (
                    <ArrowDown className="w-5 h-5 text-sky-500/60 my-2 animate-bounce" />
                  )}
                </div>
              ))}
            </div>
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
