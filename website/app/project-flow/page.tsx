"use client";

import React, { useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Database,
  Filter,
  Dna,
  GitBranch,
  Layers,
  ShieldCheck,
  Sparkles,
  Search,
  CheckCircle2,
  Terminal,
  ArrowRight,
  ArrowDown,
  Cpu
} from "lucide-react";
import Link from "next/link";

export default function ProjectFlowPage() {
  const [activeStep, setActiveStep] = useState(0);

  const workflowSteps = [
    {
      id: 1,
      title: "1. Problem Formulation & Planning",
      phase: "Planning",
      icon: FileText,
      color: "border-sky-500/40 bg-sky-500/10 text-sky-400",
      description: "Define 4-class pathogenicity classification scope for avian influenza viral genomes and establish performance metrics (MCC, Cohen's Kappa, Macro ROC-AUC).",
      deliverable: "Software Design Document (SDD) & Research Specifications"
    },
    {
      id: 2,
      title: "2. Bio-Curated Dataset Collection",
      phase: "Dataset",
      icon: Database,
      color: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
      description: "Collect 14 raw surveillance FASTA sequences across NCBI & GISAID repositories spanning Benign, Likely Benign, Likely Pathogenic, and Pathogenic isolates.",
      deliverable: "14 Raw FASTA files across 4 class directories"
    },
    {
      id: 3,
      title: "3. Quality Control & Preprocessing",
      phase: "Preprocessing",
      icon: Filter,
      color: "border-violet-500/40 bg-violet-500/10 text-violet-400",
      description: "Perform IUPAC nucleotide string validation, header standardization, N-density filtering, and synthetic SNP biological augmentation (5.14x factor).",
      deliverable: "72 Quality-Checked & Augmented FASTA files"
    },
    {
      id: 4,
      title: "4. k-mer Sequence Representation",
      phase: "Representation",
      icon: Dna,
      color: "border-amber-500/40 bg-amber-500/10 text-amber-400",
      description: "Slide 6-mer window (stride=1, vocab_size=4096) to tokenize sequences into dense integer PyTorch LongTensors of uniform length seq_len=128.",
      deliverable: "14,391 PyTorch Token Tensors [B, 128]"
    },
    {
      id: 5,
      title: "5. Feature Extraction (HyenaDNA + Alignment)",
      phase: "Feature Extraction",
      icon: GitBranch,
      color: "border-orange-500/40 bg-orange-500/10 text-orange-400",
      description: "Parallel extraction: HyenaDNA sub-quadratic long-context convolution (256d) + Needleman-Wunsch pairwise mutation matrix alignment [32, 64].",
      deliverable: "Hyena Embeddings [B, 128, 256] + Mutation Vector [B, 32, 64]"
    },
    {
      id: 6,
      title: "6. Self-Attention & Feature Fusion",
      phase: "Feature Fusion",
      icon: Layers,
      color: "border-purple-500/40 bg-purple-500/10 text-purple-400",
      description: "Pass Hyena features through 2-layer TransformerEncoder (4 attention heads, 512d) and concatenate mean-pooled features with flattened mutation matrix.",
      deliverable: "Fused Feature Vector [B, 2560]"
    },
    {
      id: 7,
      title: "7. Multi-Class Logit Classification",
      phase: "Classification",
      icon: ShieldCheck,
      color: "border-rose-500/40 bg-rose-500/10 text-rose-400",
      description: "Project fused vectors through Linear(2560 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 4) to output class probability logits.",
      deliverable: "Softmax Probability Tensor [B, 4]"
    },
    {
      id: 8,
      title: "8. Stratified Evaluation & Verification",
      phase: "Evaluation",
      icon: Sparkles,
      color: "border-cyan-500/40 bg-cyan-500/10 text-cyan-400",
      description: "Evaluate hold-out test set (2,791 sequence windows) calculating Test Accuracy (86.03%), Balanced Acc (75.29%), ROC-AUC (89.47%), MCC (0.7821), and Kappa (0.7795).",
      deliverable: "evaluation_metrics.json & classification_report.csv"
    },
    {
      id: 9,
      title: "9. Explainability & Diagnostics",
      phase: "Explainability",
      icon: Search,
      color: "border-yellow-500/40 bg-yellow-500/10 text-yellow-400",
      description: "Generate prediction confidence histograms, per-class ROC/PR curves, 4x4 confusion matrices, and feature importance interpretations.",
      deliverable: "PNG Plot Artifacts in eval_results_fixed/plots/"
    },
    {
      id: 10,
      title: "10. Automated Report Generation",
      phase: "Prediction",
      icon: Terminal,
      color: "border-teal-500/40 bg-teal-500/10 text-teal-400",
      description: "Generate sample-by-sample predictions CSV/JSON, Markdown documentation, and standalone HTML executive report.",
      deliverable: "evaluation_report.html & predictions.csv"
    },
    {
      id: 11,
      title: "11. Production Deployment & MLOps",
      phase: "Deployment",
      icon: Cpu,
      color: "border-blue-500/40 bg-blue-500/10 text-blue-400",
      description: "Package PyTorch model state dict (`best_model.pt`), CLI entrypoints (`evaluate_advanced.py`), and Next.js research portal for public access.",
      deliverable: "Deployable PyTorch Model & Research Web Portal"
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Page Hero */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              End-to-End Scientific Architecture
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Complete Research Workflow
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Trace the 11 sequential phases of the genomic AI framework from initial planning to production model deployment.
            </p>
          </div>

          {/* Step Selector Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2 mb-12">
            {workflowSteps.map((step, idx) => {
              const Icon = step.icon;
              const isCurrent = activeStep === idx;
              return (
                <button
                  key={step.id}
                  onClick={() => setActiveStep(idx)}
                  className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between ${
                    isCurrent
                      ? "bg-slate-900 border-sky-500 shadow-lg shadow-sky-500/20"
                      : "bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400"
                  }`}
                >
                  <Icon className={`w-4 h-4 mb-2 ${isCurrent ? "text-sky-400" : "text-slate-500"}`} />
                  <div>
                    <span className="text-[9px] font-mono text-slate-500 block">PHASE 0{step.id}</span>
                    <span className={`text-xs font-bold truncate block ${isCurrent ? "text-white" : "text-slate-300"}`}>
                      {step.phase}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Active Step Detailed Card */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeStep}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="p-8 sm:p-10 rounded-3xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl mb-16 shadow-2xl relative overflow-hidden"
            >
              <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800">
                <div className="flex items-center gap-4">
                  <div className={`p-3.5 rounded-2xl border ${workflowSteps[activeStep].color}`}>
                    {React.createElement(workflowSteps[activeStep].icon, { className: "w-7 h-7" })}
                  </div>
                  <div>
                    <span className="text-xs font-mono font-semibold text-sky-400 uppercase tracking-wider">
                      PHASE 0{workflowSteps[activeStep].id} — {workflowSteps[activeStep].phase}
                    </span>
                    <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
                      {workflowSteps[activeStep].title}
                    </h2>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    disabled={activeStep === 0}
                    onClick={() => setActiveStep((p) => p - 1)}
                    className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-semibold disabled:opacity-30 text-slate-300"
                  >
                    Previous
                  </button>
                  <button
                    disabled={activeStep === workflowSteps.length - 1}
                    onClick={() => setActiveStep((p) => p + 1)}
                    className="px-3 py-1.5 rounded-lg bg-sky-500 text-slate-950 text-xs font-bold disabled:opacity-30"
                  >
                    Next Phase
                  </button>
                </div>
              </div>

              <div className="space-y-6">
                <div>
                  <h4 className="text-xs font-bold uppercase text-slate-400 tracking-wider mb-2">Phase Description</h4>
                  <p className="text-slate-200 text-base leading-relaxed">{workflowSteps[activeStep].description}</p>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-xs font-bold uppercase text-emerald-400 block mb-1">Key Deliverable</span>
                  <span className="text-xs font-mono text-slate-200">{workflowSteps[activeStep].deliverable}</span>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Interactive Step-by-Step Vertical Flowchart */}
          <div className="space-y-6">
            <h3 className="text-xl font-bold text-white mb-6 text-center">Complete Research Flowchart</h3>
            <div className="space-y-3 max-w-4xl mx-auto">
              {workflowSteps.map((st, idx) => (
                <div key={st.id} className="flex flex-col items-center">
                  <div
                    onClick={() => setActiveStep(idx)}
                    className={`w-full p-4 rounded-2xl border transition-all cursor-pointer flex items-center justify-between gap-4 ${
                      activeStep === idx
                        ? "bg-slate-900 border-sky-500/80 shadow-lg shadow-sky-500/10"
                        : "bg-slate-950/80 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono font-bold text-sky-400 bg-sky-500/10 px-2.5 py-1 rounded-lg border border-sky-500/20">
                        PHASE {st.id}
                      </span>
                      <span className="text-sm font-bold text-white">{st.title}</span>
                    </div>
                    <span className="text-xs font-mono text-slate-400 hidden sm:inline-block">{st.deliverable}</span>
                  </div>
                  {idx < workflowSteps.length - 1 && (
                    <ArrowDown className="w-4 h-4 text-sky-500/60 my-1.5 animate-bounce" />
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
