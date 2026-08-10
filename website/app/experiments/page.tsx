"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { motion } from "framer-motion";
import { Sliders, Database, Cpu, ShieldCheck, CheckCircle2, Terminal } from "lucide-react";
import { metrics } from "@/lib/content";

export default function ExperimentsPage() {
  const hparams = [
    { key: "optimizer", val: "AdamW", desc: "Adaptive Moment Estimation with Weight Decay" },
    { key: "learning_rate", val: "0.0001 (1e-4)", desc: "Base initial learning rate" },
    { key: "weight_decay", val: "0.01", desc: "L2 regularization penalty" },
    { key: "betas", val: "[0.9, 0.999]", desc: "Adam momentum coefficients" },
    { key: "scheduler", val: "CosineAnnealing + Warmup", desc: "5-epoch linear warmup -> cosine decay" },
    { key: "warmup_epochs", val: "5", desc: "Linear learning rate warmup duration" },
    { key: "min_lr", val: "1e-6", desc: "Minimum learning rate threshold" },
    { key: "batch_size", val: "16", desc: "Mini-batch size per optimization step" },
    { key: "gradient_clipping", val: "1.0", desc: "Maximum gradient norm threshold" },
    { key: "use_amp", val: "True (Mixed Precision)", desc: "Automatic mixed precision FP16/FP32" },
    { key: "max_epochs", val: "50", desc: "Maximum training epochs" },
    { key: "early_stopping_patience", val: "15", desc: "Epochs without val_f1 improvement" },
  ];

  const datasetStats = [
    { key: "Total Sequences", val: "72 FASTA Files", detail: "Class-balanced root directory layout" },
    { key: "Benign (Class 0)", val: "3 Files (4%)", detail: "Assigned 1 train, 1 val, 1 test" },
    { key: "Likely Benign (Class 1)", val: "37 Files (51%)", detail: "Assigned 25 train, 5 val, 7 test" },
    { key: "Likely Pathogenic (Class 2)", val: "22 Files (31%)", detail: "Assigned 15 train, 3 val, 4 test" },
    { key: "Pathogenic (Class 3)", val: "10 Files (14%)", detail: "Assigned 7 train, 1 val, 2 test" },
    { key: "Partitioning Ratio", val: "70 / 15 / 15", detail: "Train (48), Val (10), Test (14)" },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              Experimental Setup & Configuration
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Training Hyperparameters & Environment
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Complete specification of optimization settings, dataset splits, hardware environment, and reproducibility flags used during 50-epoch model training.
            </p>
          </div>

          {/* Hyperparameters Grid */}
          <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl mb-16">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20">
                <Sliders className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Optimization & Hyperparameter Matrix</h2>
                <p className="text-xs text-slate-400">Specified in `configs/train_config.yaml`</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {hparams.map((hp) => (
                <div key={hp.key} className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 font-mono text-xs">
                  <span className="text-slate-500 uppercase tracking-wider text-[10px] block mb-1">{hp.key}</span>
                  <span className="text-sky-400 font-bold text-sm block mb-1">{hp.val}</span>
                  <span className="text-slate-400 font-sans text-[11px] block">{hp.desc}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Dataset Partitioning Table */}
          <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl mb-16">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Dataset Statistics & Stratified Split</h2>
                <p className="text-xs text-slate-400">Guaranteed 4-class distribution across all evaluation splits</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {datasetStats.map((ds) => (
                <div key={ds.key} className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 font-mono text-xs">
                  <span className="text-slate-500 uppercase tracking-wider text-[10px] block mb-1">{ds.key}</span>
                  <span className="text-emerald-400 font-bold text-sm block mb-1">{ds.val}</span>
                  <span className="text-slate-400 font-sans text-[11px] block">{ds.detail}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Hardware & Safeguards Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
              <div className="flex items-center gap-3 mb-6">
                <Cpu className="w-5 h-5 text-violet-400" />
                <h3 className="text-lg font-bold text-white">Execution Environment</h3>
              </div>

              <div className="space-y-3 text-xs font-mono">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                  <span className="text-slate-400">Framework:</span>
                  <span className="text-sky-400 font-bold">PyTorch 2.0+ (Python 3.14)</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                  <span className="text-slate-400">Hardware Accelerator:</span>
                  <span className="text-violet-400 font-bold">CPU / Auto CUDA Detection</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                  <span className="text-slate-400">Random Seed:</span>
                  <span className="text-emerald-400 font-bold">42 (Deterministic)</span>
                </div>
              </div>
            </div>

            <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
              <div className="flex items-center gap-3 mb-6">
                <ShieldCheck className="w-5 h-5 text-amber-400" />
                <h3 className="text-lg font-bold text-white">Training Safety Mechanisms</h3>
              </div>

              <div className="space-y-3 text-xs font-sans">
                <div className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="text-slate-300"><strong>NaN-Loss Detection:</strong> Automatically catches NaN/Inf gradients and skips corrupted optimization steps.</span>
                </div>
                <div className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="text-slate-300"><strong>Gradient Explosion Alarm:</strong> Logs CRITICAL alert if total gradient norm exceeds 100.0 threshold.</span>
                </div>
                <div className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span className="text-slate-300"><strong>Class-Weight Loss:</strong> Applies inverse-frequency weights <code className="text-sky-400">[11.98, 0.48, 0.80, 1.71]</code> to CrossEntropyLoss.</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
