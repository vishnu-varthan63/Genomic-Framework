"use client";

import React, { useState } from "react";
import { metrics } from "@/lib/content";
import {
  Activity,
  Award,
  CheckCircle2,
  Cpu,
  Clock,
  Database,
  Layers,
  ShieldCheck,
  TrendingUp,
  BarChart3,
  Sparkles
} from "lucide-react";

export default function ModelPerformanceDashboard() {
  const [activeTab, setActiveTab] = useState<"classification" | "timing" | "dataset">("classification");

  const classificationMetrics = [
    { label: "Overall Accuracy", value: `${metrics.accuracy}%`, detail: "2,401 / 2,791 correct predictions", highlight: true },
    { label: "Balanced Accuracy", value: `${metrics.balancedAccuracy}%`, detail: "Unweighted mean class recall", highlight: true },
    { label: "Macro F1-Score", value: `${metrics.macroF1}%`, detail: "Mean unweighted F1", highlight: true },
    { label: "Weighted F1-Score", value: `${metrics.weightedF1}%`, detail: "Sample-weighted F1", highlight: true },
    { label: "Macro ROC-AUC", value: `${metrics.rocAuc}%`, detail: "One-vs-Rest ROC AUC", highlight: true },
    { label: "Macro PR-AUC", value: `${metrics.prAuc}%`, detail: "Area under Precision-Recall", highlight: true },
    { label: "Matthews Corr (MCC)", value: metrics.mcc.toString(), detail: "Balanced correlation index (-1 to +1)", highlight: false },
    { label: "Cohen's Kappa", value: metrics.cohensKappa.toString(), detail: "Inter-rater agreement score", highlight: false },
  ];

  const timingMetrics = [
    { label: "Total Inference Duration", value: `${metrics.totalInferenceDurationSec}s`, desc: "Execution time for 2,791 test sequences" },
    { label: "Per-Sample Latency", value: metrics.inferenceTime, desc: "Average CPU prediction time per sequence" },
    { label: "Training Epochs", value: `${metrics.trainingEpochs} Epochs`, desc: "AdamW + Cosine Warmup training run" },
    { label: "Best Model Selection Epoch", value: "Epoch 42", desc: "Selected via highest val_f1 score" },
  ];

  const datasetMetrics = [
    { label: "Test Sequences Evaluated", value: metrics.testFastaSequences.toLocaleString(), desc: "Hold-out test sequences" },
    { label: "Total Sequence Records", value: metrics.totalBalancedFastaSequences.toLocaleString(), desc: "Full active dataset sequences" },
    { label: "Augmented FASTA Files", value: `${metrics.balancedFastaFiles} Files`, desc: "Stratified 4-class set" },
    { label: "Raw FASTA Files", value: `${metrics.rawFastaFiles} Files`, desc: "Surveillance isolates" },
  ];

  return (
    <div className="p-8 rounded-3xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl shadow-2xl space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 text-xs font-semibold mb-2">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Interactive Benchmark Dashboard</span>
          </div>
          <h2 className="text-2xl font-extrabold text-white">Model Performance Dashboard</h2>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center gap-1 bg-slate-950 p-1.5 rounded-2xl border border-slate-800">
          <button
            onClick={() => setActiveTab("classification")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all ${
              activeTab === "classification"
                ? "bg-sky-500 text-slate-950"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Classification Scores
          </button>
          <button
            onClick={() => setActiveTab("timing")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all ${
              activeTab === "timing"
                ? "bg-sky-500 text-slate-950"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Timing & Latency
          </button>
          <button
            onClick={() => setActiveTab("dataset")}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all ${
              activeTab === "dataset"
                ? "bg-sky-500 text-slate-950"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Dataset Volume
          </button>
        </div>
      </div>

      {/* Content View */}
      {activeTab === "classification" && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {classificationMetrics.map((m) => (
            <div key={m.label} className="p-5 rounded-2xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold uppercase text-slate-500 block mb-1">{m.label}</span>
                <span className="text-2xl font-extrabold text-white block mb-1">{m.value}</span>
              </div>
              <span className="text-[11px] text-slate-400 font-sans">{m.detail}</span>
            </div>
          ))}
        </div>
      )}

      {activeTab === "timing" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {timingMetrics.map((t) => (
            <div key={t.label} className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
              <Clock className="w-5 h-5 text-sky-400" />
              <span className="text-xs text-slate-400 block font-semibold">{t.label}</span>
              <span className="text-2xl font-bold text-white font-mono block">{t.value}</span>
              <span className="text-[11px] text-slate-500 block">{t.desc}</span>
            </div>
          ))}
        </div>
      )}

      {activeTab === "dataset" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {datasetMetrics.map((d) => (
            <div key={d.label} className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
              <Database className="w-5 h-5 text-emerald-400" />
              <span className="text-xs text-slate-400 block font-semibold">{d.label}</span>
              <span className="text-2xl font-bold text-white font-mono block">{d.value}</span>
              <span className="text-[11px] text-slate-500 block">{d.desc}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
