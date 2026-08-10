"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import {
  Download,
  FileCode,
  FileText,
  Brain,
  BookOpen,
  BarChart3,
  Presentation,
  CheckCircle2,
  Terminal,
  Settings,
  Layers,
  Database
} from "lucide-react";
import { site } from "@/lib/content";

export default function DownloadsPage() {
  const downloads = [
    {
      title: "Full Research Paper (SDD 100+ Pages)",
      category: "Research Paper",
      icon: Presentation,
      size: "PDF Document",
      badge: "Available (Markdown/PDF)",
      desc: "Complete Software Design Document covering system requirements, module architecture, sequence diagrams, and mathematical formulation.",
      link: site.github,
      external: true,
      color: "border-sky-500/30 text-sky-400 bg-sky-500/10",
    },
    {
      title: "GitHub Source Code Repository",
      category: "Codebase",
      icon: FileCode,
      size: "~1.2 MB",
      badge: "Available (MIT)",
      desc: "Complete PyTorch codebase including preprocessor, engine, advanced trainer, evaluator, audit tools, and configs.",
      link: site.github,
      external: true,
      color: "border-emerald-500/30 text-emerald-400 bg-emerald-500/10",
    },
    {
      title: "Pre-trained Model Checkpoint (best_model.pt)",
      category: "Weights",
      icon: Brain,
      size: "122.8 MB",
      badge: "State Dict",
      desc: "PyTorch model weights for the 10.21M parameter GenomicClassificationEngine trained for 50 epochs with 86.03% test accuracy.",
      link: site.github,
      external: true,
      color: "border-violet-500/30 text-violet-400 bg-violet-500/10",
    },
    {
      title: "Evaluation & Metrics Deliverables",
      category: "Reports",
      icon: FileText,
      size: "~350 KB",
      badge: "JSON / CSV / HTML",
      desc: "Full `evaluation_metrics.json`, `classification_report.csv`, and stand-alone `evaluation_report.html` executive summary.",
      link: "/reports",
      external: false,
      color: "border-amber-500/30 text-amber-400 bg-amber-500/10",
    },
    {
      title: "Sample Predictions Dataset (predictions.csv)",
      category: "Predictions",
      icon: Terminal,
      size: "2,791 Predictions",
      badge: "CSV / JSON",
      desc: "Sample-by-sample ground truth vs predicted class probabilities and confidence scores across 2,791 sequence windows.",
      link: "/reports",
      external: false,
      color: "border-rose-500/30 text-rose-400 bg-rose-500/10",
    },
    {
      title: "Configuration YAML (train_config.yaml)",
      category: "Configuration",
      icon: Settings,
      size: "~2.5 KB",
      badge: "YAML Config",
      desc: "Production hyperparameter configuration specifying k-mer window size (k=6), warmup epochs, and stratified dataset settings.",
      link: "/documentation",
      external: false,
      color: "border-cyan-500/30 text-cyan-400 bg-cyan-500/10",
    },
    {
      title: "Training Logs & Execution History",
      category: "Logs",
      icon: Terminal,
      size: "Execution Log",
      badge: "Audit Log",
      desc: "Complete 50-epoch training stdout log recording loss decay, val_f1 improvements, learning rate schedules, and step counts.",
      link: "/documentation",
      external: false,
      color: "border-blue-500/30 text-blue-400 bg-blue-500/10",
    },
    {
      title: "Dataset Manifest & Audit Report",
      category: "Dataset Metadata",
      icon: Database,
      size: "JSON Manifest",
      badge: "Audit Report",
      desc: "File path duplicate verification logs, zero-leakage split matrices, and class distribution manifests for the 72 FASTA files.",
      link: "/dataset",
      external: false,
      color: "border-indigo-500/30 text-indigo-400 bg-indigo-500/10",
    },
    {
      title: "Architecture & Tensor Diagrams",
      category: "Diagrams",
      icon: Layers,
      size: "PNG Graphics",
      badge: "High-Res Visuals",
      desc: "Diagrams depicting sub-module parameter counts, tensor flow paths, and Smith-Waterman mutation matrix dimensions.",
      link: "/architecture",
      external: false,
      color: "border-teal-500/30 text-teal-400 bg-teal-500/10",
    },
    {
      title: "Technical Researcher Manual",
      category: "Documentation",
      icon: BookOpen,
      size: "Online Manual",
      badge: "Interactive Guide",
      desc: "Developer guide covering installation, PyTorch DataLoader construction, GPU/CPU auto-detection, and evaluation execution.",
      link: "/documentation",
      external: false,
      color: "border-yellow-500/30 text-yellow-400 bg-yellow-500/10",
    },
    {
      title: "Project Slide Presentation Deck",
      category: "Presentation",
      icon: Presentation,
      size: "PPTX / PDF",
      badge: "Slide Deck",
      desc: "Executive slide presentation summarizing research motivation, model architecture, benchmark metrics, and biosecurity impact.",
      link: site.github,
      external: true,
      color: "border-pink-500/30 text-pink-400 bg-pink-500/10",
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
              Download Center & Deliverables
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Research Deliverables Hub
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Access pre-trained weights, research papers, evaluation reports, predictions datasets, configuration YAMLs, and technical manuals.
            </p>
          </div>

          {/* Downloads Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {downloads.map((item) => {
              const Icon = item.icon;
              return (
                <div key={item.title} className="p-6 rounded-3xl bg-slate-900/60 border border-slate-800 hover:border-sky-500/30 transition-all flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className={`p-3 rounded-xl border ${item.color}`}>
                        <Icon className="w-5 h-5" />
                      </div>
                      <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                        {item.badge}
                      </span>
                    </div>

                    <h3 className="text-base font-bold text-white mb-1">{item.title}</h3>
                    <span className="text-[11px] font-mono text-slate-400 block mb-3">{item.size}</span>
                    <p className="text-xs text-slate-400 leading-relaxed mb-6">{item.desc}</p>
                  </div>

                  <a
                    href={item.link}
                    target={item.external ? "_blank" : "_self"}
                    rel={item.external ? "noopener noreferrer" : ""}
                    className="w-full py-3 rounded-xl bg-slate-950 border border-slate-800 hover:border-sky-500/40 text-sky-400 font-bold text-xs flex items-center justify-center gap-2 transition-all group"
                  >
                    <Download className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
                    <span>Access {item.category}</span>
                  </a>
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
