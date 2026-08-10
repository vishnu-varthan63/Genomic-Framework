"use client";

import React, { useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { BookOpen, Copy, Check, Dna, Layers, ShieldCheck, Quote, ArrowUpRight } from "lucide-react";
import { site } from "@/lib/content";

export default function ResearchPage() {
  const [copiedBib, setCopiedBib] = useState(false);

  const bibtex = `@article{senthilkumar2026avianflu,
  title={Avian Influenza Genomic Sequence Classification Using HyenaDNA and Transformer Feature Fusion},
  author={Senthilkumar, Vishnu Varthan},
  journal={Genomic Framework Research AI},
  year={2026},
  volume={1},
  pages={1--15},
  publisher={GitHub Research}
}`;

  const copyBibtex = () => {
    navigator.clipboard.writeText(bibtex);
    setCopiedBib(true);
    setTimeout(() => setCopiedBib(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              Research & Scientific Scope
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Genomic AI Research Methodology
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Detailed exploration of the scientific principles, related baseline architectures, and citation instructions for this project.
            </p>
          </div>

          {/* Abstract / Executive Summary */}
          <div className="p-8 sm:p-10 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl mb-16 space-y-4">
            <h2 className="text-2xl font-bold text-white">Abstract</h2>
            <p className="text-slate-300 text-sm leading-relaxed">
              Rapid classification of avian influenza viral genomes is vital for early pandemic detection and livestock biosecurity. We present a 10.21-million-parameter deep learning engine combining sub-quadratic HyenaDNA long-context sequence modeling with 4-head Transformer attention layers and pairwise Smith-Waterman mutation matrix extraction. Evaluated across 4 pathogenicity classes (Benign, Likely Benign, Likely Pathogenic, Pathogenic) using a stratified 70/15/15 partition, our framework achieves 86.03% test accuracy, 89.47% macro ROC-AUC, 0.7821 Matthews Correlation Coefficient (MCC), and 0.7795 Cohen's Kappa score.
            </p>
          </div>

          {/* Related Work & Baselines */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
            <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-3">
              <span className="text-xs font-mono font-bold text-sky-400">BASELINE 01</span>
              <h3 className="text-lg font-bold text-white">DNABERT & DNABERT-2</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Uses 6-mer tokenization and BERT attention to model local nucleotide contexts. Limited by quadratic attention scaling on long genome segments.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-3">
              <span className="text-xs font-mono font-bold text-violet-400">BASELINE 02</span>
              <h3 className="text-lg font-bold text-white">HyenaDNA</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Introduces sub-quadratic implicit long convolutions, enabling single-nucleotide resolution modeling up to 1 million tokens.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-3">
              <span className="text-xs font-mono font-bold text-emerald-400">OUR INNOVATION</span>
              <h3 className="text-lg font-bold text-white">AGFX Hybrid Engine</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Combines Hyena long-context implicit filters with Transformer attention and explicit 32x64 mutation matrix alignment fusion.
              </p>
            </div>
          </div>

          {/* BibTeX Citation Card */}
          <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl relative space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Quote className="w-5 h-5 text-sky-400" />
                <h3 className="text-xl font-bold text-white">BibTeX Citation</h3>
              </div>
              <button
                onClick={copyBibtex}
                className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs font-semibold text-slate-300 hover:text-white"
              >
                {copiedBib ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                <span>{copiedBib ? "Copied!" : "Copy BibTeX"}</span>
              </button>
            </div>

            <div className="rounded-2xl bg-slate-950 border border-slate-800 p-5 font-mono text-xs text-sky-300 overflow-x-auto">
              <pre><code>{bibtex}</code></pre>
            </div>
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
