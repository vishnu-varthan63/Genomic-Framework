"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { motion } from "framer-motion";
import { ShieldAlert, Target, Sparkles, CheckCircle2, FlaskConical, Award, ArrowUpRight } from "lucide-react";
import Link from "next/link";
import { site } from "@/lib/content";

export default function AboutPage() {
  const objectives = [
    {
      title: "Hybrid Genomic Encoder",
      desc: "Develop a specialized sequence representation combining HyenaDNA long-range modeling with k-mer tokenization (k=6, vocab=4096) for sub-quadratic context processing.",
    },
    {
      title: "Scientific Reliability & Metrics",
      desc: "Rigorously validate model performance using MCC (0.7821), Cohen's Kappa (0.7795), and Balanced Accuracy to prevent false confidence from class imbalance.",
    },
    {
      title: "End-to-End MLOps Pipeline",
      desc: "Build an automated workflow from raw FASTA quality check through mutation matrix extraction, stratified splitting, to formatted evaluation report generation.",
    },
    {
      title: "Reproducibility & Leakage Guard",
      desc: "Ensure zero data leakage between train/val/test splits with explicit path deduplication, fixed random seeds, and class-balanced samplers.",
    },
  ];

  const highlights = [
    "86.03% Multi-class Test Accuracy",
    "89.47% Macro-averaged ROC-AUC",
    "Sub-quadratic Hyena sequence representation",
    "Stratified 4-class dataset partitioning",
    "Automatic NaN-loss skip guards & explosion alarms",
    "CSV, JSON, Excel & HTML reporting outputs",
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          {/* Header */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              Project Motivation & Objectives
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              About AvianFlu Genomic Intelligence
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Addressing the urgent global challenge of rapid avian influenza pathogenicity classification using state-of-the-art genomic deep learning.
            </p>
          </div>

          {/* Problem Statement Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="p-8 sm:p-10 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl mb-16 relative overflow-hidden"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
                <ShieldAlert className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-bold text-white">Problem Statement</h2>
            </div>
            
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed mb-6">
              Avian influenza viruses (e.g., H5N1, H5N8, H7N9) represent severe pandemic risks to poultry and human populations worldwide. Traditional laboratory virulence assays require time-consuming wet-lab animal trials or specialized sequencing protocols. Computational sequence classification is frequently limited by standard Transformer quadratic memory costs (<code className="text-sky-400 font-mono">O(N²)</code>), dataset class imbalance, and data leakage across splits.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-slate-800/80">
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="text-amber-400 font-bold text-xs uppercase mb-1">Challenge 1</div>
                <div className="text-white font-semibold text-sm">Long Sequence Context</div>
                <div className="text-slate-400 text-xs mt-1">Full genome segments require sub-quadratic sequence modeling.</div>
              </div>
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="text-sky-400 font-bold text-xs uppercase mb-1">Challenge 2</div>
                <div className="text-white font-semibold text-sm">Class Imbalance</div>
                <div className="text-slate-400 text-xs mt-1">Pathogenic samples are rarer than benign surveillance samples.</div>
              </div>
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
                <div className="text-violet-400 font-bold text-xs uppercase mb-1">Challenge 3</div>
                <div className="text-white font-semibold text-sm">Data Leakage Prevention</div>
                <div className="text-slate-400 text-xs mt-1">Random splitting fails to guarantee class representation in test sets.</div>
              </div>
            </div>
          </motion.div>

          {/* Research Objectives */}
          <div className="mb-16">
            <h2 className="text-2xl font-bold text-white mb-8 text-center">Core Research Objectives</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {objectives.map((obj, idx) => (
                <div
                  key={obj.title}
                  className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 hover:border-sky-500/30 transition-all flex items-start gap-4"
                >
                  <div className="p-2.5 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20 font-mono text-xs font-bold shrink-0">
                    0{idx + 1}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white mb-2">{obj.title}</h3>
                    <p className="text-slate-400 text-xs leading-relaxed">{obj.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Highlights & Achievements */}
          <div className="p-8 sm:p-10 rounded-3xl bg-gradient-to-r from-sky-950/30 via-slate-900 to-violet-950/30 border border-sky-500/20 mb-16">
            <div className="flex items-center gap-3 mb-6">
              <Award className="w-6 h-6 text-sky-400" />
              <h2 className="text-2xl font-bold text-white">Project Innovations & Highlights</h2>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {highlights.map((item) => (
                <div key={item} className="flex items-center gap-3 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                  <span className="text-xs font-semibold text-slate-200">{item}</span>
                </div>
              ))}
            </div>
          </div>

          {/* CTA Banner */}
          <div className="text-center">
            <Link
              href="/pipeline"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-bold text-sm shadow-lg shadow-sky-500/25 hover:shadow-sky-500/40 hover:scale-[1.02] transition-all"
            >
              <span>Explore Interactive AI Pipeline</span>
              <ArrowUpRight className="w-4 h-4" />
            </Link>
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
