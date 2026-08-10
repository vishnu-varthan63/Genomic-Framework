"use client";

import React from "react";
import { motion } from "framer-motion";
import { Dna, Layers, ShieldCheck, GitBranch, FlaskConical, Cpu, ArrowUpRight } from "lucide-react";
import Link from "next/link";

export default function FeaturesSection() {
  const features = [
    {
      icon: Dna,
      title: "k-mer Tokenization Engine",
      description: "Extracts overlapping 6-mers across 4,096 vocabulary tokens matching biological hexamer patterns in genomic sequence data.",
      tag: "Preprocessing",
      color: "from-sky-500/20 to-sky-600/5 text-sky-400 border-sky-500/20",
    },
    {
      icon: Layers,
      title: "HyenaDNA + Transformer Hybrid",
      description: "Combines Hyena sub-quadratic long-context sequence representation with multi-head self-attention for local and global context.",
      tag: "Architecture",
      color: "from-violet-500/20 to-violet-600/5 text-violet-400 border-violet-500/20",
    },
    {
      icon: ShieldCheck,
      title: "Stratified 4-Class Partitioning",
      description: "Guarantees representation of all 4 pathogenicity tiers (Benign, Likely Benign, Likely Pathogenic, Pathogenic) across train, val, and test splits.",
      tag: "Data Integrity",
      color: "from-emerald-500/20 to-emerald-600/5 text-emerald-400 border-emerald-500/20",
    },
    {
      icon: GitBranch,
      title: "Mutation Vector Fusion",
      description: "Processes pairwise alignment and edit distance features into 32x64 mutation tensors fused directly into the classification head.",
      tag: "Feature Fusion",
      color: "from-amber-500/20 to-amber-600/5 text-amber-400 border-amber-500/20",
    },
    {
      icon: FlaskConical,
      title: "Scientific Rigor & Validation",
      description: "Evaluated using MCC (0.7821), Cohen's Kappa (0.7795), Balanced Accuracy (75.29%), and Weighted F1 (84.89%) to eliminate class bias.",
      tag: "Evaluation",
      color: "from-rose-500/20 to-rose-600/5 text-rose-400 border-rose-500/20",
    },
    {
      icon: Cpu,
      title: "Automated MLOps & Audit",
      description: "Integrated duplicate path detection, zero-leakage split verification, linear warmup cosine scheduling, and automated HTML/Markdown report generation.",
      tag: "Production",
      color: "from-cyan-500/20 to-cyan-600/5 text-cyan-400 border-cyan-500/20",
    },
  ];

  return (
    <section className="py-24 bg-slate-950 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="text-center max-w-3xl mx-auto space-y-4 mb-16">
          <h2 className="text-xs font-bold uppercase tracking-widest text-sky-400">
            Core Architecture & Innovations
          </h2>
          <h3 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Engineered for High-Confidence Pathogenicity Prediction
          </h3>
          <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
            Every module in the pipeline is designed to overcome standard genomic ML limitations such as sequence length caps, class imbalance, and data leakage.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((item, idx) => {
            const Icon = item.icon;
            return (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                className="group relative p-8 rounded-2xl bg-slate-900/50 border border-slate-800/80 hover:border-sky-500/40 transition-all duration-300 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <div className={`p-3.5 rounded-xl bg-gradient-to-br ${item.color} border shadow-lg`}>
                      <Icon className="w-6 h-6" />
                    </div>
                    <span className="text-[11px] font-semibold px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                      {item.tag}
                    </span>
                  </div>

                  <h4 className="text-lg font-bold text-white mb-2 group-hover:text-sky-400 transition-colors">
                    {item.title}
                  </h4>

                  <p className="text-slate-400 text-xs sm:text-sm leading-relaxed">
                    {item.description}
                  </p>
                </div>

                <div className="pt-6 mt-6 border-t border-slate-800/60 flex items-center justify-between text-xs font-semibold text-sky-400 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span>Learn more in documentation</span>
                  <ArrowUpRight className="w-4 h-4" />
                </div>
              </motion.div>
            );
          })}
        </div>

        <div className="mt-12 text-center">
          <Link
            href="/architecture"
            className="inline-flex items-center gap-2 text-xs font-bold text-sky-400 hover:text-sky-300 uppercase tracking-wider group"
          >
            <span>Explore Full System Architecture</span>
            <ArrowUpRight className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
          </Link>
        </div>

      </div>
    </section>
  );
}
