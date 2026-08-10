"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Dna, ArrowRight, Activity, Cpu, ShieldAlert, Sparkles, BookOpen, GitBranch } from "lucide-react";
import { site, metrics } from "@/lib/content";

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-32 pb-20 overflow-hidden bg-slate-950">
      {/* Background glow effects */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-gradient-to-tr from-sky-500/15 to-violet-600/15 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 left-1/4 w-[400px] h-[400px] bg-sky-500/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute top-1/2 right-1/4 w-[400px] h-[400px] bg-violet-600/10 rounded-full blur-[100px] pointer-events-none" />
      
      {/* Grid Pattern */}
      <div className="absolute inset-0 bg-grid opacity-30 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        <div className="text-center max-w-4xl mx-auto space-y-8">
          
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/80 border border-slate-800 backdrop-blur-md shadow-lg"
          >
            <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-semibold text-slate-300">
              Validated Research Benchmark
            </span>
            <span className="text-slate-600">•</span>
            <span className="text-xs font-bold text-sky-400">
              Accuracy {metrics.accuracy}%
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white leading-[1.1]"
          >
            Avian Influenza <br />
            <span className="gradient-text">Genomic Intelligence</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-base sm:text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed"
          >
            A production-grade PyTorch framework combining <strong className="text-slate-200">HyenaDNA long-range sequence modeling</strong> and <strong className="text-slate-200">k-mer tokenization</strong> to classify avian influenza pathogenicity with 86.03% accuracy.
          </motion.p>

          {/* Badges / Tech Tags */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-wrap items-center justify-center gap-2 pt-2"
          >
            {["HyenaDNA", "k-mer Tokenization (k=6)", "Transformer Attention", "Stratified 4-Class", "PyTorch 2.0"].map((tag) => (
              <span
                key={tag}
                className="px-3 py-1 rounded-lg text-xs font-medium bg-slate-900/90 text-slate-300 border border-slate-800/80 shadow-sm"
              >
                {tag}
              </span>
            ))}
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4"
          >
            <Link
              href="/results"
              className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-r from-sky-500 to-violet-600 text-white font-semibold text-sm shadow-lg shadow-sky-500/25 hover:shadow-sky-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2 group"
            >
              <Sparkles className="w-4 h-4" />
              <span>Explore Benchmark Results</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>

            <Link
              href="/pipeline"
              className="w-full sm:w-auto px-8 py-4 rounded-xl bg-slate-900 border border-slate-800 text-slate-200 font-semibold text-sm hover:border-slate-700 hover:bg-slate-850 hover:text-white transition-all flex items-center justify-center gap-2"
            >
              <GitBranch className="w-4 h-4 text-sky-400" />
              <span>Interactive Pipeline</span>
            </Link>
          </motion.div>

          {/* Floating Metric Preview Grid */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-12 text-left"
          >
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md hover:border-sky-500/30 transition-colors group">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">Accuracy</span>
                <Activity className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
              </div>
              <div className="text-2xl font-bold text-white">{metrics.accuracy}%</div>
              <div className="text-[11px] text-slate-400 mt-1">Multi-class test accuracy</div>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md hover:border-sky-500/30 transition-colors group">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">ROC-AUC</span>
                <ShieldAlert className="w-4 h-4 text-sky-400 group-hover:scale-110 transition-transform" />
              </div>
              <div className="text-2xl font-bold text-white">{metrics.rocAuc}%</div>
              <div className="text-[11px] text-slate-400 mt-1">Macro-averaged OVR AUC</div>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md hover:border-sky-500/30 transition-colors group">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">MCC Score</span>
                <Cpu className="w-4 h-4 text-violet-400 group-hover:scale-110 transition-transform" />
              </div>
              <div className="text-2xl font-bold text-white">{metrics.mcc}</div>
              <div className="text-[11px] text-slate-400 mt-1">Matthews correlation</div>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md hover:border-sky-500/30 transition-colors group">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">Params</span>
                <Dna className="w-4 h-4 text-amber-400 group-hover:scale-110 transition-transform" />
              </div>
              <div className="text-2xl font-bold text-white">{metrics.totalParams}</div>
              <div className="text-[11px] text-slate-400 mt-1">Engine architecture</div>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  );
}
