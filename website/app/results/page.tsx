"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import MetricsChart from "@/components/charts/MetricsChart";
import PerformanceRadarChart from "@/components/charts/RadarChart";
import ModelPerformanceDashboard from "@/components/dashboard/ModelPerformanceDashboard";
import ProjectStatisticsCards from "@/components/dashboard/ProjectStatisticsCards";
import { motion } from "framer-motion";
import { CheckCircle2, TrendingUp, BarChart3, ShieldAlert, Award, FileSpreadsheet, Download, Activity, Cpu } from "lucide-react";
import { metrics, classes } from "@/lib/content";
import Link from "next/link";

export default function ResultsPage() {
  const metricCards = [
    { label: "Test Accuracy", val: `${metrics.accuracy}%`, desc: "Overall multi-class accuracy", color: "text-emerald-400 border-emerald-500/30" },
    { label: "Balanced Accuracy", val: `${metrics.balancedAccuracy}%`, desc: "Average recall across all 4 classes", color: "text-sky-400 border-sky-500/30" },
    { label: "Macro F1-Score", val: `${metrics.macroF1}%`, desc: "Unweighted mean class F1", color: "text-violet-400 border-violet-500/30" },
    { label: "Weighted F1-Score", val: `${metrics.weightedF1}%`, desc: "Sample-weighted class F1", color: "text-purple-400 border-purple-500/30" },
    { label: "Macro ROC-AUC", val: `${metrics.rocAuc}%`, desc: "One-vs-Rest area under ROC", color: "text-amber-400 border-amber-500/30" },
    { label: "Macro PR-AUC", val: `${metrics.prAuc}%`, desc: "Area under Precision-Recall curve", color: "text-orange-400 border-orange-500/30" },
    { label: "Matthews Corr (MCC)", val: metrics.mcc.toString(), desc: "Balanced measure (-1 to +1)", color: "text-rose-400 border-rose-500/30" },
    { label: "Cohen's Kappa", val: metrics.cohensKappa.toString(), desc: "Inter-annotator agreement", color: "text-cyan-400 border-cyan-500/30" },
  ];

  const perClassResults = classes.map((c) => ({
    name: c.name,
    precision: c.precision.toFixed(4),
    recall: c.recall.toFixed(4),
    f1: c.f1.toFixed(4),
    rocAuc: c.rocAuc.toFixed(4),
    prAuc: c.prAuc.toFixed(4),
    support: c.testSupportSeqs,
    color: c.color
  }));

  const confusionMatrix = [
    [50, 89, 52, 0],
    [0, 1251, 106, 43],
    [0, 52, 715, 33],
    [0, 0, 15, 385],
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              Verified Benchmark Evaluation
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Test Performance & Benchmark Dashboard
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Empirical evaluation on 2,791 hold-out test sequence windows using the PyTorch <code className="text-sky-400 font-mono">AdvancedEvaluator</code> module.
            </p>
          </div>

          {/* Interactive Benchmark Dashboard Component */}
          <div className="mb-16">
            <ModelPerformanceDashboard />
          </div>

          {/* 8 Metric Cards Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
            {metricCards.map((mc) => (
              <div key={mc.label} className={`p-6 rounded-2xl bg-slate-900 border ${mc.color} transition-all`}>
                <span className="text-[10px] font-mono font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  {mc.label}
                </span>
                <div className="text-3xl font-extrabold text-white mb-1">{mc.val}</div>
                <div className="text-[11px] text-slate-400">{mc.desc}</div>
              </div>
            ))}
          </div>

          {/* Interactive Recharts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            <div className="p-6 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
              <h3 className="text-lg font-bold text-white mb-2">Metric Comparison Overview</h3>
              <p className="text-xs text-slate-400 mb-4">Visualizing the 8 target metrics evaluated on test set</p>
              <MetricsChart />
            </div>

            <div className="p-6 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
              <h3 className="text-lg font-bold text-white mb-2">Performance Radar Map</h3>
              <p className="text-xs text-slate-400 mb-4">Multi-axial profile showing balanced performance across metrics</p>
              <PerformanceRadarChart />
            </div>
          </div>

          {/* Per-Class Classification Report Table */}
          <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl mb-16">
            <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
              <div>
                <h3 className="text-xl font-bold text-white">Per-Class Classification Performance</h3>
                <p className="text-xs text-slate-400 mt-1">Detailed precision, recall, F1, ROC-AUC, and PR-AUC breakdown across all 4 pathogenicity tiers</p>
              </div>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 font-semibold">
                Test Support: 2,791 Sequence Windows
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="py-3.5 px-4 font-bold">Class Name</th>
                    <th className="py-3.5 px-4 font-bold">Precision</th>
                    <th className="py-3.5 px-4 font-bold">Recall</th>
                    <th className="py-3.5 px-4 font-bold">F1-Score</th>
                    <th className="py-3.5 px-4 font-bold">ROC-AUC</th>
                    <th className="py-3.5 px-4 font-bold">PR-AUC</th>
                    <th className="py-3.5 px-4 font-bold">Test Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {perClassResults.map((row) => (
                    <tr key={row.name} className="hover:bg-slate-850/50 transition-colors">
                      <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: row.color }} />
                        <span>{row.name}</span>
                      </td>
                      <td className="py-3.5 px-4 text-sky-400 font-semibold">{row.precision}</td>
                      <td className="py-3.5 px-4 text-violet-400 font-semibold">{row.recall}</td>
                      <td className="py-3.5 px-4 text-emerald-400 font-bold">{row.f1}</td>
                      <td className="py-3.5 px-4 text-amber-400 font-semibold">{row.rocAuc}</td>
                      <td className="py-3.5 px-4 text-orange-400 font-semibold">{row.prAuc}</td>
                      <td className="py-3.5 px-4 text-slate-400">{row.support.toLocaleString()} seqs</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Confusion Matrix Visual Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
            <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
              <h3 className="text-xl font-bold text-white mb-2">Confusion Matrix (2,791 Sequences)</h3>
              <p className="text-xs text-slate-400 mb-6">4x4 ground truth vs predicted class matrix</p>

              <div className="grid grid-cols-4 gap-2 font-mono text-center text-xs">
                {confusionMatrix.map((row, rIdx) =>
                  row.map((val, cIdx) => (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      className={`p-4 rounded-xl border flex flex-col items-center justify-center font-bold ${
                        rIdx === cIdx
                          ? val > 0
                            ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-300"
                            : "bg-slate-950 border-slate-800 text-slate-600"
                          : val > 0
                          ? "bg-rose-500/20 border-rose-500/40 text-rose-300"
                          : "bg-slate-950 border-slate-800 text-slate-600"
                      }`}
                    >
                      <span className="text-lg">{val}</span>
                      <span className="text-[9px] font-normal opacity-70 mt-0.5">
                        {rIdx === cIdx ? "TP" : val > 0 ? "FP/FN" : "TN"}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Inference Performance Summary */}
            <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl flex flex-col justify-between">
              <div>
                <h3 className="text-xl font-bold text-white mb-2">Inference Latency & MLOps</h3>
                <p className="text-xs text-slate-400 mb-6">Benchmark execution statistics from `evaluate_advanced.py` run</p>

                <div className="space-y-4 text-xs font-mono">
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-slate-400">Total Evaluation Time:</span>
                    <span className="text-sky-400 font-bold">{metrics.totalInferenceDurationSec} seconds</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-slate-400">Per-Sample Latency:</span>
                    <span className="text-emerald-400 font-bold">{metrics.inferenceTime}</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-slate-400">Test Sequence Windows:</span>
                    <span className="text-violet-400 font-bold">{metrics.testFastaSequences.toLocaleString()} sequences</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex justify-between">
                    <span className="text-slate-400">Checkpoint Loaded:</span>
                    <span className="text-amber-400 font-bold">output_advanced/checkpoints/best_model.pt</span>
                  </div>
                </div>
              </div>

              <div className="pt-6 mt-6 border-t border-slate-800 flex items-center justify-between">
                <Link
                  href="/reports"
                  className="inline-flex items-center gap-2 text-xs font-bold text-sky-400 hover:text-sky-300"
                >
                  <FileSpreadsheet className="w-4 h-4" />
                  <span>View & Download CSV / HTML Reports</span>
                </Link>
              </div>
            </div>
          </div>

          {/* Project Statistics Cards Component */}
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-white text-center mb-6">Overall Framework Statistics</h3>
            <ProjectStatisticsCards />
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
