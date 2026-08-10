"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { FileText, FileSpreadsheet, Download, CheckCircle2, FileCode, ArrowUpRight } from "lucide-react";
import { metrics } from "@/lib/content";
import Link from "next/link";

export default function ReportsPage() {
  const reportsList = [
    {
      title: "Full Evaluation Metrics JSON",
      format: "JSON",
      path: "eval_results_fixed/metrics/evaluation_metrics.json",
      desc: "Raw computer-readable evaluation metrics containing accuracy, balanced_accuracy, weighted_f1, rocAuc, prAuc, mcc, cohensKappa, and per-class precision/recall dictionaries.",
      status: "Generated & Verified",
      color: "border-sky-500/30 text-sky-400 bg-sky-500/10",
    },
    {
      title: "Classification Report CSV",
      format: "CSV",
      path: "eval_results_fixed/metrics/classification_report.csv",
      desc: "Tabular per-class precision, recall, F1-score, and support metrics structured for direct import into pandas or Excel.",
      status: "Generated & Verified",
      color: "border-emerald-500/30 text-emerald-400 bg-emerald-500/10",
    },
    {
      title: "Stand-Alone HTML Executive Report",
      format: "HTML",
      path: "eval_results_fixed/reports/evaluation_report.html",
      desc: "Self-contained HTML report with embedded CSS styling, executive summary tables, per-class breakdown, and full classification text output.",
      status: "Generated & Verified",
      color: "border-violet-500/30 text-violet-400 bg-violet-500/10",
    },
    {
      title: "Markdown Technical Report",
      format: "Markdown",
      path: "eval_results_fixed/reports/evaluation_report.md",
      desc: "GitHub-flavored Markdown report suitable for documentation inclusion, pull request summaries, or notebook rendering.",
      status: "Generated & Verified",
      color: "border-amber-500/30 text-amber-400 bg-amber-500/10",
    },
    {
      title: "Sample Predictions CSV",
      format: "CSV",
      path: "eval_results_fixed/predictions/predictions.csv",
      desc: "Sample-by-sample predictions containing Sample ID, Ground Truth Class, Predicted Class, Prediction Softmax Probabilities, and Confidence Score.",
      status: "Generated & Verified",
      color: "border-rose-500/30 text-rose-400 bg-rose-500/10",
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
              Automated Evaluation Deliverables
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Evaluation & Prediction Reports
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Every evaluation run automatically generates structured machine-readable JSON/CSV files, high-resolution plots, and publication-ready HTML/Markdown reports.
            </p>
          </div>

          {/* Key Summary Box */}
          <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl mb-16">
            <div className="flex items-center justify-between flex-wrap gap-4 mb-6">
              <h2 className="text-xl font-bold text-white">Summary of Latest Evaluation Run</h2>
              <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">
                Status: Exit Code 0 (Passed)
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">Accuracy</span>
                <span className="text-emerald-400 font-bold text-lg">{metrics.accuracy}%</span>
              </div>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">Macro F1</span>
                <span className="text-sky-400 font-bold text-lg">{metrics.macroF1}%</span>
              </div>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">ROC-AUC</span>
                <span className="text-violet-400 font-bold text-lg">{metrics.rocAuc}%</span>
              </div>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block text-[10px] uppercase">MCC</span>
                <span className="text-amber-400 font-bold text-lg">{metrics.mcc}</span>
              </div>
            </div>
          </div>

          {/* Reports Grid */}
          <div className="space-y-6 mb-16">
            <h2 className="text-2xl font-bold text-white mb-6">Generated Evaluation Deliverables</h2>

            {reportsList.map((rpt) => (
              <div key={rpt.title} className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 hover:border-sky-500/30 transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                <div className="space-y-2 max-w-2xl">
                  <div className="flex items-center gap-3">
                    <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded-lg border ${rpt.color}`}>
                      {rpt.format}
                    </span>
                    <h3 className="text-lg font-bold text-white">{rpt.title}</h3>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{rpt.desc}</p>
                  <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1">
                    <span>Path:</span>
                    <span className="text-slate-300">{rpt.path}</span>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-full border border-emerald-500/20">
                    {rpt.status}
                  </span>
                </div>
              </div>
            ))}
          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
