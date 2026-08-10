"use client";

import React from "react";
import { projectStats } from "@/lib/content";
import { motion } from "framer-motion";
import { Code2, Layers, Cpu, Database, FileText, BarChart2, ShieldCheck, BookOpen } from "lucide-react";

export default function ProjectStatisticsCards() {
  const statsList = [
    { label: "Core Modules", val: projectStats.totalModules.toString(), sub: "Engine, Trainer, Preprocessor, Evaluator", icon: Layers, color: "text-sky-400 border-sky-500/20 bg-sky-500/10" },
    { label: "Python Source Files", val: projectStats.totalPythonFiles.toString(), sub: "Modular framework files", icon: Code2, color: "text-violet-400 border-violet-500/20 bg-violet-500/10" },
    { label: "Pathogenicity Classes", val: projectStats.totalClasses.toString(), sub: "Benign, LB, LP, Pathogenic", icon: ShieldCheck, color: "text-emerald-400 border-emerald-500/20 bg-emerald-500/10" },
    { label: "Lines of Code (LOC)", val: projectStats.totalLinesOfCode, sub: "Clean modular Python codebase", icon: Code2, color: "text-amber-400 border-amber-500/20 bg-amber-500/10" },
    { label: "Model Parameters", val: projectStats.totalParameters, sub: "Hyena + Transformer weights", icon: Cpu, color: "text-cyan-400 border-cyan-500/20 bg-cyan-500/10" },
    { label: "Augmented FASTA Size", val: projectStats.datasetStorageSize, sub: "72 FASTA files / 14,391 seqs", icon: Database, color: "text-purple-400 border-purple-500/20 bg-purple-500/10" },
    { label: "Train Partition", val: projectStats.trainSamples, sub: "70% Stratified Split", icon: Database, color: "text-blue-400 border-blue-500/20 bg-blue-500/10" },
    { label: "Val Partition", val: projectStats.valSamples, sub: "15% Stratified Split", icon: Database, color: "text-indigo-400 border-indigo-500/20 bg-indigo-500/10" },
    { label: "Test Partition", val: projectStats.testSamples, sub: "15% Stratified Split", icon: Database, color: "text-teal-400 border-teal-500/20 bg-teal-500/10" },
    { label: "SDD Paper Document", val: `${projectStats.sddPaperPages} Pages`, sub: "Full Software Design Document", icon: BookOpen, color: "text-rose-400 border-rose-500/20 bg-rose-500/10" },
    { label: "Figures & Plots", val: projectStats.figuresCount.toString(), sub: "ROC, PR, Confusion Matrix PNGs", icon: BarChart2, color: "text-orange-400 border-orange-500/20 bg-orange-500/10" },
    { label: "Tables & Diagrams", val: `${projectStats.tablesCount + projectStats.diagramsCount}`, sub: "Per-class tables & flowcharts", icon: FileText, color: "text-yellow-400 border-yellow-500/20 bg-yellow-500/10" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {statsList.map((st, idx) => {
        const Icon = st.icon;
        return (
          <motion.div
            key={st.label}
            initial={{ opacity: 0, y: 15 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: idx * 0.04 }}
            className="p-5 rounded-2xl bg-slate-900 border border-slate-800 hover:border-sky-500/30 transition-all flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className={`p-2.5 rounded-xl border ${st.color}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span className="text-[10px] font-mono text-slate-500 uppercase">VERIFIED</span>
              </div>
              <span className="text-2xl font-bold text-white block mb-0.5">{st.val}</span>
              <span className="text-xs font-semibold text-slate-300 block mb-1">{st.label}</span>
            </div>
            <span className="text-[11px] text-slate-500 font-sans block">{st.sub}</span>
          </motion.div>
        );
      })}
    </div>
  );
}
