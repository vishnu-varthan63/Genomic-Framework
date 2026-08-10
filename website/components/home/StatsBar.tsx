"use client";

import React from "react";
import { motion } from "framer-motion";
import { metrics } from "@/lib/content";

export default function StatsBar() {
  const stats = [
    { label: "Test Accuracy", value: `${metrics.accuracy}%`, sub: "Multi-class 4-way" },
    { label: "Balanced Accuracy", value: `${metrics.balancedAccuracy}%`, sub: "Adjusted for imbalance" },
    { label: "Weighted F1-Score", value: `${metrics.weightedF1}%`, sub: "Class-weighted F1" },
    { label: "Macro ROC-AUC", value: `${metrics.rocAuc}%`, sub: "One-vs-Rest AUC" },
    { label: "Matthews Correlation", value: metrics.mcc.toString(), sub: "MCC Score (0-1)" },
    { label: "Inference Latency", value: metrics.inferenceTime, sub: "CPU Batch Inference" },
  ];

  return (
    <section className="py-12 bg-slate-900/80 border-y border-slate-800/80 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 text-center divide-y md:divide-y-0 md:divide-x divide-slate-800/80">
          {stats.map((stat, idx) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 15 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: idx * 0.08 }}
              className="pt-4 md:pt-0 px-2"
            >
              <div className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
                {stat.value}
              </div>
              <div className="text-xs font-semibold text-sky-400 mt-1 uppercase tracking-wider">
                {stat.label}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                {stat.sub}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
