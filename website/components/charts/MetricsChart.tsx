"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from "recharts";
import { metrics } from "@/lib/content";

export default function MetricsChart() {
  const data = [
    { name: "Accuracy", score: metrics.accuracy, color: "#22c55e" },
    { name: "Bal. Acc", score: metrics.balancedAccuracy, color: "#0ea5e9" },
    { name: "Macro F1", score: metrics.macroF1, color: "#8b5cf6" },
    { name: "Weighted F1", score: metrics.weightedF1, color: "#a855f7" },
    { name: "ROC-AUC", score: metrics.rocAuc, color: "#38bdf8" },
    { name: "PR-AUC", score: metrics.prAuc, color: "#f97316" },
    { name: "MCC (x100)", score: Number((metrics.mcc * 100).toFixed(1)), color: "#eab308" },
    { name: "Kappa (x100)", score: Number((metrics.cohensKappa * 100).toFixed(1)), color: "#ec4899" },
  ];

  return (
    <div className="w-full h-80 pt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 20, left: -10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
          <XAxis
            dataKey="name"
            stroke="#94a3b8"
            fontSize={11}
            tickLine={false}
            interval={0}
          />
          <YAxis
            domain={[0, 100]}
            stroke="#94a3b8"
            fontSize={11}
            tickLine={false}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              borderColor: "#334155",
              borderRadius: "0.75rem",
              color: "#f8fafc",
              fontSize: "12px",
            }}
            formatter={(value: any) => [`${value}%`, "Score"]}
          />
          <Bar dataKey="score" radius={[6, 6, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
