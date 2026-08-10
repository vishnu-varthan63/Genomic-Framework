"use client";

import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { metrics } from "@/lib/content";

export default function PerformanceRadarChart() {
  const data = [
    { subject: "Accuracy", A: metrics.accuracy, fullMark: 100 },
    { subject: "Bal. Acc", A: metrics.balancedAccuracy, fullMark: 100 },
    { subject: "Macro F1", A: metrics.macroF1, fullMark: 100 },
    { subject: "Weighted F1", A: metrics.weightedF1, fullMark: 100 },
    { subject: "ROC-AUC", A: metrics.rocAuc, fullMark: 100 },
    { subject: "PR-AUC", A: metrics.prAuc, fullMark: 100 },
    { subject: "MCC", A: Number((metrics.mcc * 100).toFixed(1)), fullMark: 100 },
    { subject: "Kappa", A: Number((metrics.cohensKappa * 100).toFixed(1)), fullMark: 100 },
  ];

  return (
    <div className="w-full h-80 pt-2">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="subject" stroke="#94a3b8" fontSize={11} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#475569" fontSize={9} />
          <Radar
            name="Model Performance"
            dataKey="A"
            stroke="#0ea5e9"
            fill="#0ea5e9"
            fillOpacity={0.3}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              borderColor: "#334155",
              borderRadius: "0.75rem",
              color: "#f8fafc",
              fontSize: "12px",
            }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
