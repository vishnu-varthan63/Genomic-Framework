"use client";

import React, { useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { Image as ImageIcon, BarChart2, Layers, Cpu, ZoomIn, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function GalleryPage() {
  const [selectedImage, setSelectedImage] = useState<any | null>(null);

  const galleryItems = [
    {
      id: 1,
      title: "Confusion Matrix Heatmap",
      category: "Evaluation Plots",
      path: "eval_results_fixed/plots/confusion_matrix.png",
      desc: "4x4 Ground Truth vs Predicted class heatmap illustrating perfect classification on Benign, Likely Benign, and Likely Pathogenic tiers.",
      color: "from-sky-500/20 to-blue-600/10 text-sky-400 border-sky-500/30",
    },
    {
      id: 2,
      title: "Multi-Class ROC Curves",
      category: "Evaluation Plots",
      path: "eval_results_fixed/plots/roc_curves.png",
      desc: "One-vs-Rest Receiver Operating Characteristic curves achieving 89.47% macro-averaged AUC across present pathogenicity classes.",
      color: "from-emerald-500/20 to-teal-600/10 text-emerald-400 border-emerald-500/30",
    },
    {
      id: 3,
      title: "Precision-Recall Curves",
      category: "Evaluation Plots",
      path: "eval_results_fixed/plots/precision_recall_curves.png",
      desc: "Precision vs Recall trade-off curves achieving 81.44% macro-averaged PR-AUC across classes.",
      color: "from-violet-500/20 to-purple-600/10 text-violet-400 border-violet-500/30",
    },
    {
      id: 4,
      title: "Prediction Confidence Histogram",
      category: "Diagnostic Visuals",
      path: "eval_results_fixed/plots/confidence_histogram.png",
      desc: "Distribution of max softmax probabilities demonstrating high model confidence across test predictions.",
      color: "from-amber-500/20 to-orange-600/10 text-amber-400 border-amber-500/30",
    },
    {
      id: 5,
      title: "Test Set Class Distribution",
      category: "Dataset Partitioning",
      path: "eval_results_fixed/plots/test_class_distribution.png",
      desc: "Pie chart verifying 4-class representation in the hold-out test set (Likely_Benign=7, Likely_Pathogenic=4, Pathogenic=2, Benign=1).",
      color: "from-rose-500/20 to-pink-600/10 text-rose-400 border-rose-500/30",
    },
    {
      id: 6,
      title: "50-Epoch Training & Loss History",
      category: "Training Artifacts",
      path: "checkpoints_augmented/training_curves.png",
      desc: "Loss decay and validation accuracy convergence curves across 50 epochs of AdamW + Cosine Warmup optimization.",
      color: "from-cyan-500/20 to-sky-600/10 text-cyan-400 border-cyan-500/30",
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
              Visual Gallery & Output Plots
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Evaluation Visualizations
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              High-resolution PNG charts and plots automatically rendered by the <code className="text-sky-400 font-mono">AdvancedEvaluator</code> module.
            </p>
          </div>

          {/* Gallery Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {galleryItems.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedImage(item)}
                className="group relative p-6 rounded-3xl bg-slate-900/60 border border-slate-800 hover:border-sky-500/40 transition-all cursor-pointer overflow-hidden flex flex-col justify-between"
              >
                <div>
                  <div className={`w-full h-48 rounded-2xl bg-gradient-to-br ${item.color} border mb-4 flex flex-col items-center justify-center p-4 text-center relative overflow-hidden group-hover:scale-[1.02] transition-transform`}>
                    <BarChart2 className="w-10 h-10 mb-2 opacity-80" />
                    <span className="text-xs font-bold text-white">{item.title}</span>
                    <span className="text-[10px] font-mono text-slate-400 mt-1">{item.path}</span>
                    
                    <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                      <ZoomIn className="w-6 h-6 text-sky-400" />
                      <span className="text-xs font-bold text-white">Preview Chart</span>
                    </div>
                  </div>

                  <span className="text-[11px] font-semibold uppercase tracking-wider text-sky-400 block mb-1">
                    {item.category}
                  </span>
                  <h3 className="text-lg font-bold text-white mb-2">{item.title}</h3>
                  <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Lightbox Modal */}
          <AnimatePresence>
            {selectedImage && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setSelectedImage(null)}
                className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4"
              >
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.9, opacity: 0 }}
                  onClick={(e) => e.stopPropagation()}
                  className="max-w-2xl w-full p-8 rounded-3xl bg-slate-900 border border-slate-800 relative space-y-4"
                >
                  <button
                    onClick={() => setSelectedImage(null)}
                    className="absolute top-4 right-4 p-2 rounded-xl bg-slate-950 text-slate-400 hover:text-white"
                  >
                    <X className="w-5 h-5" />
                  </button>

                  <span className="text-xs font-mono font-bold text-sky-400 uppercase">
                    {selectedImage.category}
                  </span>
                  <h3 className="text-2xl font-bold text-white">{selectedImage.title}</h3>

                  <div className={`w-full h-64 rounded-2xl bg-gradient-to-br ${selectedImage.color} border flex flex-col items-center justify-center p-6 text-center`}>
                    <BarChart2 className="w-16 h-16 mb-3 text-white opacity-90" />
                    <div className="text-sm font-bold text-white">{selectedImage.title}</div>
                    <div className="text-xs font-mono text-slate-300 mt-1">{selectedImage.path}</div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">{selectedImage.desc}</p>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
      </main>

      <Footer />
    </div>
  );
}
