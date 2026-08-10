"use client";

import React, { useState } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { Terminal, Copy, Check, BookOpen, Settings, Play, ShieldAlert, HelpCircle } from "lucide-react";
import { site } from "@/lib/content";

export default function DocumentationPage() {
  const [copiedSection, setCopiedSection] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("quickstart");

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(label);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  const sections = [
    { id: "quickstart", name: "Quick Start & Installation", icon: Terminal },
    { id: "training", name: "Training Pipeline", icon: Play },
    { id: "evaluation", name: "Evaluation & Testing", icon: BookOpen },
    { id: "config", name: "Configuration Guide", icon: Settings },
    { id: "troubleshooting", name: "Troubleshooting & FAQ", icon: ShieldAlert },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              Technical Documentation & User Guide
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              Developer & Researcher Manual
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              Step-by-step documentation for training models, evaluating checkpoints, customizing configurations, and troubleshooting common issues.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            
            {/* Sidebar Navigation */}
            <div className="space-y-2 lg:col-span-1">
              <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 px-3 block mb-2">
                  Documentation Index
                </span>
                {sections.map((s) => {
                  const Icon = s.icon;
                  const isActive = activeTab === s.id;
                  return (
                    <button
                      key={s.id}
                      onClick={() => setActiveTab(s.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                        isActive
                          ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                          : "text-slate-400 hover:bg-slate-850 hover:text-slate-200"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{s.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Main Content Area */}
            <div className="lg:col-span-3 space-y-8">

              {activeTab === "quickstart" && (
                <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 space-y-6">
                  <h2 className="text-2xl font-bold text-white">Installation & Environment Setup</h2>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Clone the repository, create a virtual environment, and install the required dependencies:
                  </p>

                  <div className="relative rounded-2xl bg-slate-950 border border-slate-800 p-5 font-mono text-xs text-sky-300">
                    <button
                      onClick={() => copyToClipboard(`git clone ${site.github}\ncd DNA_AvianFluenza\npip install -r requirements.txt`, "git-clone")}
                      className="absolute top-4 right-4 p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white"
                    >
                      {copiedSection === "git-clone" ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <pre><code>{`git clone ${site.github}
cd DNA_AvianFluenza
pip install -r requirements.txt`}</code></pre>
                  </div>

                  <h3 className="text-lg font-bold text-white pt-4">Dependencies</h3>
                  <ul className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono text-slate-300">
                    <li className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">PyTorch 2.0+</li>
                    <li className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">TorchMetrics</li>
                    <li className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">scikit-learn</li>
                    <li className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">NumPy & Pandas</li>
                    <li className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">Matplotlib</li>
                    <li className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">openpyxl</li>
                  </ul>
                </div>
              )}

              {activeTab === "training" && (
                <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 space-y-6">
                  <h2 className="text-2xl font-bold text-white">Running Model Training</h2>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    To execute the 50-epoch training pipeline using the stratified dataset split:
                  </p>

                  <div className="relative rounded-2xl bg-slate-950 border border-slate-800 p-5 font-mono text-xs text-sky-300">
                    <button
                      onClick={() => copyToClipboard(`python train_advanced.py --config configs/train_config.yaml`, "train-cmd")}
                      className="absolute top-4 right-4 p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white"
                    >
                      {copiedSection === "train-cmd" ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <pre><code>{`python train_advanced.py --config configs/train_config.yaml`}</code></pre>
                  </div>
                </div>
              )}

              {activeTab === "evaluation" && (
                <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 space-y-6">
                  <h2 className="text-2xl font-bold text-white">Running Model Testing & Evaluation</h2>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Run complete evaluation on the test set using a trained model checkpoint:
                  </p>

                  <div className="relative rounded-2xl bg-slate-950 border border-slate-800 p-5 font-mono text-xs text-sky-300">
                    <button
                      onClick={() => copyToClipboard(`python evaluate_advanced.py --checkpoint "output_advanced\\checkpoints\\best_model.pt" --dataset-root "dataset_augmented" --output-dir "eval_results_fixed" --config "configs\\train_config.yaml" --batch-size 16`, "eval-cmd")}
                      className="absolute top-4 right-4 p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white"
                    >
                      {copiedSection === "eval-cmd" ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <pre className="whitespace-pre-wrap"><code>{`python evaluate_advanced.py --checkpoint "output_advanced\\checkpoints\\best_model.pt" --dataset-root "dataset_augmented" --output-dir "eval_results_fixed" --config "configs\\train_config.yaml" --batch-size 16`}</code></pre>
                  </div>
                </div>
              )}

              {activeTab === "config" && (
                <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 space-y-6">
                  <h2 className="text-2xl font-bold text-white">Configuration YAML Structure</h2>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Key settings in <code className="text-sky-400 font-mono">configs/train_config.yaml</code>:
                  </p>

                  <div className="rounded-2xl bg-slate-950 border border-slate-800 p-5 font-mono text-xs text-slate-300 overflow-x-auto">
                    <pre><code>{`data:
  batch_size: 16
  stratified: true     # Ensures 4-class representation in test
  class_balance: true

model:
  k_mer_size: 6        # Matches 4096 vocab size (4^6)
  seq_len: 128
  dropout: 0.2

optimization:
  learning_rate: 0.0001
  nan_loss_detection: true

scheduler:
  name: "cosine"
  warmup_epochs: 5`}</code></pre>
                  </div>
                </div>
              )}

              {activeTab === "troubleshooting" && (
                <div className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 space-y-6">
                  <h2 className="text-2xl font-bold text-white">Troubleshooting & FAQ</h2>
                  
                  <div className="space-y-4 text-xs">
                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                      <h4 className="font-bold text-sky-400 mb-1">Q: Why did test set previously have 0 Pathogenic samples?</h4>
                      <p className="text-slate-300">A: Unstratified global random shuffling sliced small classes unevenly. Fixed by enabling `stratified: true` in config.</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                      <h4 className="font-bold text-sky-400 mb-1">Q: How to fix openpyxl export warning?</h4>
                      <p className="text-slate-300">A: Run <code className="text-emerald-400">pip install openpyxl</code> to enable Excel `.xlsx` prediction exports.</p>
                    </div>
                  </div>
                </div>
              )}

            </div>

          </div>

        </div>
      </main>

      <Footer />
    </div>
  );
}
