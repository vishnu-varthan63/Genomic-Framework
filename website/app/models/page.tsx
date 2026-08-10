"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import { motion } from "framer-motion";
import { Cpu, Dna, Layers, GitBranch, ShieldCheck, Code2, ArrowRight } from "lucide-react";
import { metrics } from "@/lib/content";

export default function ModelsPage() {
  const components = [
    {
      id: "engine",
      name: "GenomicClassificationEngine",
      category: "Main Engine",
      params: "10.21M",
      input: "token_ids [B, 128], mutation_vectors [B, 32, 64]",
      output: "logits [B, 4] -> Softmax [B, 4]",
      desc: "Top-level PyTorch nn.Module encapsulating Hyena sequence backbone, pairwise mutation aligner, 4-head Transformer attention layers, and linear classification projection head.",
      code: `class GenomicClassificationEngine(nn.Module):
    def __init__(self, seq_len=128, vocab_size=4096, hyena_dim=256, hidden_dim=512, num_classes=4):
        super().__init__()
        self.tokenizer_emb = nn.Embedding(vocab_size, hyena_dim)
        self.hyena_block = HyenaBlock(dim=hyena_dim, seq_len=seq_len)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )`
    },
    {
      id: "hyena",
      name: "HyenaDNA Implicit Long Convolution",
      category: "Sequence Backbone",
      params: "4.82M",
      input: "embedded_seq [B, 128, 256]",
      output: "hyena_features [B, 128, 256]",
      desc: "Replaces standard O(N²) attention with sub-quadratic implicit long convolution filter for processing long-context genomic sequences efficiently.",
      code: `class HyenaBlock(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.filter = ExponentialModulatedFFTFilter(seq_len=seq_len)
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        y = self.filter(k) * v
        return self.out_proj(y)`
    },
    {
      id: "mutation",
      name: "Pairwise Mutation Alignment Processor",
      category: "Feature Extractor",
      params: "0.45M",
      input: "Raw FASTA Sequence + Class Reference FASTA",
      output: "mutation_vector [B, 32, 64]",
      desc: "Performs pairwise Needleman-Wunsch / Smith-Waterman alignment to extract position-wise single nucleotide substitutions and insertion/deletion events.",
      code: `class MutationProcessor(nn.Module):
    def extract_mutations(self, seq, ref_seq, max_mutations=32):
        alignments = pairwise2.align.globalms(seq, ref_seq, 2, -1, -0.5, -0.1)
        top_align = alignments[0]
        mutation_matrix = compute_mutation_matrix(top_align, max_mutations=32, dim=64)
        return torch.tensor(mutation_matrix, dtype=torch.float32)`
    },
    {
      id: "transformer",
      name: "Multi-Head Self-Attention Transformer",
      category: "Contextual Refinement",
      params: "3.14M",
      input: "hyena_features [B, 128, 256]",
      output: "transformer_features [B, 128, 512]",
      desc: "Refines Hyena embeddings through 2 TransformerEncoder layers using 4 attention heads to capture localized interactions between k-mer motifs.",
      code: `encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, 
    nhead=4, 
    dim_feedforward=2048, 
    dropout=0.2, 
    activation="relu"
)`
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow pt-32 pb-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Title */}
          <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
            <span className="text-xs font-bold uppercase tracking-widest text-sky-400">
              Model Components & Source Code
            </span>
            <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight">
              AI Model Specifications
            </h1>
            <p className="text-slate-400 text-base leading-relaxed">
              In-depth technical breakdown of every sub-module in the 10.21M parameter <code className="text-sky-400 font-mono">GenomicClassificationEngine</code> architecture.
            </p>
          </div>

          {/* Component Modules List */}
          <div className="space-y-12 mb-16">
            {components.map((comp) => (
              <div key={comp.id} className="p-8 rounded-3xl bg-slate-900/60 border border-slate-800 backdrop-blur-xl">
                <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800">
                  <div>
                    <span className="text-xs font-mono font-bold text-sky-400 uppercase tracking-wider block mb-1">
                      {comp.category}
                    </span>
                    <h2 className="text-2xl font-bold text-white">{comp.name}</h2>
                  </div>
                  <span className="text-xs font-mono font-bold text-slate-300 bg-slate-950 px-3 py-1.5 rounded-full border border-slate-800">
                    {comp.params} Parameters
                  </span>
                </div>

                <p className="text-slate-300 text-sm leading-relaxed mb-6">{comp.desc}</p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 font-mono text-xs">
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 text-[10px] uppercase block mb-1">Input Specification</span>
                    <span className="text-slate-200">{comp.input}</span>
                  </div>
                  <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                    <span className="text-slate-500 text-[10px] uppercase block mb-1">Output Specification</span>
                    <span className="text-emerald-400 font-bold">{comp.output}</span>
                  </div>
                </div>

                {/* PyTorch Code Snippet */}
                <div className="rounded-2xl bg-slate-950 border border-slate-800 p-5 font-mono text-xs overflow-x-auto text-sky-300">
                  <div className="flex items-center gap-2 text-slate-500 text-[11px] mb-3 pb-2 border-b border-slate-900">
                    <Code2 className="w-3.5 h-3.5 text-sky-400" />
                    <span>PyTorch Implementation Snippet</span>
                  </div>
                  <pre className="text-slate-200 leading-relaxed font-mono"><code>{comp.code}</code></pre>
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
