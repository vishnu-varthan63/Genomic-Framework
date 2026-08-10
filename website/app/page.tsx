"use client";

import React from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import Hero from "@/components/home/Hero";
import StatsBar from "@/components/home/StatsBar";
import FeaturesSection from "@/components/home/FeaturesSection";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, CheckCircle2, FileText, Database, ShieldAlert, Sparkles } from "lucide-react";
import { metrics, classes } from "@/lib/content";

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Navbar />

      <main className="flex-grow">
        <Hero />
        <StatsBar />
        <FeaturesSection />

        {/* Pathogenicity Class Breakdown Teaser */}
        <section className="py-20 bg-slate-900/40 border-t border-slate-800/80 relative">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto space-y-4 mb-14">
              <h2 className="text-xs font-bold uppercase tracking-widest text-sky-400">
                4-Class Taxonomy
              </h2>
              <h3 className="text-3xl font-extrabold text-white">
                Multi-Class Avian Influenza Pathogenicity Classification
              </h3>
              <p className="text-slate-400 text-sm">
                Each genomic sequence is evaluated and mapped to one of four clinical pathogenicity tiers.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {classes.map((c) => (
                <div
                  key={c.name}
                  className="p-6 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-slate-700 transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-xs font-mono font-bold text-slate-500">
                        CLASS 0{c.index}
                      </span>
                      <div
                        className="w-3 h-3 rounded-full shadow-lg"
                        style={{ backgroundColor: c.color }}
                      />
                    </div>
                    <h4 className="text-lg font-bold text-white mb-2">{c.name}</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">{c.description}</p>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-800/60 flex items-center justify-between text-xs font-semibold text-slate-300">
                    <span>Index {c.index}</span>
                    <span className="text-sky-400">Target Label</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Documentation & Quick Start CTA */}
        <section className="py-20 bg-slate-950 relative overflow-hidden">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="p-10 rounded-3xl bg-gradient-to-r from-sky-950/40 via-slate-900 to-violet-950/40 border border-sky-500/20 backdrop-blur-xl relative overflow-hidden flex flex-col lg:flex-row items-center justify-between gap-8">
              
              <div className="space-y-4 max-w-2xl text-left">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 text-xs font-semibold">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Ready for Reproduction</span>
                </div>
                <h3 className="text-2xl sm:text-3xl font-extrabold text-white">
                  Reproduce the Evaluation Pipeline in Minutes
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Run standard PyTorch inference, inspect CSV/JSON prediction outputs, or execute custom dataset audits using our CLI tools.
                </p>
                <div className="flex flex-wrap gap-4 text-xs font-medium text-slate-300">
                  <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4 text-emerald-400" /> CLI Support</span>
                  <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4 text-emerald-400" /> Pre-trained Weights</span>
                  <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4 text-emerald-400" /> HTML Reports</span>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center gap-4 w-full lg:w-auto">
                <Link
                  href="/documentation"
                  className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs shadow-lg shadow-sky-500/25 transition-all text-center"
                >
                  View Documentation
                </Link>
                <Link
                  href="/reports"
                  className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-slate-900 border border-slate-700 hover:border-slate-600 text-white font-semibold text-xs transition-all text-center"
                >
                  Read Evaluation Report
                </Link>
              </div>

            </div>
          </div>
        </section>

      </main>

      <Footer />
    </div>
  );
}
