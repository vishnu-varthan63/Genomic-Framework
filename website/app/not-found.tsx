"use client";

import React from "react";
import Link from "next/link";
import { Dna, ArrowLeft } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-4">
      <div className="text-center max-w-md space-y-6">
        <div className="w-16 h-16 rounded-3xl bg-sky-500/10 border border-sky-500/20 text-sky-400 flex items-center justify-center mx-auto">
          <Dna className="w-8 h-8 animate-spin-slow" />
        </div>

        <h1 className="text-6xl font-extrabold text-white font-mono">404</h1>
        <h2 className="text-xl font-bold text-slate-200">Page Not Found</h2>
        <p className="text-xs text-slate-400 leading-relaxed">
          The requested research route does not exist or has been moved.
        </p>

        <Link
          href="/"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-sky-500 text-slate-950 font-bold text-xs shadow-lg shadow-sky-500/25 hover:bg-sky-400 transition-all"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Return to Research Home</span>
        </Link>
      </div>
    </div>
  );
}
