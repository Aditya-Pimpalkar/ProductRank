"use client";

import { useState } from "react";
import { VARIANT_COLOR } from "@/lib/api";

// A compact, professional explainer of the four-stage pipeline. Tied visually to the
// result columns below via the same variant colours. No gradients or emoji — plain,
// clean, and dismissible.
const STEPS: { dot: string; label: string; body: string }[] = [
  { dot: VARIANT_COLOR.bm25, label: "Keyword", body: "Finds documents sharing words with the query." },
  { dot: VARIANT_COLOR.dense, label: "Meaning", body: "Finds documents with similar meaning, not just words." },
  { dot: VARIANT_COLOR.hybrid, label: "Fusion", body: "Merges both lists by rank position (RRF)." },
  { dot: VARIANT_COLOR.hybrid_rerank, label: "Rerank", body: "A cross-encoder re-reads and reorders the top candidates." },
];

export function HowItWorks() {
  const [open, setOpen] = useState(true);

  return (
    <section className="rounded-xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between px-4 py-2.5">
        <h2 className="text-sm font-semibold text-slate-700">How it works</h2>
        <button
          onClick={() => setOpen((o) => !o)}
          className="text-xs text-slate-400 hover:text-slate-700"
        >
          {open ? "Hide" : "Show"}
        </button>
      </div>
      {open && (
        <ol className="grid grid-cols-1 gap-px border-t border-slate-100 bg-slate-100 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s, i) => (
            <li key={s.label} className="bg-white px-4 py-3">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: s.dot }} />
                <span className="text-xs font-semibold text-slate-700">
                  {i + 1}. {s.label}
                </span>
              </div>
              <p className="mt-1 text-xs leading-snug text-slate-500">{s.body}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
