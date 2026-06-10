"use client";

import { useState } from "react";
import { EXAMPLE_QUERIES, searchAll, type SearchResponse } from "@/lib/api";
import { computeDivergence } from "@/lib/divergence";
import { ResultColumn } from "@/components/ResultColumn";
import { HowItWorks } from "@/components/HowItWorks";
import { GlossaryTerm } from "@/components/InfoTip";

const TOP_K = 10;

export default function SearchComparisonPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResponse[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredDoc, setHoveredDoc] = useState<string | null>(null);

  async function run(q: string) {
    const text = q.trim();
    if (!text) return;
    setQuery(text);
    setLoading(true);
    setError(null);
    try {
      setResults(await searchAll(text, TOP_K));
    } catch (e) {
      setError(e instanceof Error ? e.message : "search failed");
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  const divergence = results ? computeDivergence(results, TOP_K) : null;

  return (
    <div className="flex flex-col gap-5">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Search Comparison</h1>
        <p className="mt-1 text-sm text-slate-500">
          One question, four search strategies, side by side. Coloured cards are{" "}
          <GlossaryTerm term="divergence">documents that disagree</GlossaryTerm> — they rank
          high in one strategy and low in another.
        </p>
      </section>

      <HowItWorks />

      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(query);
        }}
        className="flex gap-2"
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question…"
          className="flex-1 rounded-lg border border-slate-300 px-4 py-2.5 text-sm shadow-sm focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-500">Try one:</span>
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            onClick={() => run(q)}
            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-700"
          >
            {q}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error} — is the API running on :8000?
        </div>
      )}

      {results && divergence && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
          {results.map((r, i) => (
            <ResultColumn
              key={r.variant}
              result={r}
              divergence={divergence}
              hoveredDoc={hoveredDoc}
              onHover={setHoveredDoc}
              tipAlign={i < 2 ? "left" : "right"}
            />
          ))}
        </div>
      )}

      {!results && !loading && (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white py-16 text-center text-sm text-slate-400">
          Pick a “Try one” question above, or type your own, to compare the four strategies
          side by side.
        </div>
      )}
    </div>
  );
}
