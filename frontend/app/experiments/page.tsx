"use client";

import { useEffect, useRef, useState } from "react";
import {
  getExperiment,
  startExperiment,
  VARIANTS,
  VARIANT_LABEL,
  type Experiment,
  type Variant,
} from "@/lib/api";
import { GlossaryTerm } from "@/components/InfoTip";

const METRIC_LABEL: Record<string, string> = {
  ndcg_cut_10: "NDCG@10",
  ndcg_cut_100: "NDCG@100",
  recip_rank: "MRR",
  map: "MAP",
  recall_10: "Recall@10",
  recall_100: "Recall@100",
  P_10: "P@10",
};
const METRIC_ORDER = [
  "ndcg_cut_10",
  "ndcg_cut_100",
  "recip_rank",
  "map",
  "recall_10",
  "recall_100",
  "P_10",
];

export default function ExperimentsPage() {
  const [variantA, setVariantA] = useState<Variant>("bm25");
  const [variantB, setVariantB] = useState<Variant>("hybrid_rerank");
  const [size, setSize] = useState(100);
  const [exp, setExp] = useState<Experiment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  async function run() {
    setError(null);
    try {
      const job = await startExperiment(variantA, variantB, size);
      setExp(job);
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const latest = await getExperiment(job.id);
          setExp(latest);
          if (latest.status === "completed" || latest.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch {
          /* keep polling */
        }
      }, 1500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to start experiment");
    }
  }

  const sigByMetric = new Map((exp?.significance ?? []).map((s) => [s.metric, s]));

  return (
    <div className="flex flex-col gap-5">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Experiment Runner</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          Pick two strategies and race them over many questions at once. We then run a{" "}
          <GlossaryTerm term="significance">significance test</GlossaryTerm> and mark a win
          with ✱ only when it&apos;s real — not luck on these particular questions.
        </p>
      </section>

      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-slate-200 bg-white p-4">
        <VariantSelect label="Variant A" value={variantA} onChange={setVariantA} />
        <VariantSelect label="Variant B" value={variantB} onChange={setVariantB} />
        <label className="flex flex-col gap-1 text-xs font-medium text-slate-600">
          Query set size
          <input
            type="number"
            min={2}
            max={648}
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
            className="w-28 rounded-md border border-slate-300 px-2 py-1.5 text-sm"
          />
        </label>
        <button
          onClick={run}
          disabled={exp?.status === "running" || exp?.status === "pending"}
          className="rounded-lg bg-indigo-600 px-5 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          Run A/B
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {exp && (exp.status === "running" || exp.status === "pending") && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="mb-2 text-sm text-slate-600">
            Running {exp.status === "pending" ? "(starting…)" : `· ${Math.round((exp.progress || 0) * 100)}%`}
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full bg-indigo-500 transition-all"
              style={{ width: `${Math.max(5, (exp.progress || 0) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {exp?.status === "error" && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          Experiment failed: {exp.error}
        </div>
      )}

      {exp?.status === "completed" && exp.metrics_a && exp.metrics_b && (
        <div className="rounded-xl border border-slate-200 bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="rounded-tl-xl px-4 py-2">Metric</th>
                <th className="px-4 py-2">{VARIANT_LABEL[exp.variant_a!]}</th>
                <th className="px-4 py-2">{VARIANT_LABEL[exp.variant_b!]}</th>
                <th className="px-4 py-2">Δ (B−A)</th>
                <th className="px-4 py-2"><GlossaryTerm term="p-value" align="right" /></th>
                <th className="rounded-tr-xl px-4 py-2"><GlossaryTerm term="95% CI" align="right" /></th>
              </tr>
            </thead>
            <tbody>
              {METRIC_ORDER.map((m, idx) => {
                const a = exp.metrics_a![m];
                const b = exp.metrics_b![m];
                const sig = sigByMetric.get(m);
                const isLast = idx === METRIC_ORDER.length - 1;
                return (
                  <tr key={m} className="border-t border-slate-100">
                    <td className={`px-4 py-2 font-medium ${isLast ? "rounded-bl-xl" : ""}`}>
                      <GlossaryTerm term={METRIC_LABEL[m] ?? m} align="left" />
                    </td>
                    <td className="px-4 py-2 font-mono">{a?.toFixed(4)}</td>
                    <td className="px-4 py-2 font-mono">{b?.toFixed(4)}</td>
                    <td className="px-4 py-2 font-mono">
                      {sig ? (
                        <span className={sig.mean_diff >= 0 ? "text-emerald-600" : "text-red-600"}>
                          {sig.mean_diff >= 0 ? "+" : ""}
                          {sig.mean_diff.toFixed(4)}
                          {sig.significant ? " ✱" : ""}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="px-4 py-2 font-mono">{sig ? sig.p_value.toFixed(4) : "—"}</td>
                    <td
                      className={`px-4 py-2 font-mono text-xs text-slate-500 ${
                        isLast ? "rounded-br-xl" : ""
                      }`}
                    >
                      {sig ? `[${sig.ci_low.toFixed(3)}, ${sig.ci_high.toFixed(3)}]` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="px-4 py-2 text-xs text-slate-400">
            ✱ = paired t-test p &lt; 0.05 and bootstrap 95% CI excludes 0. Over{" "}
            {exp.query_set_size} queries.
          </p>
        </div>
      )}
    </div>
  );
}

function VariantSelect({
  label,
  value,
  onChange,
}: {
  label: string;
  value: Variant;
  onChange: (v: Variant) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-slate-600">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as Variant)}
        className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
      >
        {VARIANTS.map((v) => (
          <option key={v} value={v}>
            {VARIANT_LABEL[v]}
          </option>
        ))}
      </select>
    </label>
  );
}
