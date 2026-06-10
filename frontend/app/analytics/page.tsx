"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { VARIANT_COLOR, VARIANT_LABEL, VARIANTS, type Variant } from "@/lib/api";
import { getResults, type ResultsResponse } from "@/lib/results";
import { GlossaryTerm } from "@/components/InfoTip";

const METRIC_ORDER = [
  "ndcg_cut_10",
  "ndcg_cut_100",
  "recip_rank",
  "map",
  "recall_10",
  "recall_100",
  "P_10",
];
const METRIC_LABEL: Record<string, string> = {
  ndcg_cut_10: "NDCG@10",
  ndcg_cut_100: "NDCG@100",
  recip_rank: "MRR",
  map: "MAP",
  recall_10: "Recall@10",
  recall_100: "Recall@100",
  P_10: "P@10",
};

export default function AnalyticsPage() {
  const [data, setData] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getResults().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error)
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
        No results yet — run <code className="font-mono">productrank eval</code> to populate the
        dashboard. ({error})
      </div>
    );
  if (!data) return <div className="text-sm text-slate-400">Loading…</div>;

  const present = VARIANTS.filter((v) => data.aggregate[v]);
  const chartData = present.map((v) => ({
    variant: VARIANT_LABEL[v],
    key: v,
    ndcg: data.aggregate[v]?.ndcg_cut_10 ?? 0,
  }));

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          Measured over {data.num_queries} {data.split} queries (top-{data.top_k}), scored against
          human relevance judgments with <GlossaryTerm term="NDCG@10">NDCG</GlossaryTerm> and
          related metrics. Higher is better.
        </p>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-slate-700">
          <GlossaryTerm term="NDCG@10" align="left" /> by strategy
        </h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="variant" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, "auto"]} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v: number) => v.toFixed(4)} />
              <Bar dataKey="ndcg" radius={[6, 6, 0, 0]}>
                {chartData.map((d) => (
                  <Cell key={d.key} fill={VARIANT_COLOR[d.key as Variant]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white">
        <h2 className="rounded-t-xl border-b border-slate-100 px-4 py-2 text-sm font-semibold text-slate-700">
          Full metrics table
        </h2>
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
            <tr>
              <th className="px-4 py-2">Metric</th>
              {present.map((v) => (
                <th key={v} className="px-4 py-2">
                  {VARIANT_LABEL[v]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRIC_ORDER.map((m) => {
              const best = Math.max(...present.map((v) => data.aggregate[v]?.[m] ?? 0));
              return (
                <tr key={m} className="border-t border-slate-100">
                  <td className="px-4 py-2 font-medium">
                    <GlossaryTerm term={METRIC_LABEL[m] ?? m} align="left" />
                  </td>
                  {present.map((v) => {
                    const val = data.aggregate[v]?.[m] ?? 0;
                    const isBest = val === best && best > 0;
                    return (
                      <td
                        key={v}
                        className={`px-4 py-2 font-mono ${isBest ? "font-bold text-emerald-600" : ""}`}
                      >
                        {val.toFixed(4)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
            <tr className="border-t border-slate-200 bg-slate-50">
              <td className="rounded-bl-xl px-4 py-2 font-medium text-slate-500">eval wall (s)</td>
              {present.map((v, i) => (
                <td
                  key={v}
                  className={`px-4 py-2 font-mono text-xs text-slate-500 ${
                    i === present.length - 1 ? "rounded-br-xl" : ""
                  }`}
                >
                  {data.wall_seconds[v]?.toFixed(0) ?? "—"}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
