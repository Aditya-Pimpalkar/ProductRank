"use client";

import type { SearchResponse } from "@/lib/api";
import { VARIANT_COLOR, VARIANT_LABEL } from "@/lib/api";
import type { DivergenceInfo } from "@/lib/divergence";
import { rankSummary } from "@/lib/divergence";
import { VARIANT_BLURB, VARIANT_TIP } from "@/lib/copy";
import { Term } from "@/components/InfoTip";

export function ResultColumn({
  result,
  divergence,
  hoveredDoc,
  onHover,
  tipAlign = "left",
}: {
  result: SearchResponse;
  divergence: DivergenceInfo;
  hoveredDoc: string | null;
  onHover: (docId: string | null) => void;
  tipAlign?: "left" | "center" | "right";
}) {
  const color = VARIANT_COLOR[result.variant];

  return (
    <div className="flex min-w-0 flex-col rounded-xl border border-slate-200 bg-white">
      <div className="border-b border-slate-100 px-3 py-2">
        <div className="flex items-center justify-between">
          <Term tip={VARIANT_TIP[result.variant]} align={tipAlign} className="border-none">
            <span className="flex items-center gap-2 text-sm font-semibold">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
              {VARIANT_LABEL[result.variant]}
            </span>
          </Term>
          <span className="text-xs text-slate-500" title="Time taken to answer">
            {result.total_latency_ms.toFixed(0)} ms
            {result.cache_hit ? " · cached" : ""}
          </span>
        </div>
        <p className="mt-0.5 text-[11px] text-slate-400">{VARIANT_BLURB[result.variant]}</p>
      </div>

      <div className="flex flex-wrap gap-1 px-3 py-1.5 text-[10px] text-slate-400">
        {Object.entries(result.stage_latency_ms).map(([stage, ms]) => (
          <span key={stage} className="rounded bg-slate-50 px-1.5 py-0.5">
            {stage} {ms.toFixed(0)}ms
          </span>
        ))}
      </div>

      <ol className="flex flex-col gap-1.5 p-2">
        {result.hits.map((hit) => {
          const isDivergent = divergence.divergent.has(hit.doc_id);
          const bg = isDivergent ? divergence.color[hit.doc_id] : undefined;
          const isHovered = hoveredDoc === hit.doc_id;
          return (
            <li
              key={hit.doc_id}
              onMouseEnter={() => onHover(hit.doc_id)}
              onMouseLeave={() => onHover(null)}
              className={`rounded-lg border p-2 text-xs transition ${
                isHovered ? "ring-2 ring-indigo-400" : "border-slate-100"
              }`}
              style={{ background: bg }}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] font-semibold text-slate-700">
                  #{hit.rank}
                </span>
                <span
                  className="rounded bg-slate-900/5 px-1.5 py-0.5 font-mono text-[10px]"
                  title="Relevance score this strategy gave. Compare ranks (#1, #2…) across columns, not raw scores."
                >
                  {hit.score.toFixed(3)}
                </span>
              </div>
              <p className="mt-1 line-clamp-3 text-slate-600">{hit.snippet || hit.doc_id}</p>
              {isDivergent && (
                <p className="mt-1 truncate text-[10px] font-medium text-slate-500">
                  {rankSummary(divergence.ranks[hit.doc_id])}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
