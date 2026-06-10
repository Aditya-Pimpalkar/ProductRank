// Divergence detection (PR-20 / FR-6) — the demo's whole point.
//
// A document that ranks #3 in one variant and #47 (or absent) in another is exactly the
// signal that the variants disagree. We compute, per document, the spread between its
// best and worst rank across the four columns; documents with a large spread get a
// shared highlight color so the eye is drawn to "this doc moved a lot."

import type { SearchResponse, Variant } from "./api";

export interface DivergenceInfo {
  // doc_id → rank (1-based) within each variant; missing = not in that variant's top-k.
  ranks: Record<string, Partial<Record<Variant, number>>>;
  // doc_id → true if the doc's rank spread across variants is "large".
  divergent: Set<string>;
  // doc_id → a stable highlight color (only assigned to divergent docs).
  color: Record<string, string>;
}

const HIGHLIGHT_PALETTE = [
  "#fde68a", // amber
  "#bfdbfe", // blue
  "#bbf7d0", // green
  "#fbcfe8", // pink
  "#ddd6fe", // violet
  "#fecaca", // red
];

// A doc is "divergent" if it appears in at least two variants AND the gap between its
// best and worst rank is at least this many positions.
const SPREAD_THRESHOLD = 5;

export function computeDivergence(results: SearchResponse[], topK: number): DivergenceInfo {
  const ranks: DivergenceInfo["ranks"] = {};

  for (const res of results) {
    for (const hit of res.hits) {
      (ranks[hit.doc_id] ||= {})[res.variant] = hit.rank;
    }
  }

  const divergent = new Set<string>();
  const color: Record<string, string> = {};
  let colorIdx = 0;

  // Sort doc ids by spread descending so the most divergent get the most distinct colors.
  const scored = Object.entries(ranks)
    .map(([docId, perVariant]) => {
      const present = Object.values(perVariant) as number[];
      // Treat "absent from a variant" as a rank just past the list, so dropping out counts.
      const appearedIn = present.length;
      const worst = appearedIn < results.length ? topK + 1 : Math.max(...present);
      const best = Math.min(...present);
      return { docId, spread: worst - best, appearedIn };
    })
    .filter((d) => d.appearedIn >= 2 && d.spread >= SPREAD_THRESHOLD)
    .sort((a, b) => b.spread - a.spread);

  for (const { docId } of scored) {
    divergent.add(docId);
    color[docId] = HIGHLIGHT_PALETTE[colorIdx % HIGHLIGHT_PALETTE.length];
    colorIdx += 1;
  }

  return { ranks, divergent, color };
}

export function rankSummary(
  perVariant: Partial<Record<Variant, number>>,
): string {
  // e.g. "#3 in Dense, #47 in BM25"
  const parts = Object.entries(perVariant).map(([v, r]) => `#${r} ${v}`);
  return parts.join(" · ");
}
