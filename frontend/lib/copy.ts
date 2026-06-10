// Plain-language content in one place: variant blurbs, the "how it works" steps, and a
// glossary of every jargon term shown in the UI. Goal — a non-technical visitor
// understands what they're seeing, while the real metric names stay visible for the
// technical crowd.

import type { Variant } from "./api";

// One-line, human explanation under each variant column header.
export const VARIANT_BLURB: Record<Variant, string> = {
  bm25: "Matches your exact words",
  dense: "Matches the meaning",
  hybrid: "Blends words + meaning",
  hybrid_rerank: "Re-reads each result closely",
};

// Slightly longer "why" for tooltips on the column headers.
export const VARIANT_TIP: Record<Variant, string> = {
  bm25:
    "Keyword search (BM25). Finds documents that share words with your question, favoring rare, telling words. Fast, but blind to synonyms — it can match 'potatoes' for an 'olives' question.",
  dense:
    "Meaning search. Turns your question and every document into 1,536 numbers that capture meaning, then finds the closest ones. Understands 'couch' ≈ 'sofa', but can blur exact terms.",
  hybrid:
    "Combines the keyword list and the meaning list by rank position (Reciprocal Rank Fusion) — no need to compare their incompatible scores.",
  hybrid_rerank:
    "Takes the top ~100 candidates and re-reads each one together with your question using a slower, smarter model (a cross-encoder) to reorder them. Accurate but expensive — so it only runs on the shortlist.",
};

export interface HowStep {
  icon: string;
  title: string;
  body: string;
}

export const HOW_IT_WORKS: HowStep[] = [
  {
    icon: "🔤",
    title: "1 · Match words",
    body: "Grab documents that share words with your question (fast, literal).",
  },
  {
    icon: "🧠",
    title: "2 · Match meaning",
    body: "Grab documents that mean the same thing, even with different words.",
  },
  {
    icon: "🔀",
    title: "3 · Combine",
    body: "Merge both shortlists into one ranked list.",
  },
  {
    icon: "🔍",
    title: "4 · Re-read & reorder",
    body: "A smarter model reads each finalist closely and puts the best on top.",
  },
];

// term key → { label shown, plain tooltip }. Keep tips short and concrete.
export const GLOSSARY: Record<string, string> = {
  "NDCG@10":
    "Ranking quality score from 0 to 1 for the top 10 results. Higher means the genuinely relevant results sit closer to the top. The headline number.",
  "NDCG@100": "Same ranking-quality score, but judged over the top 100 results.",
  MRR: "Mean Reciprocal Rank — how high up the FIRST correct result appears, on average.",
  MAP: "Mean Average Precision — overall quality across all the correct results, not just the first.",
  "Recall@10": "Of all the correct results that exist, what fraction showed up in the top 10.",
  "Recall@100": "Of all the correct results that exist, what fraction showed up in the top 100.",
  "P@10": "Precision@10 — of the top 10 results shown, what fraction were actually correct.",
  score:
    "The raw relevance score this strategy gave the document. Scales differ between strategies, so compare ranks (#1, #2…), not raw scores across columns.",
  latency: "How long this strategy took to answer, in milliseconds.",
  divergence:
    "When the same document lands at very different positions across strategies (e.g. #2 here, #47 there). That disagreement is exactly what this demo is built to show.",
  significance:
    "A statistical check (paired t-test + bootstrap) answering: is this improvement real, or just luck on these questions? ✱ means real.",
  "p-value":
    "Probability the difference is just luck. Below 0.05 is the usual bar for 'this is a real effect'.",
  "95% CI":
    "95% confidence interval for the improvement. If it doesn't cross 0, the effect is real with 95% confidence.",
};
