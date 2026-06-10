// Typed API client. All calls go through Next's /api rewrite to the FastAPI backend,
// so the browser never holds a backend URL or any secret (ARCHITECTURE §12.5).

export type Variant = "bm25" | "dense" | "hybrid" | "hybrid_rerank";

export const VARIANTS: Variant[] = ["bm25", "dense", "hybrid", "hybrid_rerank"];

export const VARIANT_LABEL: Record<Variant, string> = {
  bm25: "BM25",
  dense: "Dense",
  hybrid: "Hybrid (RRF)",
  hybrid_rerank: "Hybrid + Rerank",
};

export const VARIANT_COLOR: Record<Variant, string> = {
  bm25: "#6366f1",
  dense: "#10b981",
  hybrid: "#f59e0b",
  hybrid_rerank: "#ef4444",
};

export interface Hit {
  rank: number;
  doc_id: string;
  score: number;
  title: string;
  snippet: string;
}

export interface SearchResponse {
  variant: Variant;
  query: string;
  total_latency_ms: number;
  stage_latency_ms: Record<string, number>;
  candidate_counts: Record<string, number>;
  cache_hit: boolean;
  hits: Hit[];
}

export interface SignificanceEntry {
  metric: string;
  mean_a: number;
  mean_b: number;
  mean_diff: number;
  p_value: number;
  ci_low: number;
  ci_high: number;
  significant: boolean;
}

export interface Experiment {
  id: string;
  status: "pending" | "running" | "completed" | "error";
  progress: number;
  variant_a?: Variant;
  variant_b?: Variant;
  query_set_size?: number;
  metrics_a?: Record<string, number>;
  metrics_b?: Record<string, number>;
  significance?: SignificanceEntry[];
  error?: string;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE || "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

export function search(query: string, variant: Variant, topK = 10): Promise<SearchResponse> {
  return post<SearchResponse>("/v1/search", { query, variant, top_k: topK });
}

// Fetch all four variants concurrently for the side-by-side comparison.
export function searchAll(query: string, topK = 10): Promise<SearchResponse[]> {
  return Promise.all(VARIANTS.map((v) => search(query, v, topK)));
}

export function startExperiment(
  variantA: Variant,
  variantB: Variant,
  querySetSize: number,
): Promise<Experiment> {
  return post<Experiment>("/v1/experiments", {
    variant_a: variantA,
    variant_b: variantB,
    query_set_size: querySetSize,
  });
}

export function getExperiment(id: string): Promise<Experiment> {
  return get<Experiment>(`/v1/experiments/${id}`);
}

// Example queries for the demo — real queries from the loaded MS MARCO `dev` sample, so
// each has a judged relevant passage in the corpus and the rerank lift is visible.
// (Swap these for financial questions if you seed FiQA instead.)
export const EXAMPLE_QUERIES: string[] = [
  "where did olives originate from",
  "what is the definition of pessimistic",
  "how long does it take corn to cook on the grill",
  "what causes spots on tree leaves",
  "where does microtubule formation occur",
  "what is the elevation of white pass in washington",
];
