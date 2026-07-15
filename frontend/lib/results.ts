import type { Dataset, Variant } from "./api";
import { DEFAULT_DATASET } from "./api";

export interface ResultsResponse {
  split: string;
  num_queries: number;
  top_k: number;
  aggregate: Record<Variant, Record<string, number>>;
  wall_seconds: Record<string, number>;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE || "/api";

// The backend maps dataset → split (msmarco→dev, fiqa→test), so the dashboard and live
// search always reflect the same corpus.
export async function getResults(dataset: Dataset = DEFAULT_DATASET): Promise<ResultsResponse> {
  const res = await fetch(`${BASE}/v1/results?dataset=${dataset}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/v1/results → ${res.status}`);
  return res.json();
}
