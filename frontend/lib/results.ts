import type { Variant } from "./api";

export interface ResultsResponse {
  split: string;
  num_queries: number;
  top_k: number;
  aggregate: Record<Variant, Record<string, number>>;
  wall_seconds: Record<string, number>;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE || "/api";

export async function getResults(split = "dev"): Promise<ResultsResponse> {
  const res = await fetch(`${BASE}/v1/results?split=${split}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`/v1/results → ${res.status}`);
  return res.json();
}
