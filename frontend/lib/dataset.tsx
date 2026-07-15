"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { DEFAULT_DATASET, type Dataset } from "./api";

// Shared dataset selection across all pages. Persisted to localStorage so the choice
// survives navigation and reloads. The toggle and every page read the same value, so the
// search dataset and the analytics dashboard always move together.
interface DatasetCtx {
  dataset: Dataset;
  setDataset: (d: Dataset) => void;
}

const Ctx = createContext<DatasetCtx>({
  dataset: DEFAULT_DATASET,
  setDataset: () => {},
});

const KEY = "productrank.dataset";

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [dataset, setDatasetState] = useState<Dataset>(DEFAULT_DATASET);

  useEffect(() => {
    const saved = localStorage.getItem(KEY);
    if (saved === "msmarco" || saved === "fiqa") setDatasetState(saved);
  }, []);

  function setDataset(d: Dataset) {
    setDatasetState(d);
    try {
      localStorage.setItem(KEY, d);
    } catch {
      /* ignore storage errors */
    }
  }

  return <Ctx.Provider value={{ dataset, setDataset }}>{children}</Ctx.Provider>;
}

export function useDataset(): DatasetCtx {
  return useContext(Ctx);
}
