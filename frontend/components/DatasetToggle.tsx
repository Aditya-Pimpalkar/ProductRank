"use client";

import { DATASETS, DATASET_LABEL } from "@/lib/api";
import { useDataset } from "@/lib/dataset";

// Segmented control in the header. Switching it moves the live search dataset AND the
// analytics dashboard together (both read the shared dataset context).
export function DatasetToggle() {
  const { dataset, setDataset } = useDataset();
  return (
    <div className="ml-auto flex items-center gap-2">
      <span className="hidden text-xs text-slate-400 sm:inline">Dataset</span>
      <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-0.5">
        {DATASETS.map((d) => (
          <button
            key={d}
            onClick={() => setDataset(d)}
            aria-pressed={dataset === d}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
              dataset === d
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {DATASET_LABEL[d]}
          </button>
        ))}
      </div>
    </div>
  );
}
