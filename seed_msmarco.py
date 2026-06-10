"""Seed a *sampled* MS MARCO passage-ranking corpus into Postgres.

    python seed_msmarco.py                 # wipe + load 1000 dev queries, ~51K passages
    python seed_msmarco.py --queries 2000 --distractors 80000

MS MARCO is the PRD §10 contingency dataset: on it the cross-encoder is in-domain, so
reranking gives the textbook lift over first-stage retrieval (which FiQA didn't show).
The full corpus is 8.8M passages; we load a self-contained sample (see data/ingest/
msmarco.py). This WIPES any existing corpus first — doc ids aren't namespaced across
datasets, so FiQA and MS MARCO can't coexist in one schema.

After seeding: `python -m productrank.cli embed` then `... eval --split dev`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.ingest.msmarco import MsMarcoSample, download_msmarco

from productrank import ingest
from productrank.db import SessionLocal


def main() -> None:
    p = argparse.ArgumentParser(description="Seed sampled MS MARCO into Postgres")
    p.add_argument("--split", default="dev", help="qrels split (dev is MS MARCO's eval split)")
    p.add_argument("--queries", type=int, default=1000, help="number of queries to sample")
    p.add_argument("--distractors", type=int, default=50_000, help="non-relevant passages to add")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--no-wipe", action="store_true", help="skip wiping existing corpus")
    args = p.parse_args()

    print("→ downloading / locating MS MARCO (~1 GB) …")
    ms_dir = download_msmarco(Path(args.raw_dir))

    print(f"→ building sample: {args.queries} queries + {args.distractors} distractors …")
    sample = MsMarcoSample(
        ms_dir,
        split=args.split,
        n_queries=args.queries,
        n_distractors=args.distractors,
        seed=args.seed,
    )
    print(f"  sampled {len(sample.query_ids)} queries, "
          f"{len(sample.relevant_doc_ids)} relevant passages, {len(sample.qrels)} qrels")

    with SessionLocal() as session:
        if not args.no_wipe:
            print("→ wiping existing corpus (dataset swap) …")
            ingest.wipe(session)

        print("→ loading sampled corpus (relevant + distractors) …")
        ingest.load_documents(session, sample.iter_corpus())

        print("→ loading queries …")
        ingest.load_queries(session, sample.iter_queries())

        print("→ loading qrels …")
        ingest.load_qrels(session, sample.iter_qrels())

        ingest.record_phase(session, f"msmarco:{args.split}", len(sample.qrels))
        totals = ingest.counts(session)

    print("\nTotals in DB:")
    for k, v in totals.items():
        print(f"  {k:10s}: {v}")
    print("\nNext: python -m productrank.cli embed   # builds dense vectors + IVFFlat index")
    print("Then: productrank eval --split dev")


if __name__ == "__main__":
    main()
