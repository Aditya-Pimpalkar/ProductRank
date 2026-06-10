"""One-command seed: bring a clean DB to a fully populated FiQA corpus + qrels.

    python seed.py            # download (if needed) + load corpus/queries/qrels
    python seed.py --split dev

Idempotent: re-running inserts nothing. Embeddings are a separate, heavier step
(`python -m productrank.cli embed`) so the data load stays fast and API-key-free.
Order matters: documents and queries load before qrels (FK targets).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.ingest.fiqa import (
    download_fiqa,
    iter_corpus,
    iter_qrels,
    iter_queries,
)

from productrank import ingest
from productrank.db import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed FiQA into Postgres")
    parser.add_argument("--split", default="test", help="qrels split to load (test/dev/train)")
    parser.add_argument("--raw-dir", default="data/raw", help="dataset cache dir")
    args = parser.parse_args()

    print("→ downloading / locating FiQA …")
    fiqa_dir = download_fiqa(Path(args.raw_dir))

    with SessionLocal() as session:
        before = ingest.counts(session)

        print("→ loading documents …")
        ingest.load_documents(session, iter_corpus(fiqa_dir))

        print(f"→ loading queries (split={args.split}) …")
        ingest.load_queries(session, iter_queries(fiqa_dir, args.split))

        print("→ loading qrels …")
        ingest.load_qrels(session, iter_qrels(fiqa_dir, args.split))

        after = ingest.counts(session)
        # Report inserts as count deltas: psycopg returns rowcount -1 for the
        # ON CONFLICT multi-row inserts, so deltas are the reliable signal.
        added = {k: after[k] - before[k] for k in after}
        ingest.record_phase(
            session, f"ingest:{args.split}", added["documents"] + added["queries"] + added["qrels"]
        )

    print("\nInserted this run:")
    for k in ("documents", "queries", "qrels"):
        print(f"  {k:10s}: +{added[k]}")
    print("\nTotals in DB:")
    for k, v in after.items():
        print(f"  {k:10s}: {v}")
    if sum(added.values()) == 0:
        print("\n(no-op: corpus already seeded)")
    print("\nNext: python -m productrank.cli embed   # builds dense vectors + IVFFlat index")


if __name__ == "__main__":
    main()
