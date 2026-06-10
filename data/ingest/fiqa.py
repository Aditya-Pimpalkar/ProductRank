"""BEIR / FiQA-2018 download and parsing.

Why download the raw BEIR zip directly instead of pulling in the `beir` package:
`beir` drags in a large dependency tree (its own torch stack, faiss, elasticsearch
clients) that ProductRank does not use — NFR-6 says every dependency must earn its
place. The dataset is three plainly-formatted files; parsing them ourselves is a few
generators and keeps the dependency surface honest.

Dataset layout inside fiqa.zip:
  fiqa/corpus.jsonl       {"_id", "title", "text", "metadata"}
  fiqa/queries.jsonl      {"_id", "text", "metadata"}
  fiqa/qrels/{train,dev,test}.tsv   header: query-id  corpus-id  score
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
from tqdm import tqdm

FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

# Default on-disk cache. Git-ignored (data/raw/).
DEFAULT_RAW_DIR = Path("data/raw")


def download_fiqa(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download + extract fiqa.zip into raw_dir/fiqa. Idempotent: skips if present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = raw_dir / "fiqa"
    if (extracted / "corpus.jsonl").exists():
        return extracted

    zip_path = raw_dir / "fiqa.zip"
    if not zip_path.exists():
        with httpx.stream("GET", FIQA_URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                open(zip_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc="fiqa.zip") as bar,
            ):
                for chunk in r.iter_bytes(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return extracted


def iter_corpus(fiqa_dir: Path) -> Iterator[dict]:
    """Yield {id, title, text, doc_metadata} rows from corpus.jsonl."""
    with open(fiqa_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield {
                "id": d["_id"],
                "title": d.get("title") or "",
                "text": d.get("text") or "",
                "doc_metadata": d.get("metadata") or {},
            }


def iter_queries(fiqa_dir: Path, split: str = "test") -> Iterator[dict]:
    """Yield {id, text, split} rows, restricted to queries that appear in the split's
    qrels (BEIR ships all queries in one file; the split is defined by the qrels)."""
    split_qids = {qid for qid, _, _ in iter_qrels(fiqa_dir, split)}
    with open(fiqa_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["_id"] in split_qids:
                yield {"id": d["_id"], "text": d.get("text") or "", "split": split}


def iter_qrels(fiqa_dir: Path, split: str = "test") -> Iterator[tuple[str, str, int]]:
    """Yield (query_id, doc_id, relevance) from qrels/{split}.tsv."""
    path = fiqa_dir / "qrels" / f"{split}.tsv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header: query-id corpus-id score
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], row[2]
            yield qid, did, int(score)
