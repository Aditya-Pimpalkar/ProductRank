"""BEIR / MS MARCO passage-ranking download + **sampled** ingest.

Why MS MARCO, and why sampled (PRD §5 alternative, §10 contingency):
FiQA's strong dense embedding beats the MS-MARCO-trained cross-encoder, so the rerank
lift is undramatic there. On MS MARCO the cross-encoder is *in-domain*, so reranking
gives the textbook lift over first-stage retrieval. The full corpus is 8.8M passages —
far too large/expensive to embed on a personal budget — so we build a **self-contained
sample**: take a fixed set of dev queries, keep every judged (relevant) passage for them,
and add a fixed pool of distractor passages. qrels stay valid because every relevant
passage is in the corpus; the task stays honest because the distractors are real
passages the retriever must rank below the relevant one.

BEIR ships MS MARCO in the same layout as FiQA (corpus.jsonl / queries.jsonl /
qrels/{dev,train}.tsv), so the parsing mirrors data/ingest/fiqa.py. Evaluation uses the
`dev` qrels (MS MARCO's standard eval split; the `test` qrels are not public).
"""

from __future__ import annotations

import csv
import json
import random
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
from tqdm import tqdm

MSMARCO_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
DEFAULT_RAW_DIR = Path("data/raw")


def download_msmarco(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download + extract msmarco.zip (~1 GB) into raw_dir/msmarco. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = raw_dir / "msmarco"
    if (extracted / "corpus.jsonl").exists():
        return extracted

    zip_path = raw_dir / "msmarco.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".zip.part")
        with httpx.stream("GET", MSMARCO_URL, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with (
                open(tmp, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc="msmarco.zip") as bar,
            ):
                for chunk in r.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return extracted


def _read_qrels(msmarco_dir: Path, split: str) -> list[tuple[str, str, int]]:
    path = msmarco_dir / "qrels" / f"{split}.tsv"
    rows: list[tuple[str, str, int]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header: query-id corpus-id score
        for row in reader:
            if len(row) >= 3:
                rows.append((row[0], row[1], int(row[2])))
    return rows


class MsMarcoSample:
    """A reproducible MS MARCO sample: which queries, which qrels, which doc ids to keep."""

    def __init__(
        self,
        msmarco_dir: Path,
        split: str = "dev",
        n_queries: int = 1000,
        n_distractors: int = 50_000,
        seed: int = 13,
    ) -> None:
        self.dir = msmarco_dir
        self.split = split
        self.n_distractors = n_distractors

        all_qrels = _read_qrels(msmarco_dir, split)
        all_qids = sorted({q for q, _, _ in all_qrels})
        rng = random.Random(seed)
        sampled = set(rng.sample(all_qids, min(n_queries, len(all_qids))))

        self.query_ids: set[str] = sampled
        self.qrels: list[tuple[str, str, int]] = [
            (q, d, r) for (q, d, r) in all_qrels if q in sampled
        ]
        self.relevant_doc_ids: set[str] = {d for _, d, _ in self.qrels}

    def iter_corpus(self) -> Iterator[dict]:
        """Stream corpus.jsonl once; emit every relevant passage plus the first
        `n_distractors` non-relevant passages. Constant memory over the 8.8M-line file."""
        distractors = 0
        with open(self.dir / "corpus.jsonl", encoding="utf-8") as f:
            for line in tqdm(f, desc="scanning msmarco corpus", unit=" docs"):
                if not line.strip():
                    continue
                d = json.loads(line)
                did = d["_id"]
                keep = did in self.relevant_doc_ids
                if not keep and distractors < self.n_distractors:
                    keep = True
                    distractors += 1
                if keep:
                    yield {
                        "id": did,
                        "title": d.get("title") or "",
                        "text": d.get("text") or "",
                        "doc_metadata": d.get("metadata") or {},
                    }

    def iter_queries(self) -> Iterator[dict]:
        with open(self.dir / "queries.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["_id"] in self.query_ids:
                    yield {"id": d["_id"], "text": d.get("text") or "", "split": self.split}

    def iter_qrels(self) -> Iterator[tuple[str, str, int]]:
        yield from self.qrels
