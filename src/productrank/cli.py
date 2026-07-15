"""ProductRank CLI.

python -m productrank.cli embed         # embed corpus (resumable) + build index
python -m productrank.cli embed-only    # embed without (re)building the index
python -m productrank.cli build-index   # (re)build the IVFFlat index only
python -m productrank.cli eval          # run all four variants, print metrics (PR-10)
"""

from __future__ import annotations

import argparse

from productrank.config import DATASET_SPLIT, DATASETS, DEFAULT_DATASET


def _cmd_embed(args: argparse.Namespace) -> None:
    from productrank.embed import build_ivfflat_index, embed_corpus

    embed_corpus(dataset=args.dataset)
    build_ivfflat_index(dataset=args.dataset)


def _cmd_embed_only(args: argparse.Namespace) -> None:
    from productrank.embed import embed_corpus

    embed_corpus(dataset=args.dataset)


def _cmd_build_index(args: argparse.Namespace) -> None:
    from productrank.embed import build_ivfflat_index

    build_ivfflat_index(dataset=args.dataset)


def _cmd_eval(args: argparse.Namespace) -> None:
    from productrank.evaluation.run import run_evaluation

    only = args.variants.split(",") if args.variants else None
    # Default the qrels split from the dataset unless the caller overrides it.
    split = args.split or DATASET_SPLIT[args.dataset]
    run_evaluation(
        limit=args.limit,
        top_k=args.top_k,
        split=split,
        only=only,
        tag=args.tag,
        dataset=args.dataset,
    )


def _add_dataset_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset",
        choices=DATASETS,
        default=DEFAULT_DATASET,
        help=f"which dataset's database to target (default: {DEFAULT_DATASET})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="productrank")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("embed", "embed corpus (resumable) then build IVFFlat index"),
        ("embed-only", "embed corpus without building the index"),
        ("build-index", "(re)build the IVFFlat dense index"),
    ]:
        _add_dataset_arg(sub.add_parser(name, help=help_text))

    ev = sub.add_parser("eval", help="evaluate all four variants over the query set")
    _add_dataset_arg(ev)
    ev.add_argument("--limit", type=int, default=None, help="max queries (default: all)")
    ev.add_argument("--top-k", type=int, default=100, help="retrieval depth for metrics")
    ev.add_argument(
        "--split", default=None, help="qrels split (default: derived from --dataset)"
    )
    ev.add_argument(
        "--variants",
        default=None,
        help="comma-separated subset to run, e.g. 'hybrid_rerank' (default: all); "
        "merges into the existing results file",
    )
    ev.add_argument(
        "--tag",
        default=None,
        help="suffix the output file (e.g. --tag sample100 → results/eval_test_sample100.json)",
    )

    args = parser.parse_args()
    {
        "embed": _cmd_embed,
        "embed-only": _cmd_embed_only,
        "build-index": _cmd_build_index,
        "eval": _cmd_eval,
    }[args.command](args)


if __name__ == "__main__":
    main()
