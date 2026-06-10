"""ProductRank CLI.

python -m productrank.cli embed         # embed corpus (resumable) + build index
python -m productrank.cli embed-only    # embed without (re)building the index
python -m productrank.cli build-index   # (re)build the IVFFlat index only
python -m productrank.cli eval          # run all four variants, print metrics (PR-10)
"""

from __future__ import annotations

import argparse


def _cmd_embed(args: argparse.Namespace) -> None:
    from productrank.embed import build_ivfflat_index, embed_corpus

    embed_corpus()
    build_ivfflat_index()


def _cmd_embed_only(args: argparse.Namespace) -> None:
    from productrank.embed import embed_corpus

    embed_corpus()


def _cmd_build_index(args: argparse.Namespace) -> None:
    from productrank.embed import build_ivfflat_index

    build_ivfflat_index()


def _cmd_eval(args: argparse.Namespace) -> None:
    from productrank.evaluation.run import run_evaluation

    only = args.variants.split(",") if args.variants else None
    run_evaluation(
        limit=args.limit, top_k=args.top_k, split=args.split, only=only, tag=args.tag
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="productrank")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("embed", help="embed corpus (resumable) then build IVFFlat index")
    sub.add_parser("embed-only", help="embed corpus without building the index")
    sub.add_parser("build-index", help="(re)build the IVFFlat dense index")

    ev = sub.add_parser("eval", help="evaluate all four variants over the query set")
    ev.add_argument("--limit", type=int, default=None, help="max queries (default: all)")
    ev.add_argument("--top-k", type=int, default=100, help="retrieval depth for metrics")
    ev.add_argument("--split", default="test", help="query split to evaluate")
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
