"""Entry point: run the evidence-review agent over a claims CSV -> output.csv.

Usage:
    python code/main.py                      # dataset/claims.csv -> dataset/output.csv
    python code/main.py --limit 5            # smoke test on first 5 rows
    python code/main.py --no-cache           # ignore cached responses
    python code/main.py --claims path --out path --dataset-dir path
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import cache, loaders  # noqa: E402
from agent.client import VisionClient  # noqa: E402
from agent.pipeline import Pipeline  # noqa: E402
from agent.schema import OUTPUT_COLUMNS  # noqa: E402
from config import build_config  # noqa: E402


def write_output(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow([row.get(col, "") for col in OUTPUT_COLUMNS])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-modal evidence review")
    parser.add_argument("--claims", type=Path, default=None,
                        help="claims CSV (default dataset/claims.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV (default dataset/output.csv)")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="root the image_paths are relative to")
    parser.add_argument("--limit", type=int, default=None, help="process only N rows")
    parser.add_argument("--no-cache", action="store_true", help="ignore response cache")
    args = parser.parse_args(argv)

    config = build_config(dataset_dir=args.dataset_dir)
    claims_path = args.claims or config.claims_csv
    out_path = args.out or (config.dataset_dir / "output.csv")

    claims = loaders.read_claims(claims_path)
    if args.limit:
        claims = claims[: args.limit]

    pipeline = Pipeline(
        config=config,
        client=VisionClient(config.api_key, config.base_url, config.model,
                            timeout=config.request_timeout, max_retries=config.max_retries),
        user_history=loaders.load_user_history(config.user_history_csv),
        evidence_reqs=loaders.load_evidence_requirements(config.evidence_csv),
        response_cache=cache.ResponseCache(config.cache_dir, enabled=not args.no_cache),
    )

    print(f"Model: {config.model} via {config.base_url}")
    print(f"Processing {len(claims)} claims from {claims_path} ...")
    start = time.time()
    rows: list[dict] = []
    for i, claim in enumerate(claims, 1):
        row = pipeline.process(claim)
        rows.append(row)
        print(f"  [{i}/{len(claims)}] {claim.get('user_id','?'):<10} "
              f"-> {row['claim_status']:<22} ({row['issue_type']}/{row['object_part']})")

    write_output(rows, out_path)
    elapsed = time.time() - start
    s = pipeline.stats
    print(f"\nDone in {elapsed:.1f}s -> {out_path}")
    print(f"  model calls: {s['calls']}  cache hits: {s['cache_hits']}  "
          f"images: {s['images']}  errors: {s['errors']}")
    print(f"  tokens: prompt={s['prompt_tokens']} completion={s['completion_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
