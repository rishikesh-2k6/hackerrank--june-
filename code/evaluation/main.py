"""Evaluation workflow: score the agent on the labeled sample set.

Runs the pipeline on dataset/sample_claims.csv (inputs only), compares predictions
to the labeled columns, and writes:
  - evaluation/sample_predictions.csv
  - evaluation/evaluation_report.md   (metrics + operational/cost analysis)

Usage:
    python code/evaluation/main.py
    python code/evaluation/main.py --limit 5
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from agent import cache, loaders  # noqa: E402
from agent.client import VisionClient  # noqa: E402
from agent.pipeline import Pipeline  # noqa: E402
from agent.schema import OUTPUT_COLUMNS  # noqa: E402
from config import build_config  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# Representative paid pricing for the operational estimate (USD per 1M tokens).
# NVIDIA NIM is currently free / rate-limited; these are stated assumptions only.
ASSUMED_INPUT_USD_PER_M = 0.60
ASSUMED_OUTPUT_USD_PER_M = 2.00

INPUT_COLS = ["user_id", "image_paths", "user_claim", "claim_object"]
EXACT_MATCH_FIELDS = [
    "evidence_standard_met", "issue_type", "object_part",
    "claim_status", "valid_image", "severity",
]


def _set(value: str) -> set[str]:
    return {v.strip() for v in (value or "").split(";") if v.strip() and v.strip() != "none"}


def set_prf(pred: str, gold: str) -> tuple[int, int, int]:
    """Return (true_positive, pred_count, gold_count) for set-valued fields."""
    p, g = _set(pred), _set(gold)
    return len(p & g), len(p), len(g)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate on the labeled sample set")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    config = build_config()
    labeled = loaders.read_claims(config.sample_csv)
    if args.limit:
        labeled = labeled[: args.limit]

    pipeline = Pipeline(
        config=config,
        client=VisionClient(config.api_key, config.base_url, config.model,
                            timeout=config.request_timeout, max_retries=config.max_retries),
        user_history=loaders.load_user_history(config.user_history_csv),
        evidence_reqs=loaders.load_evidence_requirements(config.evidence_csv),
        response_cache=cache.ResponseCache(config.cache_dir, enabled=not args.no_cache),
    )

    print(f"Evaluating {len(labeled)} sample claims with {config.model} ...")
    start = time.time()
    preds: list[dict] = []
    for i, gold in enumerate(labeled, 1):
        claim = {c: gold.get(c, "") for c in INPUT_COLS}
        pred = pipeline.process(claim)
        preds.append(pred)
        ok = "OK " if pred["claim_status"] == gold.get("claim_status") else "XX "
        print(f"  [{i}/{len(labeled)}] {ok}{claim['user_id']:<10} "
              f"pred={pred['claim_status']:<22} gold={gold.get('claim_status','')}")
    elapsed = time.time() - start

    # ---- metrics ----
    field_correct = Counter()
    for pred, gold in zip(preds, labeled):
        for fld in EXACT_MATCH_FIELDS:
            if pred[fld] == (gold.get(fld) or "").strip():
                field_correct[fld] += 1
    n = len(labeled)

    confusion = defaultdict(Counter)
    for pred, gold in zip(preds, labeled):
        confusion[gold.get("claim_status", "")][pred["claim_status"]] += 1

    # set-valued: risk_flags, supporting_image_ids
    set_metrics = {}
    for fld in ("risk_flags", "supporting_image_ids"):
        tp = pc = gc = 0
        for pred, gold in zip(preds, labeled):
            a, b, c = set_prf(pred[fld], gold.get(fld, ""))
            tp += a; pc += b; gc += c
        precision = tp / pc if pc else 1.0
        recall = tp / gc if gc else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        set_metrics[fld] = (precision, recall, f1)

    _write_predictions(preds)
    _write_report(config, n, elapsed, pipeline.stats, field_correct, confusion, set_metrics)

    acc = field_correct["claim_status"] / n if n else 0
    print(f"\nclaim_status accuracy: {acc:.1%}  ({field_correct['claim_status']}/{n})")
    print(f"Report -> {EVAL_DIR / 'evaluation_report.md'}")
    return 0


def _write_predictions(preds: list[dict]) -> None:
    path = EVAL_DIR / "sample_predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(OUTPUT_COLUMNS)
        for row in preds:
            writer.writerow([row.get(c, "") for c in OUTPUT_COLUMNS])


def _write_report(config, n, elapsed, stats, field_correct, confusion, set_metrics) -> None:
    calls = stats["calls"] + stats["cache_hits"]
    avg_in = stats["prompt_tokens"] / stats["calls"] if stats["calls"] else 0
    avg_out = stats["completion_tokens"] / stats["calls"] if stats["calls"] else 0

    # Extrapolate to the full test set (44 claims) using per-call averages.
    test_claims = 44
    est_in = avg_in * test_claims
    est_out = avg_out * test_claims
    est_cost = (est_in / 1e6 * ASSUMED_INPUT_USD_PER_M
                + est_out / 1e6 * ASSUMED_OUTPUT_USD_PER_M)

    lines = []
    lines.append("# Evaluation Report — Multi-Modal Evidence Review\n")
    lines.append(f"- Model: `{config.model}` via `{config.base_url}`")
    lines.append(f"- Sample claims evaluated: {n}")
    lines.append(f"- Wall-clock runtime: {elapsed:.1f}s "
                 f"({elapsed / n:.1f}s/claim)\n" if n else "\n")

    lines.append("## Accuracy on the labeled sample set\n")
    lines.append("| Field | Exact-match accuracy |")
    lines.append("|---|---|")
    for fld in EXACT_MATCH_FIELDS:
        lines.append(f"| `{fld}` | {field_correct[fld] / n:.1%} ({field_correct[fld]}/{n}) |")
    lines.append("")
    lines.append("| Set field | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for fld, (p, r, f1) in set_metrics.items():
        lines.append(f"| `{fld}` | {p:.2f} | {r:.2f} | {f1:.2f} |")
    lines.append("")

    lines.append("### claim_status confusion matrix (rows = gold, cols = predicted)\n")
    statuses = ["supported", "contradicted", "not_enough_information"]
    lines.append("| gold \\ pred | " + " | ".join(statuses) + " |")
    lines.append("|" + "---|" * (len(statuses) + 1))
    for g in statuses:
        row = " | ".join(str(confusion[g][p]) for p in statuses)
        lines.append(f"| **{g}** | {row} |")
    lines.append("")

    lines.append("## Operational analysis\n")
    lines.append(f"- Model calls this run: **{stats['calls']}** "
                 f"(+{stats['cache_hits']} served from cache)")
    lines.append(f"- Images processed: **{stats['images']}**")
    lines.append(f"- Errors / fallbacks: {stats['errors']}")
    lines.append(f"- Tokens (sample run): input={stats['prompt_tokens']}, "
                 f"output={stats['completion_tokens']}")
    lines.append(f"- Avg per call: ~{avg_in:.0f} input / ~{avg_out:.0f} output tokens\n")
    lines.append("### Full test set (44 claims) projection\n")
    lines.append(f"- ~44 model calls (1 per claim), ~82 images")
    lines.append(f"- Estimated tokens: ~{est_in:,.0f} input / ~{est_out:,.0f} output")
    lines.append(f"- **Estimated cost: ~${est_cost:.3f}** "
                 f"(assumes ${ASSUMED_INPUT_USD_PER_M}/1M input, "
                 f"${ASSUMED_OUTPUT_USD_PER_M}/1M output)")
    lines.append("- NVIDIA NIM is currently free / rate-limited, so real spend is ~$0; "
                 "the figure above is a paid-equivalent reference.\n")
    lines.append("### Cost, latency & rate-limit strategy\n")
    lines.append("- **1 call per claim.** All of a row's images go in a single request; "
                 "no per-image or multi-pass calls.")
    lines.append("- **Disk cache** keyed on (model + claim + requirement + history + image "
                 "bytes): re-runs and the sample/test overlap cost zero calls.")
    lines.append("- **Image downscaling** to a "
                 f"{config.max_image_edge}px long edge bounds image tokens and payload size.")
    lines.append("- **Retry with exponential backoff** on transient/429 errors; "
                 "JSON-mode with graceful fallback to plain parsing.")
    lines.append("- Sequential requests keep well under NIM RPM limits; raise concurrency "
                 "only if a larger test set needs it.")
    lines.append("")

    (EVAL_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
