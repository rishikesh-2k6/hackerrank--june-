# Multi-Modal Evidence Review — Design

Date: 2026-06-19
Status: implemented

## Problem

Verify damage claims about a `car`, `laptop`, or `package` by inspecting
submitted images alongside the claim chat transcript, the user's claim history,
and minimum evidence requirements. Read `dataset/claims.csv`; produce
`output.csv` with the exact 14-column schema in `problem_statement.md`. Include an
evaluation workflow. No hardcoded test labels.

## Decisions

- **Provider:** OpenAI-compatible API. Target = GLM-5.1 on NVIDIA NIM
  (`integrate.api.nvidia.com`, model `z-ai/glm-5.1`), key from `.env`
  (`NVIDIA_API_KEY`). Endpoint/model swappable via env for any vision model.
- **Architecture:** one structured vision call per claim (chosen over multi-stage
  or multi-agent for cost/latency on a 64-row dataset).
- **Determinism:** `temperature=0` + JSON output + disk cache keyed on the full
  input. Each image/claim is sent to the model at most once ("read once").
- **History rule:** history adds risk context only; never overrides clear visual
  evidence. Applied deterministically in a post-processing layer.

## Components (`code/`)

| File | Responsibility |
|---|---|
| `config.py` | Paths, model config, `.env` loading. Paths overridable via CLI. |
| `agent/schema.py` | 14-col order, allowed-value vocab, coercion/clamping. |
| `agent/loaders.py` | CSV reads, history/requirement lookup, image encode+downscale. |
| `agent/prompt.py` | System prompt + multimodal user message. |
| `agent/client.py` | OpenAI-compatible call, JSON extraction (handles reasoning/fences), retry. |
| `agent/risk_rules.py` | History-risk merge, manual-review escalation, schema clamp, fallback row. |
| `agent/cache.py` | Disk cache of model responses. |
| `agent/pipeline.py` | Process one claim end-to-end; track stats. |
| `main.py` | CLI: `claims.csv` → `output.csv`. |
| `evaluation/main.py` | Score vs `sample_claims.csv` labels; write report. |

## Data flow

claims.csv row → encode images (downscaled, once) → cache lookup →
(miss) build prompt → model JSON → cache write → risk-rule post-process →
14-col output row → output.csv.

Missing/unusable images or a failed call → deterministic schema-legal fallback
row (`not_enough_information`, `manual_review_required`).

## Evaluation

`evaluation/main.py` runs the pipeline on the labeled sample inputs and reports:
per-field exact-match accuracy, `claim_status` confusion matrix, set-based
precision/recall/F1 for `risk_flags` and `supporting_image_ids`, plus the
required operational analysis (calls, images, tokens, projected test-set
cost/latency, and the caching/retry/rate-limit strategy).

## Out of scope (YAGNI)

No multi-agent orchestration, vector retrieval, web UI, or database — 64 rows.

## Open risk

`z-ai/glm-5.1` vision support on NVIDIA NIM is unconfirmed; if it rejects images,
set `GLM_MODEL=z-ai/glm-4.5v` (or another vision model) in `.env`. First run
should be `python code/main.py --limit 2` to confirm image input works.
