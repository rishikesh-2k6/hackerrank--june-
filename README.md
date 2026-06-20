# Multi-Modal Evidence Review

A system that verifies damage claims for **cars, laptops, and packages** by
inspecting the submitted images alongside the claim conversation, the user's
claim history, and the minimum evidence requirements. It reads
`dataset/claims.csv` and produces `output.csv` with the exact 14-column schema in
[`problem_statement.md`](./problem_statement.md).

Built for the **HackerRank Orchestrate** (June 2026) challenge.

---

## Approach

**One structured vision-model call per claim.** For each row, the system sends a
single request containing:

- all of the claim's images (decoded, downscaled, and sent **once** each),
- the claim chat transcript (what to verify),
- the matched minimum-evidence requirement for that object type,
- the user's history summary (as *risk context only*).

The model returns every decision field as JSON. A **deterministic rule layer**
then folds in non-visual, history-based risk and clamps every field to the
allowed vocabularies, so each output row is schema-legal and reproducible.

Guiding principles, straight from the task:

- **Images are the source of truth.** The conversation defines what to check.
- **History adds risk context only** — it never overrides clear visual evidence
  on its own. That contribution is applied deterministically and is auditable.
- **No hardcoded labels.** Decisions come from the model + generic rules.

### Why this design

| Choice | Reason |
|---|---|
| Single call per claim (not multi-agent) | Cheapest / lowest latency for a 64-row dataset; easy to make deterministic. |
| Deterministic post-processing | History risk + enum clamping are auditable and reproducible, not left to the model. |
| Disk cache keyed on the full input | Each image/claim hits the model at most once; re-runs and sample/test overlap are free. |
| Provider-agnostic (OpenAI-compatible) | Endpoint/model swappable via `.env` with no code changes. |

---

## Model / provider

Provider-agnostic through the OpenAI-compatible chat API. The default target is
**GLM-5.1 served on NVIDIA NIM**:

```
NVIDIA_BASE_URL = https://integrate.api.nvidia.com/v1
GLM_MODEL       = z-ai/glm-5.1
```

Any OpenAI-compatible vision model works — change `GLM_MODEL` / `NVIDIA_BASE_URL`
in `.env`. (If a model can't accept images, switch to a vision variant such as
`z-ai/glm-4.5v`.)

---

## Repository layout

```text
.
├── README.md                     # You are here
├── problem_statement.md          # Full task spec and I/O schema
├── AGENTS.md                     # Agent rules + chat-transcript logging
├── .env.example                  # Copy to .env and add your key
├── code/                         # The solution (this is what gets zipped)
│   ├── main.py                   # CLI: claims.csv -> output.csv
│   ├── config.py                 # Paths, model config, .env loading
│   ├── requirements.txt
│   ├── README.md                 # Code-level docs
│   ├── agent/
│   │   ├── schema.py             # 14-col order + allowed values + coercion
│   │   ├── loaders.py            # CSV + image loading / downscaling
│   │   ├── prompt.py             # System prompt + multimodal user message
│   │   ├── client.py             # OpenAI-compatible call + JSON parsing + retry
│   │   ├── risk_rules.py         # History-risk merge + schema clamping
│   │   ├── cache.py              # Disk cache of model responses
│   │   └── pipeline.py           # Process one claim end-to-end
│   └── evaluation/
│       ├── main.py               # Score vs sample labels
│       └── evaluation_report.md  # Generated: metrics + cost/latency analysis
├── docs/superpowers/specs/       # Design spec
└── dataset/                      # Provided corpus (excluded from the code zip)
```

---

## Setup

```bash
pip install -r code/requirements.txt
cp .env.example .env          # then paste your key into .env
```

`.env` (repo root — **git-ignored, never committed**):

```
NVIDIA_API_KEY=...                 # required
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
GLM_MODEL=z-ai/glm-5.1
```

Secrets are read from the environment / `.env` only — nothing is hardcoded.

---

## Run

```bash
# Smoke test on the first 2 rows (confirms the model + image input work)
python code/main.py --limit 2

# Full test set -> dataset/output.csv
python code/main.py

# Evaluation on the labeled sample set -> code/evaluation/
python code/evaluation/main.py
```

Useful flags: `--limit N`, `--no-cache`, `--claims PATH`, `--out PATH`,
`--dataset-dir PATH` (so a grader can point at its own data).

---

## Output schema

`output.csv` has one row per input claim with these columns, in order:

`user_id, image_paths, user_claim, claim_object, evidence_standard_met,
evidence_standard_met_reason, risk_flags, issue_type, object_part, claim_status,
claim_status_justification, supporting_image_ids, valid_image, severity`

`claim_status` ∈ `supported | contradicted | not_enough_information`. All other
enumerated fields are clamped to the vocabularies in `problem_statement.md`.

---

## Evaluation

`python code/evaluation/main.py` runs the pipeline on `sample_claims.csv`
(inputs only), compares predictions to the labeled columns, and writes
`code/evaluation/evaluation_report.md` containing:

- per-field exact-match accuracy and the `claim_status` confusion matrix,
- set-based precision / recall / F1 for `risk_flags` and `supporting_image_ids`,
- **operational analysis**: model calls, images processed, token usage, projected
  full-test-set cost, runtime, and the caching / retry / rate-limit strategy.

The model is configurable via `.env`, so swapping `GLM_MODEL` lets you compare
configurations against the same harness.

---

## Cost, latency & reliability

- **~1 model call per claim** (~44 for the test set, ~82 images). All of a row's
  images go in one request — no per-image or multi-pass calls.
- **Disk cache** (`code/.cache/`) keyed on model + claim + requirement + history
  + image bytes: re-runs cost zero calls.
- **Image downscaling** to a bounded long edge limits image tokens and payload.
- **Retry with exponential backoff** on transient/429 errors; JSON-mode with a
  graceful fallback to robust parsing; deterministic fallback row when an image
  set is unusable or a call fails.
- NVIDIA NIM's free tier means real spend is ≈ $0; the report also gives a
  paid-equivalent reference cost.

---

## Chat transcript logging

Per `AGENTS.md`, the development chat transcript is appended to a log file
outside the repo (never committed):

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate\log.txt` |

This is uploaded as the chat transcript at submission time.

---

## Submission checklist

1. **Code zip** — zip `code/` (excludes venvs, caches, and `dataset/`).
2. **Predictions CSV** — `output.csv` for every row in `dataset/claims.csv`.
3. **Chat transcript** — `log.txt` from the path above.

Before submitting, confirm `output.csv` has one row per input row and the exact
columns in the exact order, and that the `evaluation/` folder is included.
