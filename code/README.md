# Multi-Modal Evidence Review

Verifies damage claims (car / laptop / package) by inspecting submitted images
together with the claim conversation, the user's history, and the minimum
evidence requirements. Reads `dataset/claims.csv` and produces `output.csv`.

## Approach

One **vision model call per claim**. Each row's images + claim transcript +
matched evidence requirement + user-history summary go in a single structured
request; the model returns all decision fields as JSON. A deterministic rule
layer then folds in non-visual (history-based) risk and clamps every field to the
allowed vocabularies from `problem_statement.md`, so each output row is
schema-legal and reproducible.

Images are the source of truth; history only adds risk context and never
overrides clear visual evidence on its own.

## Model / provider

Provider-agnostic via the OpenAI-compatible API. Default target is **GLM-5.1 on
NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`, model `z-ai/glm-5.1`).
Swap the model/endpoint in `.env` to use any OpenAI-compatible vision model.

> **Vision note:** the pipeline sends images as `image_url` data URIs. If
> `z-ai/glm-5.1` does not accept images on your endpoint, set `GLM_MODEL` to a
> GLM vision variant (e.g. `z-ai/glm-4.5v`) in `.env`.

## Setup

```bash
pip install -r code/requirements.txt
cp .env.example .env          # then paste your NVIDIA_API_KEY into .env
```

`.env` (repo root):

```
NVIDIA_API_KEY=...                 # required
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
GLM_MODEL=z-ai/glm-5.1
```

## Run

```bash
# Smoke test on 2 rows
python code/main.py --limit 2

# Full test set -> dataset/output.csv
python code/main.py

# Evaluation on the labeled sample set -> code/evaluation/
python code/evaluation/main.py
```

## Layout

```
code/
├── main.py                 # CLI: claims.csv -> output.csv
├── config.py               # paths, model, .env loading
├── requirements.txt
├── agent/
│   ├── schema.py           # output columns + allowed values + coercion
│   ├── loaders.py          # CSV + image loading/encoding
│   ├── prompt.py           # system prompt + multimodal user message
│   ├── client.py           # OpenAI-compatible call + JSON extraction + retry
│   ├── risk_rules.py       # history-risk merge + schema clamping
│   ├── cache.py            # disk cache of model responses
│   └── pipeline.py         # process one claim end-to-end
└── evaluation/
    ├── main.py             # score vs sample_claims.csv labels
    └── evaluation_report.md  # generated: metrics + cost/latency analysis
```

## Reproducibility & cost

- `temperature=0`, structured JSON output, and a disk cache keyed on the full
  input make runs deterministic and avoid repeated calls.
- ~1 call/claim, ~82 images for the full test set. See
  `evaluation/evaluation_report.md` for the measured token/cost/latency analysis.

No secrets are hardcoded; the key is read from `.env` / environment only.
`.env`, caches, and `dataset/` are git-ignored and excluded from the code zip.
