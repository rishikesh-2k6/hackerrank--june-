"""End-to-end processing of a single claim row."""
from __future__ import annotations

from pathlib import Path

from . import cache, loaders, prompt, risk_rules
from .client import VisionClient


class Pipeline:
    def __init__(self, config, client: VisionClient,
                 user_history: dict, evidence_reqs: dict,
                 response_cache: cache.ResponseCache):
        self.config = config
        self.client = client
        self.user_history = user_history
        self.evidence_reqs = evidence_reqs
        self.cache = response_cache
        # running totals for the operational report
        self.stats = {"calls": 0, "cache_hits": 0, "images": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def _requirement_text(self, claim_object: str) -> str:
        reqs = loaders.requirements_for(self.evidence_reqs, claim_object)
        return "\n".join(f"- {r.get('minimum_image_evidence', '').strip()}" for r in reqs)

    def process(self, claim: dict) -> dict:
        claim_object = claim.get("claim_object", "")
        history = self.user_history.get(claim.get("user_id", ""))
        history_summary = (history or {}).get("history_summary", "") if history else ""
        rel_paths = loaders.parse_image_paths(claim.get("image_paths", ""))

        # Encode images (read once each).
        encoded: list[tuple[str, str, str]] = []  # (id, media_type, b64)
        for rel in rel_paths:
            abs_path = self.config.dataset_dir / rel
            enc = loaders.encode_image(abs_path, self.config.max_image_edge)
            if enc is not None:
                encoded.append((loaders.image_id(rel), enc[0], enc[1]))
        valid_ids = [e[0] for e in encoded]
        self.stats["images"] += len(encoded)

        if not encoded:
            self.stats["errors"] += 1
            return risk_rules.fallback_row(
                claim, history, "No usable images were submitted for this claim.")

        requirement_text = self._requirement_text(claim_object)

        # Cache key: everything that affects the model output.
        key = cache.make_key(
            self.config.model, claim_object, claim.get("user_claim", ""),
            requirement_text, history_summary,
            *[f"{i}:{b}" for i, _, b in encoded],
        )
        model_out = self.cache.get(key)
        if model_out is not None:
            self.stats["cache_hits"] += 1
        else:
            blocks = prompt.build_user_blocks(claim, requirement_text, history_summary, encoded)
            try:
                model_out = self.client.complete_json(prompt.SYSTEM_PROMPT, blocks)
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                return risk_rules.fallback_row(
                    claim, history, f"Automated review failed: {type(e).__name__}.")
            self.stats["calls"] += 1
            self.stats["prompt_tokens"] += self.client.last_usage.get("prompt_tokens", 0)
            self.stats["completion_tokens"] += self.client.last_usage.get("completion_tokens", 0)
            self.cache.set(key, model_out)

        return risk_rules.build_output_row(claim, model_out, history, valid_ids)
