"""Thin wrapper over the OpenAI-compatible NVIDIA NIM endpoint (hosting GLM).

Handles the chat call, JSON extraction from possibly-reasoning output, and retry.
"""
from __future__ import annotations

import json
import re
import time

from openai import OpenAI

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of model output that may include reasoning/markdown."""
    if not text:
        raise ValueError("empty model response")
    cleaned = _THINK_RE.sub("", text).strip()
    # Strip ```json fences if present.
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    raise ValueError(f"no JSON object found in response: {cleaned[:200]!r}")


class VisionClient:
    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout: float = 120.0, max_retries: int = 4):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.last_usage: dict = {}

    def complete_json(self, system_prompt: str, user_blocks: list[dict],
                      max_tokens: int = 1500) -> dict:
        """Call the model and return the parsed JSON object.

        Tries response_format=json_object first; falls back to plain parsing if the
        endpoint rejects that parameter. Retries transient errors with backoff.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_blocks},
        ]
        use_json_mode = True
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, messages=messages,
                              max_tokens=max_tokens, temperature=0.0)
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                self._record_usage(resp)
                content = resp.choices[0].message.content or ""
                return extract_json(content)
            except Exception as e:  # noqa: BLE001 - surface after retries
                msg = str(e).lower()
                last_err = e
                # If json mode is unsupported, drop it and retry immediately.
                if use_json_mode and ("response_format" in msg or "json" in msg
                                      and "schema" in msg):
                    use_json_mode = False
                    continue
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        raise RuntimeError(f"model call failed after {self.max_retries} attempts: {last_err}")

    def _record_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
