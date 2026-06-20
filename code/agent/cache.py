"""Disk cache of model responses, keyed by a hash of the full input.

Guarantees each unique (images + claim + requirement + history + model) is sent to
the model at most once — re-runs and the dev loop are free and reproducible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def make_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class ResponseCache:
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.dir = cache_dir
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self.dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self.dir / f"{key}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
