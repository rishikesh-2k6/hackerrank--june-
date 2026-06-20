"""Runtime configuration: paths, model, and credentials.

Paths default to the repo layout (``code/`` and ``dataset/`` as siblings) but
every one can be overridden on the CLI so the grader can point at its own data.
Secrets are read from the environment / .env only — never hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements
    load_dotenv = None

# code/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
DEFAULT_DATASET = REPO_ROOT / "dataset"


def _load_env() -> None:
    """Load .env from the repo root if python-dotenv is available."""
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")


@dataclass
class Config:
    api_key: str
    base_url: str
    model: str
    dataset_dir: Path
    cache_dir: Path
    max_image_edge: int = 1280  # downscale long edge to bound tokens/payload
    request_timeout: float = 120.0
    max_retries: int = 4

    @property
    def claims_csv(self) -> Path:
        return self.dataset_dir / "claims.csv"

    @property
    def sample_csv(self) -> Path:
        return self.dataset_dir / "sample_claims.csv"

    @property
    def user_history_csv(self) -> Path:
        return self.dataset_dir / "user_history.csv"

    @property
    def evidence_csv(self) -> Path:
        return self.dataset_dir / "evidence_requirements.csv"


def build_config(dataset_dir: Path | None = None,
                 cache_dir: Path | None = None,
                 require_key: bool = True) -> Config:
    _load_env()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if require_key and not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY is not set. Add it to the .env file at the repo root "
            "(see .env.example)."
        )
    return Config(
        api_key=api_key,
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip(),
        model=os.getenv("GLM_MODEL", "z-ai/glm-5.1").strip(),
        dataset_dir=(dataset_dir or DEFAULT_DATASET).resolve(),
        cache_dir=(cache_dir or (CODE_DIR / ".cache")).resolve(),
    )
