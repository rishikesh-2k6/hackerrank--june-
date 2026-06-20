"""Loading and parsing of the dataset CSVs and image files."""
from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def read_claims(path: Path) -> list[dict]:
    """Read a claims CSV (input rows) into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path: Path) -> dict[str, dict]:
    """Map user_id -> history row."""
    history: dict[str, dict] = {}
    if not path.exists():
        return history
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history[row["user_id"]] = row
    return history


def load_evidence_requirements(path: Path) -> dict[str, list[dict]]:
    """Map claim_object -> list of requirement rows (includes the 'all' bucket)."""
    reqs: dict[str, list[dict]] = {}
    if not path.exists():
        return reqs
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reqs.setdefault(row["claim_object"], []).append(row)
    return reqs


def requirements_for(reqs: dict[str, list[dict]], claim_object: str) -> list[dict]:
    """Requirements that apply to a claim: object-specific plus the 'all' bucket."""
    return reqs.get("all", []) + reqs.get(claim_object, [])


def parse_image_paths(image_paths: str) -> list[str]:
    """Split the ';'-separated image_paths field into individual relative paths."""
    return [p.strip() for p in image_paths.split(";") if p.strip()]


def image_id(rel_path: str) -> str:
    """Image id = filename without extension, e.g. 'img_1'."""
    return Path(rel_path).stem


def encode_image(abs_path: Path, max_edge: int) -> tuple[str, str] | None:
    """Return (media_type, base64_data) for an image, downscaled if Pillow is present.

    Returns None if the file is missing/unreadable.
    """
    if not abs_path.exists():
        return None
    try:
        raw = abs_path.read_bytes()
    except OSError:
        return None

    if Image is not None:
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                if max(im.size) > max_edge:
                    scale = max_edge / max(im.size)
                    new_size = (round(im.width * scale), round(im.height * scale))
                    im = im.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
                return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode()
        except Exception:
            pass  # fall through to raw bytes

    media = "image/png" if abs_path.suffix.lower() == ".png" else "image/jpeg"
    return media, base64.standard_b64encode(raw).decode()
