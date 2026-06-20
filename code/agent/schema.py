"""Output schema, allowed-value vocabularies, and coercion helpers.

All vocabularies come straight from ``problem_statement.md``. The model is asked
to emit these fields; this module is the single source of truth for validating
and clamping whatever comes back so every output row is schema-legal.
"""
from __future__ import annotations

# Exact output column order required by problem_statement.md.
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# The fields the model is responsible for producing (the rest are passthrough inputs).
MODEL_FIELDS = [
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

OBJECT_PART = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    },
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}


def coerce_enum(value, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return default


def coerce_bool_str(value) -> str:
    """Normalize a truthy/falsey value to the string 'true'/'false'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower()
    return "false"


def coerce_object_part(value, claim_object: str) -> str:
    allowed = OBJECT_PART.get(claim_object, {"unknown"})
    return coerce_enum(value, allowed, "unknown")


def coerce_risk_flags(value) -> str:
    """Accept a list or ';'-joined string; keep only allowed flags, dedupe."""
    if isinstance(value, str):
        items = [v.strip() for v in value.split(";")]
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = []
    seen: list[str] = []
    for item in items:
        if item in RISK_FLAGS and item != "none" and item not in seen:
            seen.append(item)
    return ";".join(seen) if seen else "none"


def coerce_supporting_ids(value) -> str:
    if isinstance(value, str):
        items = [v.strip() for v in value.split(";")]
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = []
    seen: list[str] = []
    for item in items:
        if item and item.lower() != "none" and item not in seen:
            seen.append(item)
    return ";".join(seen) if seen else "none"
