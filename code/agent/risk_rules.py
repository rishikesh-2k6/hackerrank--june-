"""Deterministic post-processing: merge user-history risk, clamp to allowed values.

The model judges the images; this layer adds non-visual (history-based) risk and
guarantees every field is schema-legal. Keeping it deterministic makes the
history contribution auditable and the output reproducible.
"""
from __future__ import annotations

from . import schema


def _to_int(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def history_risk_flags(history: dict | None) -> list[str]:
    """Risk flags implied by the user's claim history (non-visual context)."""
    if not history:
        return []
    flags: list[str] = []
    hist_flags = (history.get("history_flags") or "").strip().lower()
    rejected = _to_int(history.get("rejected_claim"))
    manual = _to_int(history.get("manual_review_claim"))
    recent = _to_int(history.get("last_90_days_claim_count"))

    if hist_flags and hist_flags != "none":
        flags.append("user_history_risk")
    elif rejected >= 1 or recent >= 3 or manual >= 2:
        flags.append("user_history_risk")
    return flags


def build_output_row(claim: dict, model_out: dict, history: dict | None,
                     valid_image_ids: list[str]) -> dict:
    """Combine passthrough inputs + coerced model output + history risk into a row."""
    claim_object = claim.get("claim_object", "")

    evidence_met = schema.coerce_bool_str(model_out.get("evidence_standard_met"))
    valid_image = schema.coerce_bool_str(model_out.get("valid_image"))
    issue_type = schema.coerce_enum(model_out.get("issue_type"), schema.ISSUE_TYPE, "unknown")
    object_part = schema.coerce_object_part(model_out.get("object_part"), claim_object)
    claim_status = schema.coerce_enum(
        model_out.get("claim_status"), schema.CLAIM_STATUS, "not_enough_information")
    severity = schema.coerce_enum(model_out.get("severity"), schema.SEVERITY, "unknown")

    # Risk flags: model's visual flags + history-derived flags.
    model_flags = model_out.get("risk_flags", [])
    merged_flags = schema.coerce_risk_flags(model_flags)
    flag_list = [] if merged_flags == "none" else merged_flags.split(";")
    for hf in history_risk_flags(history):
        if hf not in flag_list:
            flag_list.append(hf)

    # Operational rule: escalate to manual review where automation is least safe.
    needs_manual = (
        claim_status == "not_enough_information"
        or evidence_met == "false"
        or ("user_history_risk" in flag_list and claim_status == "contradicted")
    )
    if needs_manual and "manual_review_required" not in flag_list:
        flag_list.append("manual_review_required")

    risk_flags = ";".join(flag_list) if flag_list else "none"

    # Supporting ids: keep only ids that belong to this claim's images.
    supporting = schema.coerce_supporting_ids(model_out.get("supporting_image_ids"))
    if supporting != "none":
        kept = [i for i in supporting.split(";") if i in valid_image_ids]
        supporting = ";".join(kept) if kept else "none"

    return {
        "user_id": claim.get("user_id", ""),
        "image_paths": claim.get("image_paths", ""),
        "user_claim": claim.get("user_claim", ""),
        "claim_object": claim_object,
        "evidence_standard_met": evidence_met,
        "evidence_standard_met_reason": str(model_out.get("evidence_standard_met_reason", "")).strip(),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": str(model_out.get("claim_status_justification", "")).strip(),
        "supporting_image_ids": supporting,
        "valid_image": valid_image,
        "severity": severity,
    }


def fallback_row(claim: dict, history: dict | None, reason: str) -> dict:
    """Schema-legal row used when the image set is unusable or the call failed."""
    flag_list = ["damage_not_visible"]
    for hf in history_risk_flags(history):
        if hf not in flag_list:
            flag_list.append(hf)
    if "manual_review_required" not in flag_list:
        flag_list.append("manual_review_required")
    return {
        "user_id": claim.get("user_id", ""),
        "image_paths": claim.get("image_paths", ""),
        "user_claim": claim.get("user_claim", ""),
        "claim_object": claim.get("claim_object", ""),
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason,
        "risk_flags": ";".join(flag_list),
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": reason,
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }
