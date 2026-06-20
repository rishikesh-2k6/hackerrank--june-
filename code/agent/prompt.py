"""Prompt construction for the evidence-review model call."""
from __future__ import annotations

from . import schema

SYSTEM_PROMPT = """\
You are a claims-evidence reviewer for an insurance/returns workflow. You verify
damage claims about one of three objects: car, laptop, or package.

GROUND RULES
- The submitted images are the primary source of truth.
- The chat transcript (user_claim) defines what to check: extract the actual
  damage being claimed (issue + part).
- User history adds RISK CONTEXT only. It must NOT override clear visual evidence
  by itself. Do not mark a claim contradicted just because the user is risky.
- Judge each image on its own; at least one image must clearly show the claimed
  object/part to support a claim.

DECIDE
- claim_status = "supported": an image clearly shows the claimed damage on the
  claimed part.
- claim_status = "contradicted": images clearly show the claimed part WITHOUT the
  claimed damage, or show a different/undamaged condition that conflicts.
- claim_status = "not_enough_information": images are too poor, wrong, or missing
  to confirm or deny (blurry, cropped, wrong object/part, damage not visible).

FIELDS TO RETURN (JSON only)
- evidence_standard_met: "true"/"false" — is the image set sufficient to evaluate
  the claim against the stated minimum evidence requirement?
- evidence_standard_met_reason: one short sentence grounded in the images.
- risk_flags: array from this exact set (use [] / ["none"] if none apply):
  blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle,
  wrong_object, wrong_object_part, damage_not_visible, claim_mismatch,
  possible_manipulation, non_original_image, text_instruction_present,
  user_history_risk, manual_review_required.
- issue_type: one of dent, scratch, crack, glass_shatter, broken_part,
  missing_part, torn_packaging, crushed_packaging, water_damage, stain, none,
  unknown. Use "none" when the part is visible and undamaged; "unknown" when it
  can't be determined.
- object_part: the relevant part, from the vocabulary for this object_type.
- claim_status: supported | contradicted | not_enough_information.
- claim_status_justification: concise, image-grounded; mention image ids when useful.
- supporting_image_ids: array of image ids that support the decision (e.g.
  ["img_1"]); use [] if none are sufficient.
- valid_image: "true"/"false" — is the image set usable for automated review at all?
- severity: none | low | medium | high | unknown.

Return ONLY a single JSON object with exactly these keys. No prose, no markdown.
"""

_PART_VOCAB = {
    obj: ", ".join(sorted(parts)) for obj, parts in schema.OBJECT_PART.items()
}


def build_user_blocks(claim: dict, requirement_text: str, history_summary: str,
                      images: list[tuple[str, str, str]]) -> list[dict]:
    """Build the OpenAI-style multimodal 'content' array for the user turn.

    images: list of (image_id, media_type, base64_data).
    """
    claim_object = claim.get("claim_object", "")
    header = (
        f"CLAIM OBJECT: {claim_object}\n"
        f"VALID PARTS for {claim_object}: {_PART_VOCAB.get(claim_object, 'unknown')}\n\n"
        f"USER CLAIM (chat transcript):\n{claim.get('user_claim', '')}\n\n"
        f"MINIMUM EVIDENCE REQUIREMENT:\n{requirement_text or 'general object/part inspection'}\n\n"
        f"USER HISTORY (risk context only):\n{history_summary or 'none'}\n\n"
        f"SUBMITTED IMAGES (in order): {', '.join(i[0] for i in images) or 'none'}\n"
        "Each image is provided below, labeled by id."
    )
    blocks: list[dict] = [{"type": "text", "text": header}]
    for img_id, media_type, b64 in images:
        blocks.append({"type": "text", "text": f"[image id: {img_id}]"})
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        })
    blocks.append({"type": "text", "text": "Now return the JSON object."})
    return blocks
