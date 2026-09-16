"""
structured_pipeline.py

The "structured" Qwen3 pipeline: adds a feature glossary, a fixed decision
rubric, fixed few-shot examples, a JSON output schema, and validation --
in contrast to stock_pipeline.py's deliberately minimal design. Per your
proposal, validation checks: (1) valid JSON, (2) permitted label,
(3) numeric confidence in range, (4) explanation grounded in real input
features, (5) explanation doesn't contradict the predicted label.

Design decisions (document these in your Method chapter, in your own
words):

1. The rubric explicitly instructs checking BOTH positive and negative
   indicators before concluding, rather than just listing what "Good"
   and "Bad" mean. This is directly motivated by an observed finding from
   the stock pipeline pilot run: it showed a systematic leniency bias,
   citing genuine positive indicators while apparently not weighing
   negative ones, producing a 76% false-negative rate on true-Bad cases
   despite input-grounded (not generic) explanations. Documenting this as
   the motivation for the rubric's design is a legitimate, evidence-based
   methodological decision, not an arbitrary prompt-engineering choice.

2. Check 4 ("explanation refers only to real input features") is
   implemented as NUMERIC grounding, not keyword/topic matching: every
   standalone number mentioned in key_factors/explanation is cross-checked
   against the applicant's actual feature values (within a small
   tolerance). This is narrower than the proposal's literal wording but
   far more reliable -- generic keyword matching against paraphrased
   natural language produces too many false positives to be useful, while
   a fabricated or misremembered number is a precise, checkable signal
   (this is exactly the kind of error the stock-pipeline pilot's case_0014
   showed by hand -- a specific, wrong number-to-meaning claim).

3. Check 5 (label contradiction) is a keyword-based heuristic, not a
   semantic check -- it flags strong opposite-valence language relative to
   the stated label. This is a real limitation: it will miss subtler
   contradictions and could false-positive on legitimate hedged language.
   Worth spot-checking flagged AND unflagged cases by hand periodically,
   the same way case_0014 was found -- automated checks here are a
   screening aid, not a substitute for reading actual output.

4. The output field is named "key_factors", not "main_risk_factors" as in
   the proposal's prose -- "risk factors" reads oddly for a Good-labeled
   case's supporting evidence. This is a naming clarification only; it
   still corresponds to the same proposal requirement (the factors driving
   the classification).
"""

import json
import re
from pathlib import Path

from applicant_profile import FEATURE_DESCRIPTIONS, MAXDELQ_12M_LABELS, MAXDELQ_EVER_LABELS

RUBRIC = """A "Bad" classification means the applicant was 90 days past due or worse at least once over 24 months after opening the account. A "Good" classification means the applicant was never more than 90 days overdue in that period (FICO, 2018).

Before reaching a conclusion, systematically consider BOTH of the following categories -- do not rely solely on positive indicators:
- Indicators of GOOD credit management: no delinquencies, high percentage of trades never delinquent, low revolving/installment burden, long credit history, low inquiry counts.
- Indicators of RISK: any history of 60+ or 90+ day delinquency, derogatory public records, unknown delinquency status, high revolving burden, frequent recent inquiries, short credit history.

A classification should not be based only on the presence of positive indicators -- explicitly check for risk indicators before concluding "Good"."""


def build_feature_glossary() -> str:
    lines = ["Feature glossary:"]
    for name, (label, unit) in FEATURE_DESCRIPTIONS.items():
        unit_str = f" (unit: {unit})" if unit else ""
        lines.append(f"- {name}: {label}{unit_str}")
    return "\n".join(lines)


def load_few_shot_block(few_shot_path: Path) -> str:
    with open(few_shot_path) as f:
        examples = json.load(f)
    blocks = []
    for ex in examples:
        blocks.append(
            f"Applicant profile:\n{ex['profile_text']}\n\n"
            f"Response:\n{json.dumps(ex['ideal_response'], indent=2)}"
        )
    return "\n\n---\n\n".join(blocks)


JSON_INSTRUCTIONS = """Respond with ONLY a single JSON object, no other text, in exactly this schema:
{
  "label": "Good" or "Bad",
  "confidence": <number 0-100>,
  "key_factors": [<short phrase>, <short phrase>, ...],
  "explanation": "<1-3 sentences>",
  "uncertainty_reason": "<why you are or are not confident>"
}"""


def build_structured_prompt(profile_text: str, few_shot_path: Path) -> tuple[str, str]:
    glossary = build_feature_glossary()
    few_shot_block = load_few_shot_block(few_shot_path)

    system_prompt = (
        "You are a financial assistant helping to assess credit risk for "
        "loan applications, following a specific decision rubric.\n\n"
        f"{glossary}\n\nDecision rubric:\n{RUBRIC}\n\n{JSON_INSTRUCTIONS}\n\n"
        f"Examples:\n\n{few_shot_block}"
    )
    user_prompt = f"Now classify this applicant:\n\n{profile_text}"
    return system_prompt, user_prompt


_NUMBER_RE = re.compile(r"-?\d+\.?\d*")
_CONTRADICTION_TERMS_BAD = ["severe risk", "high risk", "significant delinquency",
                            "poor credit", "substantial risk", "concerning"]
_CONTRADICTION_TERMS_GOOD = ["strong credit", "low risk", "reliable", "excellent",
                             "minimal risk", "no significant risk"]


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def _extract_vocabulary_numbers() -> set:
    """
    Numbers appearing in the glossary/rubric/MaxDelq-category-label text
    (e.g. "90+ days delinquent", "24 months", "30 days delinquent") are
    terminology, not claims about a specific applicant. Without excluding
    these, referencing a feature by its own glossary/category description
    (exactly what the few-shot examples model, and what a correctly-
    behaving response legitimately does) gets false-flagged as fabricated.

    This was found the hard way: a real run flagged "30" as ungrounded on
    every single generation for several applicants whose MaxDelq2PublicRecLast12M
    and MaxDelqEver codes both happened to map to "30 days delinquent" --
    the model was accurately quoting its own input profile text, not
    inventing anything. An earlier version of this function's docstring
    already listed 30/120 among "known" terminology numbers, but the
    implementation never actually pulled MAXDELQ_12M_LABELS/
    MAXDELQ_EVER_LABELS text in -- the comment described an intended scope
    the code didn't cover. Fixed by actually including it below.

    Known remaining limitation: a fabricated value that happens to
    coincide with one of these vocabulary numbers (90, 60, 30, 120, 24,
    7, ...) would still slip through undetected -- a real trade-off,
    accepted because the alternative (constant false positives) makes the
    check useless rather than imperfect.
    """
    text = (
        RUBRIC + " "
        + " ".join(label for label, _ in FEATURE_DESCRIPTIONS.values()) + " "
        + " ".join(MAXDELQ_12M_LABELS.values()) + " "
        + " ".join(MAXDELQ_EVER_LABELS.values())
    )
    numbers = _extract_numbers(text)
    return {round(n, 1) for n in numbers} | {round(n) for n in numbers}


_VOCAB_NUMBERS = _extract_vocabulary_numbers()


def _row_numeric_values(row, feature_cols: list[str]) -> set[float]:
    values = set()
    for col in feature_cols:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and val != val):  # not NaN
            values.add(round(float(val), 1))
            values.add(round(float(val)))  # also allow integer-rounded match
    return values


def _check_numeric_grounding(parsed: dict, row, metadata: dict) -> tuple[bool, list[float]]:
    text = " ".join(parsed.get("key_factors", []) or []) + " " + (parsed.get("explanation") or "")
    mentioned = _extract_numbers(text)
    conf = parsed.get("confidence")
    if conf is not None:
        mentioned = [n for n in mentioned if n != conf]

    real_values = _row_numeric_values(row, metadata["feature_cols"])
    allowed = real_values | _VOCAB_NUMBERS
    ungrounded = [n for n in mentioned if round(n, 1) not in allowed and round(n) not in allowed]
    return len(ungrounded) == 0, ungrounded


def _check_label_contradiction(label: str, explanation: str) -> bool:
    """Returns True if a likely contradiction is detected (heuristic only)."""
    if not explanation:
        return False
    exp_lower = explanation.lower()
    if label == "Good" and any(term in exp_lower for term in _CONTRADICTION_TERMS_BAD):
        return True
    if label == "Bad" and any(term in exp_lower for term in _CONTRADICTION_TERMS_GOOD):
        return True
    return False


def parse_structured_response(raw_text: str, row=None, metadata: dict = None) -> dict:
    result = {
        "label": None, "confidence": None, "key_factors": None,
        "explanation": None, "uncertainty_reason": None,
        "is_valid_json": False, "label_valid": False, "confidence_valid": False,
        "explanation_grounded": None, "no_contradiction": None,
        "ungrounded_numbers": None, "parse_success": False,
    }

    cleaned = _strip_json_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return result
    if not isinstance(parsed, dict):
        return result
    result["is_valid_json"] = True

    label = parsed.get("label")
    if isinstance(label, str) and label.strip().capitalize() in ("Good", "Bad"):
        result["label"] = label.strip().capitalize()
        result["label_valid"] = True

    confidence = parsed.get("confidence")
    if isinstance(confidence, (int, float)) and 0 <= confidence <= 100:
        result["confidence"] = float(confidence)
        result["confidence_valid"] = True

    result["key_factors"] = parsed.get("key_factors")
    result["explanation"] = parsed.get("explanation")
    result["uncertainty_reason"] = parsed.get("uncertainty_reason")

    if row is not None and metadata is not None:
        grounded, ungrounded = _check_numeric_grounding(parsed, row, metadata)
        result["explanation_grounded"] = grounded
        result["ungrounded_numbers"] = ungrounded

    if result["label_valid"]:
        contradiction = _check_label_contradiction(result["label"], result["explanation"] or "")
        result["no_contradiction"] = not contradiction

    result["parse_success"] = all([
        result["is_valid_json"],
        result["label_valid"],
        result["confidence_valid"],
        result["explanation"] is not None,
        result["explanation_grounded"] is not False,
        result["no_contradiction"] is not False,
    ])
    return result