"""
applicant_profile_reworded.py

Alternate profile-text generator producing genuinely different surface
wording for the SAME underlying facts as applicant_profile.py, for
robustness testing.

"""
import hashlib
import random

import pandas as pd

from applicant_profile import (
    FEATURE_DESCRIPTIONS,
    MAXDELQ_12M_LABELS,
    MAXDELQ_EVER_LABELS,
    SENTINEL_PHRASING,
)

GENERAL_TEMPLATES = [
    lambda label, value, unit_suffix: f"The applicant's {label} is {value}{unit_suffix}.",
    lambda label, value, unit_suffix: f"For {label}, the recorded value is {value}{unit_suffix}.",
    lambda label, value, unit_suffix: f"{label[0].upper()}{label[1:]}: {value}{unit_suffix}.",
    lambda label, value, unit_suffix: f"With respect to {label}, this applicant shows {value}{unit_suffix}.",
]

PERCENT_TEMPLATES = [
    lambda label, value: f"The applicant's {label} is {value:.1f}%.",
    lambda label, value: f"For {label}, the figure is {value:.1f} percent.",
    lambda label, value: f"{label[0].upper()}{label[1:]}: {value:.1f}%.",
]

SENTINEL_TEMPLATES = [
    lambda label, phrase: f"For {label}, {phrase}.",
    lambda label, phrase: f"Regarding {label}, {phrase}.",
    lambda label, phrase: f"{label[0].upper()}{label[1:]}: {phrase}.",
]

MAXDELQ_TEMPLATES = [
    lambda label, value: f"The applicant's {label} is: {value}.",
    lambda label, value: f"Regarding {label}, the recorded status is: {value}.",
    lambda label, value: f"{label[0].upper()}{label[1:]} status: {value}.",
]


def _case_rng(case_id: str) -> random.Random:
    seed = int(hashlib.sha256(str(case_id).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def _describe_maxdelq_reworded(feature_name: str, value: float, rng: random.Random) -> str:
    mapping = MAXDELQ_12M_LABELS if feature_name == "MaxDelq2PublicRecLast12M" else MAXDELQ_EVER_LABELS
    label, _ = FEATURE_DESCRIPTIONS[feature_name]
    code = int(value)
    template = rng.choice(MAXDELQ_TEMPLATES)
    if code in mapping:
        return template(label, mapping[code])

    return (
        f"The applicant's {label} is coded {code}, which does not match "
        f"any documented category for this field (this is unexpected and "
        f"worth investigating)."
    )


def _describe_feature_reworded(feature_name: str, row: pd.Series, indicator_cols: list[str], rng: random.Random) -> str:
    label, unit = FEATURE_DESCRIPTIONS[feature_name]

    fired_indicator = None
    for ind_col in indicator_cols:
        if ind_col.startswith(f"{feature_name}_missing_") and row.get(ind_col, 0) == 1:
            fired_indicator = ind_col[len(f"{feature_name}_missing_"):]
            break

    if fired_indicator is not None:
        phrase = SENTINEL_PHRASING.get(fired_indicator, "this measure is not available")
        template = rng.choice(SENTINEL_TEMPLATES)
        return template(label, phrase)

    value = row[feature_name]
    if pd.isna(value):
        return f"For {label}, no data is available for this applicant."

    if feature_name in ("MaxDelq2PublicRecLast12M", "MaxDelqEver"):
        return _describe_maxdelq_reworded(feature_name, value, rng)

    if unit == "%":
        template = rng.choice(PERCENT_TEMPLATES)
        return template(label, value)

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    unit_suffix = f" {unit}" if unit else ""
    template = rng.choice(GENERAL_TEMPLATES)
    return template(label, value, unit_suffix)


def build_profile_text_reworded(row: pd.Series, metadata: dict) -> str:
    case_id = row.get("case_id", "")
    rng = _case_rng(case_id)
    indicator_cols = metadata["indicator_cols"]

    feature_names = [f for f in FEATURE_DESCRIPTIONS if f in row.index or f in FEATURE_DESCRIPTIONS]
    order = list(FEATURE_DESCRIPTIONS.keys())
    rng.shuffle(order)

    sentences = [_describe_feature_reworded(f, row, indicator_cols, rng) for f in order]
    return " ".join(sentences)