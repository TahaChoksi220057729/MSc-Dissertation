"""
select_few_shot_examples.py

Selects one clear "Good" example, one clear "Bad" example, and one
genuinely borderline example from the TRAINING split, and auto-generates 
their ideal structured responses. Run locally once; structured_pipeline.py 
loads the saved output as a FIXED block for every prompt in the run.

"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from applicant_profile import build_profile_text, _describe_feature

CORE_FACTORS = [
    "NumTrades90Ever2DerogPubRec",
    "NumTrades60Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "ExternalRiskEstimate",
]


def build_ideal_response(
    row: pd.Series, metadata: dict, true_label_str: str,
    confidence: float = 90, uncertainty_reason: str = None,
) -> dict:
    indicator_cols = metadata["indicator_cols"]
    key_factors = [_describe_feature(f, row, indicator_cols) for f in CORE_FACTORS]

    pct_never_delq = row.get("PercentTradesNeverDelq")
    num_90 = row.get("NumTrades90Ever2DerogPubRec")
    num_60 = row.get("NumTrades60Ever2DerogPubRec")
    has_tension = (
        pct_never_delq is not None and pct_never_delq >= 99.5
        and (
            (num_90 is not None and num_90 > 0)
            or (num_60 is not None and num_60 > 0)
        )
    )
    clarifying_note = (
        " Note that a high percentage of trades never delinquent does not "
        "rule out risk: a trade can be flagged in the 60+/90+-day-"
        "delinquent-or-derogatory-public-record count due to an associated "
        "public record rather than a late payment on that specific trade, "
        "so these two figures are not contradictory."
        if has_tension else ""
    )

    explanation = (
        f"Based on the applicant's record: {' '.join(key_factors)}"
        f"{clarifying_note} This pattern is consistent with a "
        f"{true_label_str} classification."
    )

    if uncertainty_reason is None:
        uncertainty_reason = (
            "Low uncertainty: the applicant's indicators are consistently "
            "aligned with a single classification with no significant "
            "conflicting or missing signals."
        )

    return {
        "label": true_label_str,
        "confidence": confidence,
        "key_factors": key_factors,
        "explanation": explanation,
        "uncertainty_reason": uncertainty_reason,
    }


BORDERLINE_UNCERTAINTY_TEXT = (
    "Moderate uncertainty: the applicant's indicators include a mix of "
    "positive and risk-related signals rather than a uniformly clear "
    "pattern, so this assessment is less certain than a case with fully "
    "consistent indicators."
)


def select_example(train: pd.DataFrame, target_col: str, proba, true_class: int) -> pd.Series:
    train = train.copy()
    train["_proba"] = proba
    cls_rows = train[train[target_col] == true_class]
    # want HIGH confidence in the CORRECT direction: proba close to true_class
    if true_class == 1:
        cls_rows = cls_rows.sort_values("_proba", ascending=False)
    else:
        cls_rows = cls_rows.sort_values("_proba", ascending=True)
    return cls_rows.iloc[0]


def select_borderline_example(train: pd.DataFrame, proba) -> pd.Series:
    train = train.copy()
    train["_proba"] = proba
    train["_dist_from_half"] = (train["_proba"] - 0.5).abs()
    return train.sort_values("_dist_from_half").iloc[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    with open(args.processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    train = pd.read_csv(args.processed_dir / "train.csv")

    model = joblib.load(args.model_path)
    feature_cols = metadata["feature_cols"] + metadata["indicator_cols"]
    proba = model.predict_proba(train[feature_cols])[:, 1]
    target_col = metadata["target_col"]

    good_row = select_example(train, target_col, proba, true_class=0)
    bad_row = select_example(train, target_col, proba, true_class=1)
    borderline_row = select_borderline_example(train, proba)
    borderline_label_str = "Bad" if borderline_row[target_col] == 1 else "Good"
    borderline_confidence = round(max(borderline_row["_proba"], 1 - borderline_row["_proba"]) * 100)

    examples = []
    for row, label_str, confidence, uncertainty_reason in [
        (good_row, "Good", 90, None),
        (bad_row, "Bad", 90, None),
        (borderline_row, borderline_label_str, borderline_confidence, BORDERLINE_UNCERTAINTY_TEXT),
    ]:
        profile_text = build_profile_text(row, metadata)
        ideal_response = build_ideal_response(row, metadata, label_str, confidence, uncertainty_reason)
        examples.append({"profile_text": profile_text, "ideal_response": ideal_response})

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Wrote {len(examples)} few-shot examples -> {args.output_path}")
    for ex in examples:
        print(f"\n--- {ex['ideal_response']['label']} example ---")
        print(ex["profile_text"][:200] + "...")
        print(json.dumps(ex["ideal_response"], indent=2))

if __name__ == "__main__":
    main()