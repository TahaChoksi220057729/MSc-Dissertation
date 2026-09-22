"""
investigate_grounding_failures.py

Looks for what distinguishes cases where the structured pipeline NEVER
produces a grounded explanation (fails on all 5 runs) from cases where it
succeeds at least once. Grounding failures were found to be heavily
concentrated -- 16 cases account for ~95% of all failures -- which
suggests something specific about those applicant profiles, not random
noise scattered evenly across all 100 cases.


Usage:
    python investigate_grounding_failures.py --results-path results/structured_results.jsonl --subset-path data/processed/llm_eval_subset.csv --processed-dir data/processed [--baseline-predictions-path results/baseline_per_case_predictions.csv]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def identify_always_fail_cases(df: pd.DataFrame) -> set:
    grouped = df.groupby("case_id")["parse_success"].agg(lambda x: (~x).all())
    return set(grouped[grouped].index)


def closest_real_value_distance(number, row, feature_cols):
    real_values = []
    for col in feature_cols:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            real_values.append(float(val))
    if not real_values:
        return None
    return min(abs(number - v) for v in real_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--baseline-predictions-path", type=Path, default=None,
                         help="optional -- results/baseline_per_case_predictions.csv")
    args = parser.parse_args()

    df = pd.read_json(args.results_path, lines=True)
    subset = pd.read_csv(args.subset_path)
    with open(args.processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"] + metadata["indicator_cols"]
    indicator_cols = metadata["indicator_cols"]

    always_fail = identify_always_fail_cases(df)
    print(f"Always-fail cases: {len(always_fail)} of {df['case_id'].nunique()}")

    subset = subset.set_index("case_id")
    subset["always_fails_grounding"] = subset.index.isin(always_fail)

    print("\n--- 1. Sentinel-indicator count (missing/special values in the profile) ---")
    subset["n_sentinel_flags"] = subset[indicator_cols].sum(axis=1)
    print(subset.groupby("always_fails_grounding")["n_sentinel_flags"].describe())

    print("\n--- 2. Borderline-ness (baseline model's own uncertainty) ---")
    if args.baseline_predictions_path and args.baseline_predictions_path.exists():
        preds = pd.read_csv(args.baseline_predictions_path)
        merged = subset.merge(preds[["case_id", "xgboost_proba"]], left_index=True, right_on="case_id", how="left")
        merged["dist_from_half"] = (merged["xgboost_proba"] - 0.5).abs()
        print(merged.groupby("always_fails_grounding")["dist_from_half"].describe())
        print("(Smaller distance = more borderline per the baseline model.)")
    else:
        print(
            "Skipped -- pass --baseline-predictions-path "
            "results/baseline_per_case_predictions.csv (from "
            "score_baselines_on_eval_subset.py) to include this check."
        )

    print("\n--- 3. Confidence: grounding-failed runs vs grounding-passed runs (all 500) ---")
    print(df.groupby("explanation_grounded")["confidence"].describe())

    print("\n--- 4. Distance of fabricated numbers from the NEAREST real value in the profile ---")
    distances = []
    for _, row in df[df["explanation_grounded"] == False].iterrows():
        if row["case_id"] not in subset.index:
            continue
        profile_row = subset.loc[row["case_id"]]
        numbers = row.get("ungrounded_numbers") or []
        for n in numbers:
            d = closest_real_value_distance(n, profile_row, feature_cols)
            if d is not None:
                distances.append(d)
    if distances:
        distances = pd.Series(distances)
        print(distances.describe())
        print(
            "\nSmall distances (say, within a few units) suggest misremembering "
            "or paraphrasing a real value with reduced precision. Large "
            "distances suggest the model is stating numbers with no clear "
            "basis in this applicant's actual data -- a more serious form of "
            "fabrication, worth distinguishing in your explanation-quality "
            "analysis."
        )
    else:
        print("No ungrounded numbers found to analyze.")

    print("\n--- Sample of always-fail case profiles, for manual inspection ---")
    for case_id in list(always_fail)[:3]:
        row = subset.loc[case_id]
        example = df[df["case_id"] == case_id].iloc[0]
        print(f"\n[{case_id}] sentinel_flags={int(row['n_sentinel_flags'])}")
        print(f"Ungrounded numbers flagged: {example.get('ungrounded_numbers')}")


if __name__ == "__main__":
    main()
