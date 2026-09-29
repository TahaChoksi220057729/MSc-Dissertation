"""
analyze_results.py



Usage:
    python analyze_results.py --results-path results/stock_results.jsonl
"""

import argparse
from pathlib import Path

import pandas as pd


def compute_consistency(df: pd.DataFrame) -> pd.Series:
    """

    """
    def _consistency(case_df):
        n = len(case_df)
        parsed_labels = case_df.loc[case_df["parse_success"], "label"]
        if len(parsed_labels) == 0:
            return 0.0
        max_count = parsed_labels.value_counts().iloc[0]
        return max_count / n

    return df.groupby("case_id").apply(_consistency, include_groups=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--subset-path", type=Path, default=None,
                         help="optional -- cross-checks against the planned eval subset to report completeness")
    parser.add_argument("--show-examples", type=int, default=3,
                         help="print this many false-negative raw responses for manual inspection (0 to disable)")
    args = parser.parse_args()

    df = pd.read_json(args.results_path, lines=True)
    n_total = len(df)
    n_parsed = df["parse_success"].sum()

    print(f"=== {args.results_path} ===")

    if args.subset_path is not None:
        planned = pd.read_csv(args.subset_path)
        n_planned_cases = len(planned)
        n_seen_cases = df["case_id"].nunique()
        repeats_seen = df.groupby("case_id").size()
        n_repeats = repeats_seen.mode().iloc[0] if len(repeats_seen) else 0
        n_expected = n_planned_cases * n_repeats
        if n_seen_cases < n_planned_cases or n_total < n_expected:
            print(
                f"INCOMPLETE RUN: {n_seen_cases}/{n_planned_cases} cases started, "
                f"{n_total}/{n_expected} generations present (at {n_repeats} repeats/case). "
                f"Numbers below are a partial sample -- re-run the pipeline cell in "
                f"Colab to continue; resumability means it picks up where it left off."
            )
        else:
            print(f"Run appears complete: {n_seen_cases}/{n_planned_cases} cases, {n_total}/{n_expected} generations.")

    print(f"Total records: {n_total}, parsed: {n_parsed} ({n_parsed/n_total:.1%})")

    check_cols = ["is_valid_json", "label_valid", "confidence_valid", "explanation_grounded", "no_contradiction"]
    available_checks = [c for c in check_cols if c in df.columns]
    if available_checks:
        print("\n--- Validation check breakdown (structured pipeline only) ---")
        for col in available_checks:
            n_pass = (df[col] == True).sum()
            n_fail = (df[col] == False).sum()
            n_skipped = df[col].isna().sum()
            print(f"{col:22s} pass={n_pass:4d}  fail={n_fail:4d}  not-evaluated={n_skipped:4d}")
        print(
            "'not-evaluated' means an earlier check in the chain already failed "
        )

    print("\n--- Label distribution (parsed only) ---")
    print(df[df["parse_success"]]["label"].value_counts())
    print(
        "If this is heavily skewed toward one label regardless of the "
        "true_label distribution below, that's a sign the model may be "
        "defaulting to one answer"
    )

    print("\n--- True label distribution (for comparison) ---")
    # true_label is 0/1 per data_pipeline.py's encoding: 1 = Bad, 0 = Good
    true_label_str = df["true_label"].map({1: "Bad", 0: "Good"})
    print(true_label_str.value_counts())

    print("\n--- Confidence distribution (parsed only) ---")
    conf = df[df["parse_success"]]["confidence"]
    print(conf.describe())
    if conf.std() < 5:
        print(
            f"WARNING: confidence std is only {conf.std():.2f} -- the model "
            f"may be returning a near-constant confidence value regardless "
            f"of the case, which would technically parse successfully every "
            f"time while carrying little real information. Worth checking."
        )

    print("\n--- Per-case consistency (proposal's formula) ---")
    consistency = compute_consistency(df)
    print(consistency.describe())
    low_consistency = consistency[consistency < 0.6]

    zero_valid = df.groupby("case_id").apply(
        lambda g: (g["parse_success"] == False).all(), include_groups=False
    )
    n_zero_valid = zero_valid.sum()
    n_genuinely_split = len(low_consistency) - n_zero_valid
    print(f"\nCases with consistency < 0.6: {len(low_consistency)} total")
    print(f"  - of which NEVER produced a valid label in any run: {n_zero_valid}")
    print(f"  - of which gave genuinely different labels across runs: {n_genuinely_split}")
    if n_zero_valid > 0:
        print(
            "The first group is a PARSING reliability problem, not necessarily "
            "a label-instability problem -- worth checking the validation "
            "breakdown above to see which check those cases were failing."
        )
    if len(low_consistency) == 0 and conf.std() < 5:
        print(
            "No unstable cases AND low confidence variance together is worth "
            "a closer look -- could mean the model is giving very similar "
            "answers regardless of input, not that it's genuinely reliable."
        )

    print("\n--- Confusion matrix (majority vote per case vs true label) ---")
    majority_vote = df[df["parse_success"]].groupby("case_id")["label"].agg(
        lambda x: x.value_counts().idxmax()
    )
    true_per_case = df.groupby("case_id")["true_label"].first().map({1: "Bad", 0: "Good"})
    comparison = pd.DataFrame({"predicted": majority_vote, "true": true_per_case}).dropna()
    print(pd.crosstab(comparison["true"], comparison["predicted"], margins=True))

    false_negatives = comparison[(comparison["true"] == "Bad") & (comparison["predicted"] == "Good")]
    print(
        f"\nFalse negatives (true Bad, predicted Good): {len(false_negatives)} "
        f"of {(comparison['true'] == 'Bad').sum()} true-Bad cases."
    )

    if args.show_examples > 0 and len(false_negatives) > 0:
        print(f"\n--- {min(args.show_examples, len(false_negatives))} false-negative example(s) (raw response, seed 0) ---")
        for case_id in false_negatives.index[: args.show_examples]:
            example = df[(df["case_id"] == case_id) & (df["seed"] == 0)]
            if len(example):
                print(f"\n[{case_id}]")
                print(example.iloc[0]["raw_response"])

    print("\n--- Quick accuracy check (majority vote per case, parsed only) ---")
    accuracy = (comparison["predicted"] == comparison["true"]).mean()
    print(f"Majority-vote accuracy: {accuracy:.1%} ({len(comparison)} cases)")


if __name__ == "__main__":
    main()