"""
robustness_analysis.py

Compares predictions under numeric perturbation or profile rewording
against the ORIGINAL (unperturbed) predictions for the same case_ids


Usage (numeric perturbation):
    python robustness_analysis.py --original-results results/structured_results_revalidated.jsonl --robustness-results results/structured_perturbed_results.jsonl --condition perturbation --original-subset-path data/processed/llm_eval_subset.csv --perturbed-subset-path data/processed/llm_eval_subset_perturbed_numeric.csv --show-examples 3

Usage (rewording):
    python robustness_analysis.py --original-results results/structured_results_revalidated.jsonl --robustness-results results/structured_reworded_results.jsonl --condition rewording --show-examples 3
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def majority_vote_per_case(df: pd.DataFrame) -> pd.Series:
    parsed = df[df["parse_success"]]
    return parsed.groupby("case_id")["label"].agg(lambda x: x.value_counts().idxmax())


def mean_confidence_per_case(df: pd.DataFrame) -> pd.Series:
    parsed = df[df["parse_success"]]
    return parsed.groupby("case_id")["confidence"].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--robustness-results", type=Path, required=True)
    parser.add_argument("--condition", choices=["perturbation", "rewording"], required=True)
    parser.add_argument("--original-subset-path", type=Path, default=None,
                         help="perturbation only: needed to compute delta magnitudes")
    parser.add_argument("--perturbed-subset-path", type=Path, default=None,
                         help="perturbation only: needed to compute delta magnitudes")
    parser.add_argument("--show-examples", type=int, default=3,
                         help="print this many flipped cases' raw responses for manual inspection")
    args = parser.parse_args()

    original_df = pd.read_json(args.original_results, lines=True)
    robust_df = pd.read_json(args.robustness_results, lines=True)

    original_label = majority_vote_per_case(original_df)
    robust_label = majority_vote_per_case(robust_df)
    original_conf = mean_confidence_per_case(original_df)
    robust_conf = mean_confidence_per_case(robust_df)

    attempted_robust_cases = set(robust_df["case_id"].unique())
    zero_parse_cases = attempted_robust_cases - set(robust_label.index)
    if zero_parse_cases:
        print(f"NOTE: {len(zero_parse_cases)} case(s) attempted in the robustness "
              f"run got zero successfully-parsed responses -- excluded from "
              f"comparison, not counted as 'no flip'.")

    common_cases = original_label.index.intersection(robust_label.index)
    if len(common_cases) == 0:
        raise SystemExit(
            "No case_id overlap between original and robustness results -- "
            "check you're comparing a pipeline's original run against its own "
            "robustness variant (same case_ids), not a different pipeline or "
            "a different eval subset."
        )
    print(f"=== Robustness comparison: {args.condition} ===")
    print(f"Cases compared: {len(common_cases)}")

    comparison = pd.DataFrame({
        "original_label": original_label.loc[common_cases],
        f"{args.condition}_label": robust_label.loc[common_cases],
        "original_confidence": original_conf.loc[common_cases],
        f"{args.condition}_confidence": robust_conf.loc[common_cases],
    })
    comparison["flipped"] = comparison["original_label"] != comparison[f"{args.condition}_label"]
    comparison["confidence_delta"] = (
        comparison[f"{args.condition}_confidence"] - comparison["original_confidence"]
    )

    n_flipped = int(comparison["flipped"].sum())
    print(f"\nLabel flip rate: {n_flipped}/{len(comparison)} ({n_flipped/len(comparison):.1%})")

    if n_flipped > 0:
        flip_directions = comparison[comparison["flipped"]].apply(
            lambda r: f"{r['original_label']} -> {r[f'{args.condition}_label']}", axis=1
        )
        print("Flip directions:")
        print(flip_directions.value_counts())
    
    else:
        print("No flips -- but check confidence_delta below before concluding "
              "full robustness; a model can hold its label while still being "
              "quite sensitive in HOW confident it states that label.")

    print(f"\nMean |confidence_delta|: {comparison['confidence_delta'].abs().mean():.1f}")
    print(f"Max |confidence_delta|: {comparison['confidence_delta'].abs().max():.1f}")

    if args.condition == "perturbation" and args.original_subset_path and args.perturbed_subset_path:
        orig_subset = pd.read_csv(args.original_subset_path).set_index("case_id")
        pert_subset = pd.read_csv(args.perturbed_subset_path).set_index("case_id")
        from perturbed_subset import PERTURBABLE_FEATURES

        deltas = {}
        for cid in common_cases:
            if cid not in orig_subset.index or cid not in pert_subset.index:
                continue
            deltas[cid] = sum(
                abs(pert_subset.loc[cid, f] - orig_subset.loc[cid, f])
                for f in PERTURBABLE_FEATURES if f in orig_subset.columns
                and pd.notna(orig_subset.loc[cid, f]) and pd.notna(pert_subset.loc[cid, f])
            )
        comparison = comparison.join(pd.Series(deltas, name="total_perturbation_magnitude"))

        print("\n--- Perturbation magnitude: flipped vs non-flipped cases ---")
        print(comparison.groupby("flipped")["total_perturbation_magnitude"].describe())

    if args.show_examples > 0 and n_flipped > 0:
        print(f"\n--- {min(args.show_examples, n_flipped)} flipped case example(s) ---")
        flipped_cases = comparison[comparison["flipped"]].index[: args.show_examples]
        for cid in flipped_cases:
            orig_row = original_df[(original_df["case_id"] == cid) & (original_df["parse_success"])].iloc[0]
            robust_row = robust_df[(robust_df["case_id"] == cid) & (robust_df["parse_success"])].iloc[0]
            print(f"\n[{cid}] {comparison.loc[cid, 'original_label']} -> {comparison.loc[cid, f'{args.condition}_label']}")
            print(f"ORIGINAL response: {orig_row['raw_response']}")
            print(f"{args.condition.upper()} response: {robust_row['raw_response']}")

    out_path = args.robustness_results.parent / f"robustness_analysis_{args.condition}.csv"
    comparison.to_csv(out_path)
    print(f"\nSaved full comparison to {out_path}")


if __name__ == "__main__":
    main()