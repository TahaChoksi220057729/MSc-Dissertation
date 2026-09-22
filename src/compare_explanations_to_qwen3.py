"""
compare_explanations_to_qwen3.py

Compares the structured pipeline's stated key_factors against SHAP and
LIME reference explanations from the classical baselines, for the same
cases -- the actual "does structure improve explanation grounding"
analysis this project has been building toward.


Usage:
    python compare_explanations_to_qwen3.py --structured-results results/structured_results_revalidated.jsonl --subset-path data/processed/llm_eval_subset.csv --shap-lime-path results/shap_lime_top_features.json --output-path results/explanation_comparison.json --show-examples 3
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from applicant_profile import FEATURE_DESCRIPTIONS

_SORTED_FEATURES = sorted(FEATURE_DESCRIPTIONS.items(), key=lambda kv: -len(kv[1][0]))


def map_factor_to_feature(factor_text: str) -> str | None:
    factor_lower = factor_text.lower()
    for feature_name, (label, unit) in _SORTED_FEATURES:
        if label.lower() in factor_lower:
            return feature_name
    return None


def map_factors(key_factors) -> tuple[set, int]:
    if not key_factors:
        return set(), 0
    mapped = set()
    n_unmapped = 0
    for factor in key_factors:
        feature = map_factor_to_feature(factor)
        if feature:
            mapped.add(feature)
        else:
            n_unmapped += 1
    return mapped, n_unmapped


def compute_grounding_precision(cited_features: set, reference_top_k: list) -> float:
    if not cited_features:
        return None
    return len(cited_features & set(reference_top_k)) / len(cited_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-results", type=Path, required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--shap-lime-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--show-examples", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_json(args.structured_results, lines=True)
    subset = pd.read_csv(args.subset_path).set_index("case_id")
    with open(args.shap_lime_path) as f:
        ref_data = json.load(f)

    parsed = df[df["parse_success"]].copy()
    mapped_results = parsed["key_factors"].apply(map_factors)
    parsed["mapped_features"] = mapped_results.apply(lambda x: x[0])
    parsed["n_unmapped"] = mapped_results.apply(lambda x: x[1])
    parsed["correct"] = parsed.apply(
        lambda r: r["label"] == ("Bad" if r["true_label"] == 1 else "Good"), axis=1
    )

    reference_conditions = [
        (model_name, method)
        for model_name in ["xgboost", "random_forest", "logistic_regression"]
        for method in ["shap_top", "lime_top"]
    ]

    precision_cols = []
    for model_name, method in reference_conditions:
        short_method = "shap" if method == "shap_top" else "lime"
        col_name = f"precision_{short_method}_{model_name}"
        precision_cols.append(col_name)

        def compute_row(row, model_name=model_name, method=method):
            case_lookup = ref_data.get(method, {}).get(model_name, {})
            case_id = str(row["case_id"])
            if case_id not in case_lookup:
                return None
            return compute_grounding_precision(row["mapped_features"], case_lookup[case_id])

        parsed[col_name] = parsed.apply(compute_row, axis=1)

    print("=== Grounding precision: Qwen3 key_factors vs reference explainers ===")
    print("(fraction of Qwen3's cited factors that also appear in the reference top-5)\n")
    summary = {}
    for col in precision_cols:
        valid = parsed[col].dropna()
        mean_val = valid.mean() if len(valid) else float("nan")
        summary[col] = float(mean_val) if len(valid) else None
        print(f"{col:35s} mean={mean_val:.1%}  (n={len(valid)})" if len(valid) else f"{col:35s} no data")

    print("\n--- By correctness ---")
    correctness_breakdown = parsed.groupby("correct")[precision_cols].mean()
    print(correctness_breakdown)

    total_unmapped = int(parsed["n_unmapped"].sum())
    total_factors = int(parsed["key_factors"].apply(lambda x: len(x) if x else 0).sum())
    unmapped_rate = total_unmapped / max(total_factors, 1)
    print(f"\nUnmapped factors: {total_unmapped}/{total_factors} ({unmapped_rate:.1%})")
    if unmapped_rate > 0.1:
        print(
            "WARNING: over 10% of cited factors didn't map to a known feature -- "
            "worth spot-checking a few to see if the model used phrasing the "
            "mapping doesn't recognize, which would understate true overlap "
            "rather than reflect genuinely ungrounded citations."
        )

    if args.show_examples > 0:
        scored = parsed.dropna(subset=precision_cols, how="all").copy()
        if len(scored):
            scored["mean_precision"] = scored[precision_cols].mean(axis=1, skipna=True)
            lowest = scored.sort_values("mean_precision").head(args.show_examples)
            print(f"\n--- {len(lowest)} lowest-grounding-precision example(s) ---")
            for _, row in lowest.iterrows():
                print(f"\n[{row['case_id']} seed={row['seed']}] mean precision={row['mean_precision']:.1%}, correct={row['correct']}")
                print(f"Cited (mapped to features): {row['mapped_features']}")
                print(f"Raw key_factors: {row['key_factors']}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "summary": summary,
            "correctness_breakdown": correctness_breakdown.to_dict(),
            "unmapped_rate": unmapped_rate,
        }, f, indent=2, default=str)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
