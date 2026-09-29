"""
mcnemar_tests.py

Pairwise McNemar's tests comparing all five systems (stock, structured,
LogReg, RF, XGBoost) on the same eval subset.

Usage:
    python mcnemar_tests.py --stock-results results/stock_results.jsonl --structured-results results/structured_results_revalidated.jsonl --baseline-predictions results/baseline_per_case_predictions.csv --metadata-path data/processed/metadata.json --output-path results/mcnemar_results.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def majority_vote_per_case(df: pd.DataFrame) -> pd.Series:
    parsed = df[df["parse_success"]]
    return parsed.groupby("case_id")["label"].agg(lambda x: x.value_counts().idxmax())


def run_mcnemar(correct_a: pd.Series, correct_b: pd.Series) -> dict:
    n01 = int(((~correct_a) & correct_b).sum())   # A wrong, B right
    n10 = int((correct_a & (~correct_b)).sum())   # A right, B wrong
    n00 = int(((~correct_a) & (~correct_b)).sum())
    n11 = int((correct_a & correct_b).sum())
    table = [[n11, n10], [n01, n00]]
    n_discordant = n01 + n10
    result = mcnemar(table, exact=(n_discordant < 25), correction=True)
    return {
        "n01_A_wrong_B_right": n01,
        "n10_A_right_B_wrong": n10,
        "n_discordant": n_discordant,
        "test_type": "exact_binomial" if n_discordant < 25 else "chi2_continuity_corrected",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-results", type=Path, required=True)
    parser.add_argument("--structured-results", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    with open(args.metadata_path) as f:
        metadata = json.load(f)

    stock_df = pd.read_json(args.stock_results, lines=True)
    structured_df = pd.read_json(args.structured_results, lines=True)
    baseline_df = pd.read_csv(args.baseline_predictions).set_index("case_id")

    stock_pred = majority_vote_per_case(stock_df)
    structured_pred = majority_vote_per_case(structured_df)
    true_label = stock_df.groupby("case_id")["true_label"].first().map({1: "Bad", 0: "Good"})

    label_map = {1: "Bad", 0: "Good"}
    preds = pd.DataFrame({
        "true": true_label,
        "stock": stock_pred,
        "structured": structured_pred,
        "logreg": baseline_df["logistic_regression_pred"].map(label_map),
        "rf": baseline_df["random_forest_pred"].map(label_map),
        "xgboost": baseline_df["xgboost_pred"].map(label_map),
    })
    n_before = len(preds)
    preds = preds.dropna()
    if len(preds) < n_before:
        print(f"WARNING: {n_before - len(preds)} case(s) missing a prediction from "
              f"at least one system, dropped from this analysis.")
    print(f"Comparing on {len(preds)} cases with predictions from all 5 systems.\n")

    systems = ["stock", "structured", "logreg", "rf", "xgboost"]
    correctness = pd.DataFrame({s: preds[s] == preds["true"] for s in systems})

    print("--- Accuracy per system on this common set ---")
    print(correctness.mean().sort_values(ascending=False).apply(lambda x: f"{x:.1%}"))

    primary_pairs = [
        ("structured", "stock"), ("structured", "logreg"),
        ("structured", "rf"), ("structured", "xgboost"),
    ]
    secondary_pairs = [("stock", "logreg"), ("stock", "rf"), ("stock", "xgboost")]

    print("\n=== PRIMARY comparisons (Holm-Bonferroni corrected across these 4) ===")
    print("Directly tests the proposal's hypothesis: structured outperforms stock and each classical baseline.\n")
    primary_results = []
    for a, b in primary_pairs:
        res = run_mcnemar(correctness[a], correctness[b])
        res["comparison"] = f"{a} vs {b}"
        primary_results.append(res)

    pvals = [r["p_value"] for r in primary_results]
    reject, corrected_pvals, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for r, corr_p, rej in zip(primary_results, corrected_pvals, reject):
        r["p_value_holm_corrected"] = float(corr_p)
        r["significant_at_0.05_corrected"] = bool(rej)
        print(f"{r['comparison']:20s} discordant={r['n_discordant']:3d} ({r['test_type']})  "
              f"raw p={r['p_value']:.4f}  holm p={corr_p:.4f}  significant={rej}")

    print("\n=== SECONDARY comparisons (exploratory -- NOT correction-adjusted) ===")
    secondary_results = []
    for a, b in secondary_pairs:
        res = run_mcnemar(correctness[a], correctness[b])
        res["comparison"] = f"{a} vs {b}"
        secondary_results.append(res)
        print(f"{res['comparison']:20s} discordant={res['n_discordant']:3d} ({res['test_type']})  raw p={res['p_value']:.4f}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "n_cases": len(preds),
            "accuracy_per_system": correctness.mean().to_dict(),
            "primary_comparisons": primary_results,
            "secondary_comparisons": secondary_results,
        }, f, indent=2)
    print(f"\nSaved full results to {args.output_path}")


if __name__ == "__main__":
    main()