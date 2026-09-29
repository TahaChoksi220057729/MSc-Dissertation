"""
compute_calibration.py

Computes Brier score, Expected Calibration Error (ECE), and calibration
curve data for all five systems on the LLM evaluation subset

Usage:
    python compute_calibration.py --stock-results results/stock_results.jsonl --structured-results results/structured_results_revalidated.jsonl --baseline-predictions results/baseline_per_case_predictions.csv --output-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def aggregate_prob_bad_per_case(df: pd.DataFrame) -> pd.Series:
    parsed = df[df["parse_success"]].copy()
    parsed["prob_bad_run"] = parsed.apply(
        lambda r: r["confidence"] / 100 if r["label"] == "Bad" else 1 - r["confidence"] / 100,
        axis=1,
    )
    return parsed.groupby("case_id")["prob_bad_run"].mean()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_prob_true = y_true[mask].mean()
        bin_prob_pred = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_prob_true - bin_prob_pred)
    return ece


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-results", type=Path, required=True)
    parser.add_argument("--structured-results", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    stock_df = pd.read_json(args.stock_results, lines=True)
    structured_df = pd.read_json(args.structured_results, lines=True)
    baseline_df = pd.read_csv(args.baseline_predictions).set_index("case_id")

    stock_prob = aggregate_prob_bad_per_case(stock_df)
    structured_prob = aggregate_prob_bad_per_case(structured_df)
    true_label = stock_df.groupby("case_id")["true_label"].first()  # 0/1, 1=Bad

    systems = {
        "stock": stock_prob,
        "structured": structured_prob,
        "logreg": baseline_df["logistic_regression_proba"],
        "rf": baseline_df["random_forest_proba"],
        "xgboost": baseline_df["xgboost_proba"],
    }

    results = {}
    calibration_curves = {}
    print(f"{'system':14s} {'n':>4s}  {'Brier':>7s}  {'ECE':>7s}")
    for name, prob_series in systems.items():
        common = true_label.index.intersection(prob_series.index)
        y_true = true_label.loc[common].values.astype(float)
        y_prob = prob_series.loc[common].values.astype(float)

        brier = brier_score_loss(y_true, y_prob)
        ece = compute_ece(y_true, y_prob, args.n_bins)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=args.n_bins, strategy="uniform")

        results[name] = {"n_cases": len(common), "brier_score": float(brier), "ece": float(ece)}
        calibration_curves[name] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
        print(f"{name:14s} {len(common):4d}  {brier:7.4f}  {ece:7.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "calibration_results.json", "w") as f:
        json.dump({"metrics": results, "calibration_curves": calibration_curves}, f, indent=2)

    print(f"\nSaved to {args.output_dir / 'calibration_results.json'}")
    print(
        "\nLower Brier score and lower ECE both indicate better calibration. "
        "A system that's accurate but overconfident (like stock's false "
        "negatives at 90%+ confidence, found much earlier in this project) "
        "will show a WORSE Brier/ECE than its raw accuracy alone would "
        "suggest -- that's exactly what these metrics are designed to catch."
    )


if __name__ == "__main__":
    main()
