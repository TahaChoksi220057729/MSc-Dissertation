"""
select_stratified_subset.py

Selects the stratified LLM-evaluation subset, used identically by both the
stock and structured pipelines (same cases for both -- this is what makes
the pipeline-vs-pipeline comparison fair; if they saw different cases, any
difference in results could just be case difficulty, not pipeline design).
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def select_subset(test: pd.DataFrame, target_col: str, proba, n_per_class: int, seed: int) -> pd.DataFrame:
    test = test.copy()
    test["_proba"] = proba
    test["_borderline_score"] = (test["_proba"] - 0.5).abs()

    half = n_per_class // 2
    selected_parts = []
    for cls in sorted(test[target_col].unique()):
        cls_rows = test[test[target_col] == cls]
        clear = cls_rows.sort_values("_borderline_score", ascending=False).head(half)
        borderline = cls_rows.sort_values("_borderline_score", ascending=True).head(n_per_class - half)
        selected_parts.append(clear)
        selected_parts.append(borderline)

    subset = pd.concat(selected_parts).drop_duplicates()
    subset = subset.reset_index().rename(columns={"index": "original_test_index"})
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    subset.insert(0, "case_id", [f"case_{i:04d}" for i in range(len(subset))])
    return subset.drop(columns=["_proba", "_borderline_score"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                         help="e.g. results/models/xgboost.joblib -- used only "
                              "to identify clear vs borderline cases, not as "
                              "part of the LLM evaluation itself")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    test = pd.read_csv(args.processed_dir / "test.csv")

    model = joblib.load(args.model_path)
    feature_cols = metadata["feature_cols"] + metadata["indicator_cols"]
    proba = model.predict_proba(test[feature_cols])[:, 1]

    subset = select_subset(test, metadata["target_col"], proba, args.n_per_class, args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output_path, index=False)

    print(f"Selected {len(subset)} cases -> {args.output_path}")
    print(f"Class balance:\n{subset[metadata['target_col']].value_counts()}")

if __name__ == "__main__":
    main()
