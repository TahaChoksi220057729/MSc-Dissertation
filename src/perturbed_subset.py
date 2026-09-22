"""
perturbed_subset.py

Generates a numerically-perturbed variant of a sample of the LLM
evaluation subset, for testing whether small, meaningless changes to
continuous features flip the model's prediction

Usage:
    python perturbed_subset.py --subset-path data/processed/llm_eval_subset.csv --output-path data/processed/llm_eval_subset_perturbed_numeric.csv --n-cases 20 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# feature_name -> max perturbation magnitude (applied as +/- uniform(0, magnitude))
PERTURBABLE_FEATURES = {
    "ExternalRiskEstimate": 3,
    "PercentTradesNeverDelq": 2,
    "NetFractionRevolvingBurden": 3,
    "NetFractionInstallBurden": 3,
    "PercentInstallTrades": 2,
    "PercentTradesWBalance": 2,
}


def perturb_row(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    row = row.copy()
    for feature, magnitude in PERTURBABLE_FEATURES.items():
        if feature not in row.index or pd.isna(row[feature]):
            continue
        delta = rng.uniform(-magnitude, magnitude)
        new_value = row[feature] + delta
        if "Percent" in feature or "Fraction" in feature:
            new_value = max(0.0, min(100.0, new_value))
        elif feature == "ExternalRiskEstimate":
            new_value = max(0.0, new_value)
        row[feature] = round(new_value, 1)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=20,
                         help="number of cases to perturb (a sample, not the full subset -- see design decision 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subset = pd.read_csv(args.subset_path)
    missing = [f for f in PERTURBABLE_FEATURES if f not in subset.columns]
    if missing:
        raise SystemExit(f"Expected perturbable features not found in subset: {missing}")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_cases, len(subset))
    sample = subset.sample(n=n, random_state=args.seed).copy()
    perturbed = sample.apply(lambda row: perturb_row(row, rng), axis=1)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed.to_csv(args.output_path, index=False)

    print(f"Perturbed {len(perturbed)} of {len(subset)} cases -> {args.output_path}")
    print(f"Perturbed features (max magnitude): {PERTURBABLE_FEATURES}")
    print(
        "\nSame case_id values as the original subset -- this is what lets you "
        "match perturbed predictions back to the ORIGINAL ones for comparison "
        "later. Do not re-run select_stratified_subset.py to regenerate "
        "case_ids for this file; use it as-is."
    )
    print(
        f"\nThis needs {len(perturbed)} x n_repeats NEW generations per "
        f"pipeline tested. Consider n_repeats=1-2 (not 5) when running this "
        f"through run_llm_pipeline.py -- you're testing sensitivity to INPUT "
        f"change here, not decoding randomness, which the original 5-repeat "
        f"runs already measured."
    )


if __name__ == "__main__":
    main()