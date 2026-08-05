"""
data_pipeline.py

Loads, inspects, cleans, and stratified-splits the FICO HELOC dataset.

Usage:
    python data_pipeline.py --raw-path data/raw/heloc_dataset.csv \ --output-dir data/processed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "RiskPerformance"
POSITIVE_LABEL = "Bad"  # Bad (high default risk) encoded as 1, the event of interest

# FICO HELOC sentinel codes and their documented meaning.
SPECIAL_CODES = {
    -9: "no_bureau_record",
    -8: "no_usable_trades_or_inquiries",
    -7: "condition_not_met",
}

ID_LIKE_PATTERNS = ("unnamed", "index", "_id", "^id$")

def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"Raw data file not found at {path}.\n"
            f"Download heloc_dataset.csv via the FICO Explainable ML Challenge "
            f"community page (see your proposal's FICO 2018 reference) and place "
            f"it at this path before running this script."
        )
    df = pd.read_csv(path)
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c for c in df.columns
        if any(pat in c.lower() for pat in ("unnamed", "index"))
        or c.lower() in ("id",)
    ]
    if drop_cols:
        print(f"Dropping identifier-like columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def inspect(df: pd.DataFrame, feature_cols: list[str]) -> None:
    print("\n--- Dataset inspection ---")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")

    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))

    print("\nDtypes:")
    print(df[feature_cols].dtypes.value_counts())

    print("\nSpecial-code counts per feature (top 10 by count):")
    counts = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            for code in SPECIAL_CODES:
                n = int((df[col] == code).sum())
                if n:
                    counts[f"{col} == {code} ({SPECIAL_CODES[code]})"] = n
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")
    print("--- end inspection ---\n")


def drop_fully_unusable_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # A row where every feature is <= -9 carries no usable information.
    mask = (df[feature_cols] > -9).any(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"Dropping {n_dropped} fully-unusable rows (all features <= -9).")
    return df.loc[mask].reset_index(drop=True)


def encode_special_values(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    indicator_cols = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        for code, name in SPECIAL_CODES.items():
            ind_col = f"{col}_missing_{name}"
            flag = (df[col] == code).astype(int)
            if flag.sum() > 0:
                df[ind_col] = flag
                indicator_cols.append(ind_col)
        df.loc[df[col].isin(SPECIAL_CODES.keys()), col] = np.nan
    return df, indicator_cols


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        unique_vals = set(df[TARGET_COL].dropna().unique())
        if not unique_vals <= {0, 1}:
            sys.exit(f"Target column is numeric but not binary 0/1: {unique_vals}")
        print(f"Target column '{TARGET_COL}' is already binary-encoded. "
              f"Polarity will be checked against ExternalRiskEstimate below "
              f"-- read that output before trusting the labels.")
    else:
        unexpected = set(df[TARGET_COL].unique()) - {"Good", "Bad"}
        if unexpected:
            sys.exit(f"Unexpected values in {TARGET_COL}: {unexpected}")
        df[TARGET_COL] = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)
    return df


def check_target_direction(df: pd.DataFrame) -> None:
    if "ExternalRiskEstimate" not in df.columns:
        print("Skipping target-direction check: ExternalRiskEstimate not found.")
        return
    valid = df[df["ExternalRiskEstimate"] >= 0]  # exclude sentinel codes
    means = valid.groupby(TARGET_COL)["ExternalRiskEstimate"].mean()
    print("\n--- Target polarity check ---")
    print(f"Mean ExternalRiskEstimate by target value:\n{means}")
    if means.get(1, np.nan) > means.get(0, np.nan):
        print(
            "WARNING: target==1 has a HIGHER mean ExternalRiskEstimate than "
            "target==0. If 1 is meant to represent Bad/high-risk, this is "
            "backwards"
        )
    else:
        print("--- end target polarity check ---\n")


def stratified_split(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=seed
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[TARGET_COL],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw(args.raw_path)
    df = drop_identifier_columns(df)

    if TARGET_COL not in df.columns:
        sys.exit(f"Expected target column '{TARGET_COL}' not found in {args.raw_path}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    inspect(df, feature_cols)

    df = drop_fully_unusable_rows(df, feature_cols)
    df = encode_target(df)
    check_target_direction(df)
    df, indicator_cols = encode_special_values(df, feature_cols)

    train, val, test = stratified_split(df, args.test_size, args.val_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    val.to_csv(args.output_dir / "val.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    metadata = {
        "target_col": TARGET_COL,
        "positive_label": POSITIVE_LABEL,
        "feature_cols": feature_cols,
        "indicator_cols": indicator_cols,
        "special_codes": SPECIAL_CODES,
        "seed": args.seed,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "class_balance": {
            "train": train[TARGET_COL].value_counts(normalize=True).to_dict(),
            "val": val[TARGET_COL].value_counts(normalize=True).to_dict(),
            "test": test[TARGET_COL].value_counts(normalize=True).to_dict(),
        },
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved train/val/test + metadata.json to {args.output_dir}")
    print(f"Split sizes: {metadata['split_sizes']}")
    print(f"Class balance (train): {metadata['class_balance']['train']}")


if __name__ == "__main__":
    main()
