"""
baselines.py

Trains and tunes the three classical ML baselines (Logistic Regression,
Random Forest, XGBoost) on the output of data_pipeline.py.

Design decisions (document these in your Method chapter, in your own words):

1. Preprocessing differs by model, not globally:
   - Logistic Regression: median imputation + StandardScaler (coefficient-
     based, scale-sensitive).
   - Random Forest: median imputation, no scaling (tree splits are
     scale-invariant; sklearn's RandomForestClassifier cannot accept NaN).
   - XGBoost: no imputation, no scaling. XGBoost natively learns a default
     split direction for missing values, so imputing would throw away
     information (a NaN here always means "sentinel code was present",
     since data_pipeline.py already replaced sentinels with NaN).
   Note that in every case, the sentinel-code indicator columns from
   data_pipeline.py (e.g. ExternalRiskEstimate_missing_no_bureau_record)
   are retained as ordinary binary features, so which sentinel applied is
   never lost even when the base value gets imputed.

2. No class-balancing correction is applied (no class_weight, no SMOTE).
   The processed dataset is close to balanced (52/48 in this project, per
   data_pipeline.py's target distribution output) -- this is a genuinely
   near-balanced dataset (the public HELOC release is itself a deliberate
   under-sample toward balance from a much larger population), not a
   severe-imbalance case, so no correction is applied. F1/ROC-AUC/PR-AUC
   are still reported rather than accuracy alone, per the proposal.

3. Hyperparameter tuning uses GridSearchCV with 5-fold stratified CV on
   the TRAIN split only. The val split is not used for model selection --
   it's scored afterward purely as an independent diagnostic (large
   train-vs-val gaps would flag overfitting). The test split is scored
   once, at the end, and is never touched during tuning.

Usage:
    python baselines.py --processed-dir data/processed --output-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 5],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1],
    },
}


def load_processed(processed_dir: Path):
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    return train, val, test, metadata


def build_pipeline(name: str, seed: int) -> Pipeline:
    if name == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=seed
            )),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(random_state=seed)),
        ])
    if name == "xgboost":
        return Pipeline([
            ("imputer", "passthrough"),
            ("classifier", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                missing=np.nan,
                random_state=seed,
            )),
        ])
    raise ValueError(f"Unknown model name: {name}")


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    train, val, test, metadata = load_processed(args.processed_dir)
    target_col = metadata["target_col"]
    feature_cols = metadata["feature_cols"] + metadata["indicator_cols"]

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Record training-time package versions. Pickled sklearn/xgboost
    # estimators are not guaranteed compatible across versions -- internal
    # attributes can be renamed between releases (this bit a real run of
    # this project: SimpleImputer._fill_dtype was renamed between sklearn
    # 1.6.1 and 1.9.0, producing a cryptic AttributeError deep inside
    # sklearn's own code with no indication of the real cause). Recording
    # versions here lets downstream scripts check and warn clearly instead.
    import sklearn
    import xgboost
    with open(models_dir / "_training_versions.json", "w") as f:
        json.dump({"sklearn": sklearn.__version__, "xgboost": xgboost.__version__}, f, indent=2)

    all_results = {}
    for name, grid in PARAM_GRIDS.items():
        print(f"\n=== Tuning {name} ===")
        pipeline = build_pipeline(name, args.seed)
        search = GridSearchCV(
            pipeline, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)

        print(f"Best params: {search.best_params_}")
        print(f"Best CV F1 (train folds): {search.best_score_:.4f}")

        best_model = search.best_estimator_
        val_metrics = evaluate(best_model, X_val, y_val)
        test_metrics = evaluate(best_model, X_test, y_test)

        print(f"Val F1: {val_metrics['f1']:.4f} | Test F1: {test_metrics['f1']:.4f}")
        if abs(val_metrics["f1"] - search.best_score_) > 0.05:
            print(
                f"NOTE: val F1 differs from train-CV F1 by "
                f">{abs(val_metrics['f1'] - search.best_score_):.3f} -- "
                f"worth checking for overfitting before treating this as final."
            )

        joblib.dump(best_model, models_dir / f"{name}.joblib")
        all_results[name] = {
            "best_params": search.best_params_,
            "cv_f1_train": search.best_score_,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    with open(args.output_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n=== Summary (test set) ===")
    summary = pd.DataFrame({
        name: res["test_metrics"] for name, res in all_results.items()
    }).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    print(summary.round(4))
    summary.to_csv(args.output_dir / "baseline_summary_test.csv")

    print(f"\nSaved models to {models_dir}, results to {args.output_dir}")


if __name__ == "__main__":
    main()
