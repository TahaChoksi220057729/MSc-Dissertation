"""
compute_explanations.py

Computes SHAP and LIME explanations for all three classical baselines
(LogReg, RF, XGBoost) on the LLM evaluation subset, for later comparison
against the Qwen3 pipelines' stated key_factors/explanations.


Usage:
    python compute_explanations.py --processed-dir data/processed --models-dir results/models --output-dir results
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

TOP_K = 5


def shap_xgboost(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(X)
    return pd.DataFrame(explanation.values, columns=X.columns, index=X.index)


def shap_random_forest(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    X_imp = imputer.transform(X)
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(X_imp)
    # 3D output: (n_samples, n_features, n_classes) -- take class 1 ("Bad")
    values = explanation.values[:, :, 1] if explanation.values.ndim == 3 else explanation.values
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def shap_logistic_regression(pipeline, X_train: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    X_train_t = scaler.transform(imputer.transform(X_train))
    X_t = scaler.transform(imputer.transform(X))
    background = shap.sample(X_train_t, 100) if len(X_train_t) > 100 else X_train_t
    explainer = shap.LinearExplainer(classifier, background)
    explanation = explainer(X_t)
    return pd.DataFrame(explanation.values, columns=X.columns, index=X.index)


def lime_explanations(pipeline, X_train: pd.DataFrame, X: pd.DataFrame, feature_cols: list[str]):
    """Returns (weights_df, r2_scores) -- weights_df indexed like X, columns=feature_cols."""
    def predict_fn(arr):
        return pipeline.predict_proba(pd.DataFrame(arr, columns=feature_cols))

    explainer = LimeTabularExplainer(
        training_data=X_train.fillna(0).values,
        feature_names=feature_cols,
        class_names=["Good", "Bad"],
        mode="classification",
        discretize_continuous=False,
    )

    weights = pd.DataFrame(index=X.index, columns=feature_cols, dtype=float)
    r2_scores = pd.Series(index=X.index, dtype=float)
    for idx, row in X.iterrows():
        exp = explainer.explain_instance(
            row.fillna(0).values, predict_fn, num_features=len(feature_cols), labels=[1]
        )
        for feat_idx, weight in exp.as_map()[1]:
            weights.loc[idx, feature_cols[feat_idx]] = weight
        r2_scores.loc[idx] = exp.score
    return weights, r2_scores


def top_k_features(row: pd.Series, k: int = TOP_K) -> list[str]:
    return row.abs().sort_values(ascending=False).head(k).index.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"] + metadata["indicator_cols"]

    train = pd.read_csv(args.processed_dir / "train.csv")
    subset = pd.read_csv(args.processed_dir / "llm_eval_subset.csv")
    X_train = train[feature_cols]
    X = subset[feature_cols]
    X.index = subset["case_id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shap_top_by_model = {}

    for name, shap_fn in [
        ("xgboost", lambda: shap_xgboost(joblib.load(args.models_dir / "xgboost.joblib"), X)),
        ("random_forest", lambda: shap_random_forest(joblib.load(args.models_dir / "random_forest.joblib"), X)),
        ("logistic_regression", lambda: shap_logistic_regression(
            joblib.load(args.models_dir / "logistic_regression.joblib"), X_train, X)),
    ]:
        print(f"Computing SHAP for {name}...")
        shap_values = shap_fn()
        shap_values.to_csv(args.output_dir / f"shap_values_{name}.csv")
        shap_top_by_model[name] = shap_values.apply(top_k_features, axis=1)
        print(f"  Saved shap_values_{name}.csv")

    print("\nComputing LIME (all 3 models)...")
    lime_top_by_model = {}
    lime_r2_by_model = {}
    for name in ["xgboost", "random_forest", "logistic_regression"]:
        pipeline = joblib.load(args.models_dir / f"{name}.joblib")
        weights, r2 = lime_explanations(pipeline, X_train, X, feature_cols)
        weights.to_csv(args.output_dir / f"lime_weights_{name}.csv")
        r2.to_csv(args.output_dir / f"lime_r2_{name}.csv")
        lime_top_by_model[name] = weights.apply(top_k_features, axis=1)
        lime_r2_by_model[name] = r2
        print(f"  {name}: mean local R^2 = {r2.mean():.3f} (min={r2.min():.3f})")

    print(f"\n--- SHAP vs LIME top-{TOP_K} feature overlap (per model, BEFORE any Qwen3 comparison) ---")
    overlap_summary = {}
    for name in ["xgboost", "random_forest", "logistic_regression"]:
        overlaps = [
            len(set(shap_top_by_model[name][cid]) & set(lime_top_by_model[name][cid])) / TOP_K
            for cid in X.index
        ]
        overlap_summary[name] = float(np.mean(overlaps))
        print(f"{name:22s} mean top-{TOP_K} overlap = {np.mean(overlaps):.1%}")

    print(
        "\nLow overlap here doesn't mean either method is 'wrong' -- SHAP and LIME "
        "use genuinely different attribution logic (see design decision 5 above). "
        "It DOES mean: if Qwen3's stated factors disagree with SHAP, check whether "
        "they align with LIME instead before concluding Qwen3 is ungrounded -- "
        "disagreement between two reference methods is itself informative context."
    )

    with open(args.output_dir / "shap_lime_top_features.json", "w") as f:
        json.dump({
            "shap_top": {m: shap_top_by_model[m].to_dict() for m in shap_top_by_model},
            "lime_top": {m: lime_top_by_model[m].to_dict() for m in lime_top_by_model},
            "shap_lime_overlap_by_model": overlap_summary,
        }, f, indent=2)

    print(f"\nSaved all outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
