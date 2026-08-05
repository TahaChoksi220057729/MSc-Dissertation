"""
applicant_profile.py

Converts one row of processed HELOC data into a natural-language paragraph
describing the applicant, for use as LLM input. This module is SHARED
infrastructure -- both the stock and structured pipelines use the exact
same profile text for a given applicant, differing only in how the prompt
wraps around it.

   USAGE: python applicant_profile.py --check-coverage data/processed
"""

import sys
import warnings

import pandas as pd

# canonical_name -> (human-readable label, unit string or None)
FEATURE_DESCRIPTIONS = {
    "ExternalRiskEstimate": ("consolidated external risk estimate (higher values indicate lower credit risk)", None),
    "MSinceOldestTradeOpen": ("months since the oldest credit trade was opened", "months"),
    "MSinceMostRecentTradeOpen": ("months since the most recent credit trade was opened", "months"),
    "AverageMInFile": ("average number of months trade lines have been on file", "months"),
    "NumSatisfactoryTrades": ("number of satisfactory trades", None),
    "NumTrades60Ever2DerogPubRec": ("number of trades ever 60+ days delinquent or with a derogatory public record", None),
    "NumTrades90Ever2DerogPubRec": ("number of trades ever 90+ days delinquent or with a derogatory public record", None),
    "PercentTradesNeverDelq": ("percentage of trades never delinquent", "%"),
    "MSinceMostRecentDelq": ("months since the most recent delinquency", "months"),
    "MaxDelq2PublicRecLast12M": ("worst delinquency/public-record status in the last 12 months", None),
    "MaxDelqEver": ("worst delinquency/public-record status ever recorded", None),
    "NumTotalTrades": ("total number of trades", None),
    "NumTradesOpeninLast12M": ("number of trades opened in the last 12 months", None),
    "PercentInstallTrades": ("percentage of trades that are installment trades", "%"),
    "MSinceMostRecentInqexcl7days": ("months since the most recent credit inquiry (excluding the last 7 days)", "months"),
    "NumInqLast6M": ("number of credit inquiries in the last 6 months", None),
    "NumInqLast6Mexcl7days": ("number of credit inquiries in the last 6 months, excluding the last 7 days", None),
    "NetFractionRevolvingBurden": ("revolving balance as a percentage of the revolving credit limit", "%"),
    "NetFractionInstallBurden": ("installment balance as a percentage of the original installment loan amount", "%"),
    "NumRevolvingTradesWBalance": ("number of revolving trades carrying a balance", None),
    "NumInstallTradesWBalance": ("number of installment trades carrying a balance", None),
    "NumBank2NatlTradesWHighUtilization": ("number of bank/national trades with high utilisation", None),
    "PercentTradesWBalance": ("percentage of trades carrying a balance", "%"),
}


MAXDELQ_12M_LABELS = {
    0: "a derogatory public record",
    1: "120+ days delinquent",
    2: "90 days delinquent",
    3: "60 days delinquent",
    4: "30 days delinquent",
    5: "unknown delinquency status",
    6: "unknown delinquency status",
    7: "current and never delinquent",
    8: "an other/unclassified status",
    9: "an other/unclassified status",
}
MAXDELQ_EVER_LABELS = {
    2: "a derogatory public record",
    3: "120+ days delinquent",
    4: "90 days delinquent",
    5: "60 days delinquent",
    6: "30 days delinquent",
    7: "unknown delinquency status",
    8: "current and never delinquent",
    9: "an other/unclassified status",
}

PROFILE_SECTION_ORDER = [
    ["ExternalRiskEstimate"],
    ["MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen", "AverageMInFile",
     "NumTotalTrades", "NumTradesOpeninLast12M", "NumSatisfactoryTrades"],
    ["NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec",
     "PercentTradesNeverDelq", "MSinceMostRecentDelq",
     "MaxDelq2PublicRecLast12M", "MaxDelqEver"],
    ["MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days"],
    ["PercentInstallTrades", "NetFractionRevolvingBurden", "NetFractionInstallBurden",
     "NumRevolvingTradesWBalance", "NumInstallTradesWBalance",
     "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance"],
]

SENTINEL_PHRASING = {
    "no_bureau_record": "no data is on file for this applicant (no bureau record or no investigation)",
    "no_usable_trades_or_inquiries": "this measure is not available (no usable or valid trades or inquiries of this type)",
    "condition_not_met": "this measure is not applicable (the relevant condition was not met, e.g. no inquiries or no delinquencies on record)",
}


def _describe_maxdelq(feature_name: str, value: float) -> str:
    mapping = MAXDELQ_12M_LABELS if feature_name == "MaxDelq2PublicRecLast12M" else MAXDELQ_EVER_LABELS
    label, _ = FEATURE_DESCRIPTIONS[feature_name]
    code = int(value)
    if code in mapping:
        return f"The applicant's {label} is: {mapping[code]}."
    warnings.warn(
        f"{feature_name}: code {code} not in the confirmed mapping. This "
        f"mapping is now the complete official scheme from FICO's data "
        f"dictionary, so an unmapped code here indicates a genuine data "
        f"anomaly (e.g. MaxDelqEver==1, documented as 'No such value'), "
        f"not a documentation gap -- worth investigating this specific row."
    )
    return (
        f"The applicant's {label} is coded {code}, which does not match "
        f"any documented category for this field "
    )


def _describe_feature(feature_name: str, row: pd.Series, indicator_cols: list[str]) -> str:
    label, unit = FEATURE_DESCRIPTIONS[feature_name]

    fired_indicator = None
    for ind_col in indicator_cols:
        if ind_col.startswith(f"{feature_name}_missing_") and row.get(ind_col, 0) == 1:
            sentinel_key = ind_col[len(f"{feature_name}_missing_"):]
            fired_indicator = sentinel_key
            break

    if fired_indicator is not None:
        phrase = SENTINEL_PHRASING.get(fired_indicator, "this measure is not available")
        return f"For {label}, {phrase}."

    value = row[feature_name]
    if pd.isna(value):
        warnings.warn(
            f"{feature_name} is NaN but no sentinel indicator fired for this "
            f"row -- unexpected given data_pipeline.py's encoding. Falling "
            f"back to a generic missing-data phrase."
        )
        return f"For {label}, no data is available for this applicant."

    if feature_name in ("MaxDelq2PublicRecLast12M", "MaxDelqEver"):
        return _describe_maxdelq(feature_name, value)

    if unit == "%":
        return f"The applicant's {label} is {value:.1f}%."
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"The applicant's {label} is {value}" + (f" {unit}." if unit else ".")


def build_profile_text(row: pd.Series, metadata: dict) -> str:
    indicator_cols = metadata["indicator_cols"]
    sentences = []
    for section in PROFILE_SECTION_ORDER:
        for feature_name in section:
            if feature_name not in FEATURE_DESCRIPTIONS:
                continue
            sentences.append(_describe_feature(feature_name, row, indicator_cols))
    return " ".join(sentences)


def check_coverage(processed_dir):
    import json
    from pathlib import Path

    processed_dir = Path(processed_dir)
    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    train = pd.read_csv(processed_dir / "train.csv")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _, row in train.iterrows():
            build_profile_text(row, metadata)
        fallback_warnings = [w for w in caught if "not in the confirmed mapping" in str(w.message)]

    print(f"Checked {len(train)} rows.")
    print(f"MaxDelq fallback fired {len(fallback_warnings)} times.")
    if fallback_warnings:
        print("Unmapped codes encountered")
    else:
        print("No unmapped codes found")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-coverage":
        check_coverage(sys.argv[2] if len(sys.argv) > 2 else "data/processed")
    else:
        print("Usage: python applicant_profile.py --check-coverage [processed_dir]")
