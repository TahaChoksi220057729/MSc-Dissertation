"""
export_hf_heloc.py

Exports mstz/heloc to data/raw/heloc_dataset.csv 
"""

import sys

from huggingface_hub import login

login(token="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")

from datasets import load_dataset

RENAME_MAP = {
    "estimate_of_risk": "ExternalRiskEstimate",
    "months_since_first_trade": "MSinceOldestTradeOpen",
    "months_since_last_trade": "MSinceMostRecentTradeOpen",
    "average_duration_of_resolution": "AverageMInFile",
    "number_of_satisfactory_trades": "NumSatisfactoryTrades",
    "nr_trades_insolvent_for_over_60_days": "NumTrades60Ever2DerogPubRec",
    "nr_trades_insolvent_for_over_90_days": "NumTrades90Ever2DerogPubRec",
    "percentage_of_legal_trades": "PercentTradesNeverDelq",
    "months_since_last_illegal_trade": "MSinceMostRecentDelq",
    "maximum_illegal_trades_over_last_year": "MaxDelq2PublicRecLast12M",
    "maximum_illegal_trades": "MaxDelqEver",
    "nr_total_trades": "NumTotalTrades",
    "nr_trades_initiated_in_last_year": "NumTradesOpeninLast12M",
    "percentage_of_installment_trades": "PercentInstallTrades",
    "months_since_last_inquiry_not_recent": "MSinceMostRecentInqexcl7days",
    "nr_inquiries_in_last_6_months": "NumInqLast6M",
    "nr_inquiries_in_last_6_months_not_recent": "NumInqLast6Mexcl7days",
    "net_fraction_of_revolving_burden": "NetFractionRevolvingBurden",
    "net_fraction_of_installment_burden": "NetFractionInstallBurden",
    "nr_revolving_trades_with_balance": "NumRevolvingTradesWBalance",
    "nr_installment_trades_with_balance": "NumInstallTradesWBalance",
    "nr_banks_with_high_ratio": "NumBank2NatlTradesWHighUtilization",
    "percentage_trades_with_balance": "PercentTradesWBalance",
    "is_at_risk": "RiskPerformance",
}


def main():
    dataset = load_dataset("mstz/heloc")["train"]
    df = dataset.to_pandas()

    missing = set(RENAME_MAP) - set(df.columns)
    if missing:
        sys.exit(
            f"Expected columns not found in mstz/heloc: {missing}\n"
            f"The dataset schema may have changed since this script was "
            f"written (2026-08-03) -- check the current column list on "
            f"https://huggingface.co/datasets/mstz/heloc before proceeding, "
            f"and update RENAME_MAP to match."
        )

    df = df.rename(columns=RENAME_MAP)
    out_path = "data/raw/heloc_dataset.csv"
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
