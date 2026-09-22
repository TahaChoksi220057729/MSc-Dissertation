"""
revalidate_results.py

Re-runs parse_structured_response against ALREADY-GENERATED raw_response
text, without any new GPU generation or Colab session. Use this whenever
structured_pipeline.py's validation logic changes

Usage:
    python revalidate_results.py --results-path results/structured_results.jsonl --subset-path data/processed/llm_eval_subset.csv --metadata-path data/processed/metadata.json --output-path results/structured_results_revalidated.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from structured_pipeline import parse_structured_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_json(args.results_path, lines=True)
    subset = pd.read_csv(args.subset_path).set_index("case_id")
    with open(args.metadata_path) as f:
        metadata = json.load(f)

    old_parse_success = int(df["parse_success"].sum())

    records = []
    missing_cases = set()
    flipped_to_pass = 0
    flipped_to_fail = 0
    for _, rec in df.iterrows():
        case_id = rec["case_id"]
        if case_id not in subset.index:
            missing_cases.add(case_id)
            continue
        row = subset.loc[case_id]
        parsed = parse_structured_response(rec["raw_response"], row=row, metadata=metadata)

        if not rec["parse_success"] and parsed["parse_success"]:
            flipped_to_pass += 1
        elif rec["parse_success"] and not parsed["parse_success"]:
            flipped_to_fail += 1

        new_rec = {
            "case_id": case_id,
            "pipeline": rec.get("pipeline", "structured"),
            "seed": rec["seed"],
            "true_label": rec["true_label"],
            "raw_response": rec["raw_response"],
            "elapsed_seconds": rec.get("elapsed_seconds"),
            **parsed,
        }
        records.append(new_rec)

    if missing_cases:
        print(f"WARNING: {len(missing_cases)} case_id(s) in results not found in "
              f"the eval subset -- skipped: {list(missing_cases)[:5]}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")

    new_parse_success = sum(r["parse_success"] for r in records)
    print(f"Re-validated {len(records)} records.")
    print(f"Newly passing (were failing, now correct): {flipped_to_pass}")
    print(f"Newly failing (were passing, now correctly caught): {flipped_to_fail}")
    print(
        f"parse_success: {old_parse_success}/{len(df)} ({old_parse_success/len(df):.1%}) -> "
        f"{new_parse_success}/{len(records)} ({new_parse_success/len(records):.1%})"
    )
    print(f"\nWrote {args.output_path}")
    print(
        "Run analyze_results.py on this new file for corrected diagnostics. "
        "The original file is untouched -- keep both if you want to document "
        "the before/after in your Method or AI-use chapter."
    )


if __name__ == "__main__":
    main()