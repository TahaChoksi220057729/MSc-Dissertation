"""
run_llm_pipeline.py

"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from applicant_profile import build_profile_text


def get_pipeline_functions(name: str, few_shot_path: Path = None):
    if name == "stock":
        from stock_pipeline import build_stock_prompt, parse_stock_response
        return build_stock_prompt, parse_stock_response
    if name == "structured":
        from src.structured_pipeline import build_structured_prompt, parse_structured_response
        if few_shot_path is None:
            raise ValueError("structured pipeline requires --few-shot-path")

        def build_prompt_fn(profile_text):
            return build_structured_prompt(profile_text, few_shot_path)

        return build_prompt_fn, parse_structured_response
    raise ValueError(f"Unknown pipeline: {name}")


def load_completed(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed.add((rec["case_id"], rec["seed"]))
    return completed


def run_pipeline(
    subset: pd.DataFrame,
    metadata: dict,
    pipeline_name: str,
    build_prompt_fn,
    parse_fn,
    generate_fn,
    output_path: Path,
    n_repeats: int,
    max_new_tokens: int,
    progress_every: int = 10,
) -> int:
    """
    generate_fn(user_prompt: str, seed: int, system_prompt: str,
    max_new_tokens: int) -> str. Returns the number of generations run
    (excludes ones skipped because already completed).
    """
    completed = load_completed(output_path)
    todo = [
        (idx, row, seed)
        for idx, row in subset.iterrows()
        for seed in range(n_repeats)
        if (row["case_id"], seed) not in completed
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_run = 0
    with open(output_path, "a") as out_f:
        for i, (idx, row, seed) in enumerate(todo):
            profile_text = build_profile_text(row, metadata)
            system_prompt, user_prompt = build_prompt_fn(profile_text)

            start = time.time()
            raw_response = generate_fn(user_prompt, seed, system_prompt, max_new_tokens)
            elapsed = time.time() - start

            parsed = parse_fn(raw_response, row, metadata)

            record = {
                "case_id": row["case_id"],
                "pipeline": pipeline_name,
                "seed": seed,
                "true_label": row[metadata["target_col"]],
                "raw_response": raw_response,
                "elapsed_seconds": elapsed,
                **parsed,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_run += 1

            if n_run % progress_every == 0:
                print(
                    f"[{n_run}/{len(todo)}] case={row['case_id']} seed={seed} "
                    f"parse_success={parsed['parse_success']} elapsed={elapsed:.1f}s"
                )

    return n_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["stock", "structured"], required=True)
    parser.add_argument("--subset-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--few-shot-path", type=Path, default=None,
                         help="required for --pipeline structured")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    build_prompt_fn, parse_fn = get_pipeline_functions(args.pipeline, args.few_shot_path)

    subset = pd.read_csv(args.subset_path)
    with open(args.metadata_path) as f:
        metadata = json.load(f)

    print("Loading model...")
    from qwen3_inference import load_model_and_tokenizer, generate
    model, tokenizer = load_model_and_tokenizer(quantize_4bit=True)
    print("Model loaded.")

    def generate_fn(user_prompt, seed, system_prompt, max_new_tokens):
        return generate(
            model, tokenizer, user_prompt, seed=seed,
            system_prompt=system_prompt, max_new_tokens=max_new_tokens,
        )

    total_needed = len(subset) * args.n_repeats
    already_done = len(load_completed(args.output_path))
    print(f"{already_done}/{total_needed} (case_id, seed) pairs already completed.")

    n_run = run_pipeline(
        subset, metadata, args.pipeline, build_prompt_fn, parse_fn, generate_fn,
        args.output_path, args.n_repeats, args.max_new_tokens,
    )
    print(f"Done. Ran {n_run} new generations. Results in {args.output_path}")


if __name__ == "__main__":
    main()
