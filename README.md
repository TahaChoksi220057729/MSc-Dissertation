# Structured LLM Pipelines for Credit-Risk Assessment

Code for the MSc Artificial Intelligence project *AI-Assisted Decision-Making for Credit-Risk Assessment* (City St George's, University of London).

The project compares five systems on the FICO HELOC credit-risk dataset. Three are traditional baselines: Logistic Regression, Random Forest and XGBoost. The other two are Qwen3-8B pipelines:

- **Stock pipeline:** a minimal prompt that asks for a label, a confidence and an explanation.
- **Structured pipeline:** adds a feature glossary, a decision rubric, three worked examples, a JSON output schema and five automated validation checks.

The systems are compared on accuracy, McNemar tests with Holm correction, calibration (Brier score and ECE), consistency across repeated runs, and robustness to rewording and small numeric changes. They are also compared on how far the LLM's stated factors agree with SHAP and LIME attributions.

---

## Repository structure

```
.
├── data/
│   ├── raw/heloc_dataset.csv          # HELOC data exported from Hugging Face
│   └── processed/                     # splits, metadata, evaluation subset, few-shot examples
├── notebooks/
│   ├── run_stock_pipeline.ipynb       # Colab: stock pipeline (100 cases x 5 seeds)
│   └── run_structured_pipeline.ipynb  # Colab: structured pipeline (100 cases x 5 seeds)
├── results/                           # models, metrics, raw LLM outputs, analysis outputs
├── src/                               # all Python scripts (listed below)
├── requirements.txt                   # local environment
```

### Scripts in `src/`, in the order they are run

| Script | What it does | Runs on |
|---|---|---|
| `export_hf_heloc.py` | Downloads `mstz/heloc`, renames columns to the FICO names, saves `data/raw/heloc_dataset.csv` | Local |
| `data_pipeline.py` | Removes all-sentinel rows, encodes the target, creates sentinel indicators, stratified 70/15/15 split (seed 42), writes `metadata.json` | Local |
| `baselines.py` | Grid search with 5-fold CV, trains the three baselines, scores the test set once, saves models | Local |
| `select_stratified_subset.py` | Selects the 100-case evaluation subset (50 Good / 50 Bad) from XGBoost probabilities | Local |
| `score_baselines_on_eval_subset.py` | Scores the baselines on the subset; saves per-case predictions | Local |
| `select_few_shot_examples.py` | Chooses the three worked examples from the training set | Local |
| `perturbed_subset.py` | Creates the numerically perturbed copy of 20 subset cases | Local |
| `applicant_profile.py` | Converts a data row into the text profile shared by both pipelines | Shared |
| `applicant_profile_reworded.py` | Reworded profiles for the robustness test | Colab |
| `qwen3_inference.py` | Loads Qwen3-8B (4-bit NF4) and generates a response for a given seed | Colab |
| `stock_pipeline.py` | Stock prompt and parser | Colab |
| `structured_pipeline.py` | Structured prompt and the five validation checks | Shared |
| `run_llm_pipeline.py` | Resumable run harness: one JSON line per (case, seed) | Colab |
| `revalidate_results.py` | Re-applies the validation checks to saved responses (no GPU needed) | Local |
| `analyze_results.py` | Majority-vote accuracy, consistency and parse rates | Local |
| `investigate_grounding_failures.py` | Diagnostic for grounding-check failures | Local |
| `mcnemar_tests.py` | Paired McNemar tests with Holm correction | Local |
| `compute_calibration.py` | Brier score, ECE and calibration curves | Local |
| `compute_explanations.py` | SHAP and LIME attributions for the baselines | Local |
| `compare_explanations_to_qwen3.py` | Compares stated factors with SHAP/LIME top-5 features | Local |
| `robustness_analysis.py` | Compares perturbed/reworded labels with the original labels | Local |

---

## Setup

### Local (data preparation, baselines and analysis)

Python 3.11 or later.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-learn` and `xgboost` are pinned because the saved `.joblib` models only load reliably under the versions they were trained with. If you use other versions, retrain the baselines with step 3 below.

### Google Colab (LLM inference)

A GPU runtime is needed (the project used an NVIDIA T4). The notebooks install `transformers`, `accelerate` and `bitsandbytes` in their first cell. Qwen3-8B (`Qwen/Qwen3-8B`) is downloaded from Hugging Face automatically on first use.

---

## Reproducing the results

All local commands are run from the repository root. The processed data, models and results are already included, so each stage can also be run on its own.

### 1. Download and prepare the data (local)

Log in to Hugging Face first (`huggingface-cli login`, or set the `HF_TOKEN` environment variable).

```bash
python src/export_hf_heloc.py
python src/data_pipeline.py --raw-path data/raw/heloc_dataset.csv --output-dir data/processed
```

### 2. Train the baselines (local)

```bash
python src/baselines.py --processed-dir data/processed --output-dir results
```

### 3. Build the LLM inputs (local)

```bash
python src/select_stratified_subset.py --processed-dir data/processed --model-path results/models/xgboost.joblib --output-path data/processed/llm_eval_subset.csv --n-per-class 50
python src/score_baselines_on_eval_subset.py --processed-dir data/processed --models-dir results/models --output-dir results
python src/select_few_shot_examples.py --processed-dir data/processed --model-path results/models/xgboost.joblib --output-path data/processed/few_shot_examples.json
python src/perturbed_subset.py --subset-path data/processed/llm_eval_subset.csv --output-path data/processed/llm_eval_subset_perturbed_numeric.csv --n-cases 20 --seed 42
```

### 4. Run the LLM pipelines (Colab)

1. In Google Drive, create `MyDrive/MSc_Diss/data/` and `MyDrive/MSc_Diss/results/`.
2. Upload `metadata.json`, `llm_eval_subset.csv`, `few_shot_examples.json` and `llm_eval_subset_perturbed_numeric.csv` from `data/processed/` to `MyDrive/MSc_Diss/data/`.
3. Open each notebook in Colab, select a GPU runtime and run all cells:
   - `run_stock_pipeline.ipynb` writes `results/stock_results.jsonl` (500 generations, `max_new_tokens` 400).
   - `run_structured_pipeline.ipynb` writes `results/structured_results.jsonl` (500 generations, `max_new_tokens` 500).
4. Download the two `.jsonl` files into the local `results/` folder.

Runs are resumable. If a Colab session disconnects, re-run the notebook and it will skip every (case, seed) pair already saved.

**Robustness runs.** Upload the contents of `src/` to the Colab session, then run the structured pipeline once per case (seed 0):

```bash
D=/content/drive/MyDrive/MSc_Diss

# Numeric perturbation (20 cases)
python run_llm_pipeline.py --pipeline structured --subset-path $D/data/llm_eval_subset_perturbed_numeric.csv --metadata-path $D/data/metadata.json --few-shot-path $D/data/few_shot_examples.json --output-path $D/results/structured_perturbed_results.jsonl --n-repeats 1 --max-new-tokens 500

# Rewording (100 cases)
python run_llm_pipeline.py --pipeline structured --reworded-profile --subset-path $D/data/llm_eval_subset.csv --metadata-path $D/data/metadata.json --few-shot-path $D/data/few_shot_examples.json --output-path $D/results/structured_reworded_results.jsonl --n-repeats 1 --max-new-tokens 500
```

### 5. Re-validate the structured outputs (local)

The grounding check was corrected after the runs. These commands re-apply the final checks to the saved responses, with no new generation.

```bash
python src/revalidate_results.py --results-path results/structured_results.jsonl --subset-path data/processed/llm_eval_subset.csv --metadata-path data/processed/metadata.json --output-path results/structured_results_revalidated.jsonl
python src/revalidate_results.py --results-path results/structured_perturbed_results.jsonl --subset-path data/processed/llm_eval_subset_perturbed_numeric.csv --metadata-path data/processed/metadata.json --output-path results/structured_perturbed_results_revalidated.jsonl
python src/revalidate_results.py --results-path results/structured_reworded_results.jsonl --subset-path data/processed/llm_eval_subset.csv --metadata-path data/processed/metadata.json --output-path results/structured_reworded_results_revalidated.jsonl
```

### 6. Analysis (local)

```bash
# Accuracy, consistency and parse rates
python src/analyze_results.py --results-path results/stock_results.jsonl
python src/analyze_results.py --results-path results/structured_results_revalidated.jsonl

# Statistical comparison
python src/mcnemar_tests.py --stock-results results/stock_results.jsonl --structured-results results/structured_results_revalidated.jsonl --baseline-predictions results/baseline_per_case_predictions.csv --metadata-path data/processed/metadata.json --output-path results/mcnemar_results.json

# Calibration
python src/compute_calibration.py --stock-results results/stock_results.jsonl --structured-results results/structured_results_revalidated.jsonl --baseline-predictions results/baseline_per_case_predictions.csv --output-dir results

# Explanations
python src/compute_explanations.py --processed-dir data/processed --models-dir results/models --output-dir results
python src/compare_explanations_to_qwen3.py --structured-results results/structured_results_revalidated.jsonl --subset-path data/processed/llm_eval_subset.csv --shap-lime-path results/shap_lime_top_features.json --output-path results/explanation_comparison.json

# Robustness
python src/robustness_analysis.py --original-results results/structured_results_revalidated.jsonl --robustness-results results/structured_perturbed_results_revalidated.jsonl --condition perturbation --original-subset-path data/processed/llm_eval_subset.csv --perturbed-subset-path data/processed/llm_eval_subset_perturbed_numeric.csv
python src/robustness_analysis.py --original-results results/structured_results_revalidated.jsonl --robustness-results results/structured_reworded_results_revalidated.jsonl --condition rewording
```

---

## Expected results

Running the analysis on the included files should give:

| System | Accuracy (100-case subset) | Brier | ECE |
|---|---|---|---|
| XGBoost | 75% | 0.161 | 0.077 |
| Random Forest | 73% | 0.165 | 0.067 |
| Structured Qwen3 | 73% | 0.201 | 0.099 |
| Logistic Regression | 72% | 0.164 | 0.065 |
| Stock Qwen3 | 60% | 0.345 | 0.343 |

- **McNemar, structured vs stock:** χ² = 3.51, raw p = 0.061, Holm p = 0.244. No primary comparison is significant.
- **Robustness (structured):**
  - 1 label change out of 20 perturbed cases.
  - 2 label changes out of 100 reworded cases.
- **Test set (1,481 cases):** the baselines score accuracy 0.747–0.751, F1 0.759–0.766 and ROC-AUC 0.809–0.812.
