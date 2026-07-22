# this script evaluates LLM-generated explanations using the ILORA Evaluation Framework.

import pandas as pd
import os
import glob
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


def normalize_claim(claim: str) -> str:
    return str(claim).lower().strip()


EVAL_PROMPT_FORM = """You are an expert evaluator of LLM-generated explanations.

Evaluate the QUALITY of the explanation according to the ILORA Evaluation Framework.
For each criterion, give a score from 1 to 5 (1 = lowest quality, 5 = highest quality).

CRITERIA:

1. Informativeness (I) - Does the explanation provide new information, such as background knowledge or additional context that helps understand the decision?

2. Logicality (L) - Does the explanation follow a reasonable thought process? Is there a strong causal relationship between the explanation and the result?

3. Objectivity (O) - Is the explanation objective and free from excessive subjective emotion or bias?

4. Readability (R) - Does the explanation follow proper grammatical and structural rules? Are the sentences coherent and easy to understand?

5. Accuracy (A) - Does the generated explanation align with the actual label? Does the explanation accurately reflect the result?

CONTEXT:
Claim: {claim}
Prediction: {prediction}
Justification: {justification}

CRITICAL: You MUST respond ONLY with scores in this EXACT format. Do NOT add any other text before or after:

Informativeness: <your_score_informativeness>
Logicality: <your_score_logicality>
Objectivity: <your_score_objectivity>
Readability: <your_score_readability>
Accuracy: <your_score_acc>

OUTPUT (scores only, nothing else):"""


def evaluate_explanation_form(evaluator_fn, prompt_text, claim_idx, error_log_path):
    prompt = prompt_text
    raw_output = evaluator_fn(prompt)

    scores = {}
    pattern = r"(Informativeness|Logicality|Objectivity|Readability|Accuracy|OverallScore)\s*:\s*([0-9]+(?:\.[0-9]+)?)"
    matches = re.findall(pattern, raw_output)

    if not matches:

        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"CLAIM INDEX: {claim_idx}\n")
            f.write(f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")
            f.write(f"PROMPT:\n{prompt}\n")
            f.write(f"\n{'-'*70}\n")
            f.write(f"MODEL OUTPUT:\n{raw_output}\n")
            f.write(f"{'='*70}\n\n")

        raise ValueError(f"Could not find scores. See {error_log_path}.")

    for key, val in matches:
        scores[key] = float(val)

    keys = ["Informativeness", "Logicality", "Objectivity", "Readability", "Accuracy"]
    if "OverallScore" not in scores:
        scores["OverallScore"] = sum(scores[k] for k in keys) / len(keys)

    return scores


# Retry wrapper for evaluation
def evaluate_with_retry(
    evaluator_fn, prompt_text, claim_idx, error_log_path, max_retries=3, retry_delay=5
):

    for attempt in range(1, max_retries + 1):
        try:

            eval_scores = evaluate_explanation_form(
                evaluator_fn, prompt_text, claim_idx, error_log_path
            )

            if attempt > 1:
                print(f"Succeeded on attempt {attempt}", flush=True)
            return eval_scores

        except Exception as e:

            if attempt < max_retries:
                print(
                    f"Attempt {attempt}/{max_retries} failed: {str(e)[:60]}. Retrying in {retry_delay}s...",
                    flush=True,
                )
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed: {str(e)[:80]}", flush=True)
                return None

    return None


MODEL_ID = "prometheus-eval/prometheus-13b-v1.0"

print("Loading Prometheus 13B model...", flush=True)
prom_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if prom_tokenizer.pad_token is None:
    prom_tokenizer.pad_token = prom_tokenizer.eos_token
prom_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.float16
)
print("Model loaded successfully!\n", flush=True)


def call_prometheus(prompt: str, max_new_tokens: int = 256, temp: float = 0.0):
    inputs = prom_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(prom_model.device)
    attention_mask = inputs["attention_mask"].to(prom_model.device)

    with torch.no_grad():
        outputs = prom_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temp if temp > 0 else None,
            do_sample=temp > 0,
            top_p=1.0 if temp > 0 else None,
            top_k=0 if temp > 0 else None,
            pad_token_id=prom_tokenizer.eos_token_id,
        )

    out_ids = outputs[0][len(input_ids[0]) :]
    response = prom_tokenizer.decode(out_ids, skip_special_tokens=True)

    if "OUTPUT (scores only, nothing else):" in response:
        response = response.split("OUTPUT (scores only, nothing else):")[-1].strip()

    return response


def load_processed_claims(save_path):

    if os.path.exists(save_path):
        try:
            eval_df = pd.read_csv(save_path)
            if "claim" in eval_df.columns:
                processed = set(eval_df["claim"].dropna().unique())
                return processed, len(eval_df)
            else:
                print(f"Warning: 'claim' column not found in {save_path}", flush=True)
                return set(), 0
        except Exception as e:
            print(f"Error reading {save_path}: {e}", flush=True)
            return set(), 0
    return set(), 0


file_mapping = {
    "emerald_few_shot": {
        "baseline": "path_to_emerald_few_shot_baseline_results.csv",
        "rag": "path_to_emerald_few_shot_rag_results.csv",
        "graphrag": "path_to_emerald_few_shot_graphrag_results.csv",
        "hybrid": "path_to_emerald_few_shot_hybrid_results.csv",
    },
    "emerald_zero_shot": {
        "baseline": "path_to_emerald_zero_shot_baseline_results.csv",
        "rag": "path_to_emerald_zero_shot_rag_results.csv",
        "graphrag": "path_to_emerald_zero_shot_graphrag_results.csv",
        "hybrid": "path_to_emerald_zero_shot_hybrid_results.csv",
    },
    "green_few_shot": {
        "baseline": "path_to_green_few_shot_baseline_results.csv",
        "rag": "path_to_green_few_shot_rag_results.csv",
        "graphrag": "path_to_green_few_shot_graphrag_results.csv",
        "hybrid": "path_to_green_few_shot_hybrid_results.csv",
    },
    "green_zero_shot": {
        "baseline": "path_to_green_zero_shot_baseline_results.csv",
        "rag": "path_to_green_zero_shot_rag_results.csv",
        "graphrag": "path_to_green_zero_shot_graphrag_results.csv",
        "hybrid": "path_to_green_zero_shot_hybrid_results.csv",
    },
}


os.makedirs(f"results_ilora", exist_ok=True)


error_log_path = f"results_ilora/parsing_errors.log"
print(f"Error logs will be saved to: {error_log_path}\n", flush=True)


csv_files = []
for setting_name, files in file_mapping.items():
    for pipeline_name, filepath in files.items():
        if pipeline_name != "hybrid":
            csv_files.append((setting_name, pipeline_name, filepath))

print(f"Found {len(csv_files)} files to process\n", flush=True)

for file_idx, (setting_name, pipeline_name, file) in enumerate(csv_files, 1):
    file_name = f"{pipeline_name}_{setting_name}_eval.csv"
    save_path = f"results_ilora_final/{file_name}"

    print(f"{'='*70}", flush=True)
    print(
        f"[File {file_idx}/{len(csv_files)}] {pipeline_name}_{setting_name}", flush=True
    )
    print(f"{'='*70}", flush=True)

    df = pd.read_csv(file)
    total_claims = len(df)
    processed_claims, rows_in_output = load_processed_claims(save_path)

    if rows_in_output > 0:
        print(f"Total claims in source file: {total_claims}", flush=True)
        print(f"Rows in output CSV: {rows_in_output}", flush=True)
        print(f"Unique processed claims: {len(processed_claims)}", flush=True)
        print(f"Remaining: {total_claims - len(processed_claims)}", flush=True)
    else:
        print(f"Total claims in source file: {total_claims}", flush=True)
        print(f"Starting fresh - no previous results", flush=True)

    newly_processed = 0
    skipped_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        claim = row.get("claim")
        prediction = row.get("predicted_label")
        justification = row.get("predicted_justification")

        if not pd.isna(claim) and claim in processed_claims:
            skipped_count += 1
            continue

        if pd.isna(claim) or pd.isna(prediction) or pd.isna(justification):
            continue

        full_prompt = EVAL_PROMPT_FORM.format(
            claim=claim, prediction=prediction, justification=justification
        )

        eval_scores = evaluate_with_retry(
            lambda p: call_prometheus(p, max_new_tokens=512, temp=0.001),
            full_prompt,
            claim_idx=idx,
            error_log_path=error_log_path,
            max_retries=3,
            retry_delay=5,
        )

        if eval_scores is None:
            error_count += 1
            print(
                f"[{error_count} permanent errors] Claim index {idx}: Failed after all retries",
                flush=True,
            )
            continue

        row_dict = {
            "claim": claim,
            "prediction": prediction,
            "justification": justification,
            "Informativeness": eval_scores["Informativeness"],
            "Logicality": eval_scores["Logicality"],
            "Objectivity": eval_scores["Objectivity"],
            "Readability": eval_scores["Readability"],
            "Accuracy": eval_scores["Accuracy"],
            "OverallScore": eval_scores["OverallScore"],
        }

        with open(save_path, "a", newline="", encoding="utf-8") as f:
            row_df = pd.DataFrame([row_dict])
            row_df.to_csv(f, header=f.tell() == 0, index=False)
            f.flush()
            os.fsync(f.fileno())

        processed_claims.add(claim)
        newly_processed += 1

        total_done = len(processed_claims)
        progress_pct = total_done / total_claims * 100
        print(
            f"Total: {total_done}/{total_claims} ({progress_pct:.1f}%)",
            flush=True,
        )

        time.sleep(0.5)

    print(f"\n{'─'*70}", flush=True)
    print(f"Summary for {file_name}:", flush=True)
    print(f"Newly processed this run: {newly_processed}", flush=True)
    print(f"Skipped (already done): {skipped_count}", flush=True)
    print(f"Errors (after all retries): {error_count}", flush=True)
    print(
        f"Total completed: {len(processed_claims)}/{total_claims} ({len(processed_claims)/total_claims*100:.1f}%)",
        flush=True,
    )

    _, final_rows = load_processed_claims(save_path)
    print(f"Verified rows in output file: {final_rows}", flush=True)
    print(f"{'─'*70}\n", flush=True)


print("\n" + "=" * 70, flush=True)
print("PROCESSING HYBRID FILES", flush=True)
print("=" * 70 + "\n", flush=True)

for setting_name, files in file_mapping.items():
    hybrid_file = files.get("hybrid")
    if not hybrid_file or not os.path.exists(hybrid_file):
        print(f"Hybrid file not found for {setting_name}", flush=True)
        continue

    print(f"{'='*70}", flush=True)
    print(f"Processing hybrid_{setting_name}", flush=True)
    print(f"{'='*70}", flush=True)

    df_hybrid = pd.read_csv(hybrid_file)
    df_hybrid["claim_norm"] = df_hybrid["claim"].apply(normalize_claim)

    df_rag = pd.read_csv(files["rag"])
    df_graphrag = pd.read_csv(files["graphrag"])

    df_rag["claim_norm"] = df_rag["claim"].apply(normalize_claim)
    df_graphrag["claim_norm"] = df_graphrag["claim"].apply(normalize_claim)

    file_name = f"hybrid_{setting_name}_eval.csv"
    save_path = f"results_ilora_final/{file_name}"

    total_claims = len(df_hybrid)

    processed_claims, rows_in_output = load_processed_claims(save_path)

    if rows_in_output > 0:
        print(f"Total claims: {total_claims}", flush=True)
        print(f"Rows in output CSV: {rows_in_output}", flush=True)
        print(f"Unique processed claims: {len(processed_claims)}", flush=True)
        print(f"Remaining: {total_claims - len(processed_claims)}", flush=True)
    else:
        print(f"Total claims: {total_claims}", flush=True)
        print(f"Starting fresh - no previous results", flush=True)

    newly_processed = 0
    skipped_count = 0
    error_count = 0

    for idx, row in df_hybrid.iterrows():
        claim = row.get("claim")
        claim_norm = row.get("claim_norm")
        selected_pipeline = row.get("rank_1st")

        if selected_pipeline in ["Baseline", "baseline"]:
            selected_pipeline = row.get("rank_2nd")

        if not pd.isna(claim) and claim in processed_claims:
            skipped_count += 1
            continue

        if pd.isna(claim) or pd.isna(selected_pipeline):
            continue

        if selected_pipeline == "RAG":
            pipeline_df = df_rag
        elif selected_pipeline == "GraphRAG":
            pipeline_df = df_graphrag
        else:

            raise ValueError(f"Unknown pipeline: {selected_pipeline}")

        matched = pipeline_df[pipeline_df["claim_norm"] == claim_norm]
        if matched.empty:
            raise ValueError(f"Claim not found in {selected_pipeline} dataframe")

        matched_row = matched.iloc[0]
        prediction = matched_row.get("predicted_label")
        justification = matched_row.get("predicted_justification")

        if pd.isna(prediction) or pd.isna(justification):
            continue

        full_prompt = EVAL_PROMPT_FORM.format(
            claim=claim, prediction=prediction, justification=justification
        )

        eval_scores = evaluate_with_retry(
            lambda p: call_prometheus(p, max_new_tokens=512, temp=0.001),
            full_prompt,
            claim_idx=idx,
            error_log_path=error_log_path,
            max_retries=3,
            retry_delay=5,
        )

        if eval_scores is None:
            error_count += 1
            print(
                f"[{error_count} permanent errors] Claim index {idx}: Failed after all retries",
                flush=True,
            )
            continue

        row_dict = {
            "claim": claim,
            "selected_pipeline": selected_pipeline,
            "prediction": prediction,
            "justification": justification,
            "Informativeness": eval_scores["Informativeness"],
            "Logicality": eval_scores["Logicality"],
            "Objectivity": eval_scores["Objectivity"],
            "Readability": eval_scores["Readability"],
            "Accuracy": eval_scores["Accuracy"],
            "OverallScore": eval_scores["OverallScore"],
        }

        with open(save_path, "a", newline="", encoding="utf-8") as f:
            row_df = pd.DataFrame([row_dict])
            row_df.to_csv(f, header=f.tell() == 0, index=False)
            f.flush()
            os.fsync(f.fileno())

        processed_claims.add(claim)
        newly_processed += 1

        total_done = len(processed_claims)
        progress_pct = total_done / total_claims * 100
        print(
            f"Total: {total_done}/{total_claims} ({progress_pct:.1f}%)",
            flush=True,
        )

        time.sleep(0.5)

    print(f"\n{'─'*70}", flush=True)
    print(f"Summary for {file_name}:", flush=True)
    print(f"Newly processed this run: {newly_processed}", flush=True)
    print(f"Skipped (already done): {skipped_count}", flush=True)
    print(f"Errors (after all retries): {error_count}", flush=True)
    print(
        f"Total completed: {len(processed_claims)}/{total_claims} ({len(processed_claims)/total_claims*100:.1f}%)",
        flush=True,
    )

    _, final_rows = load_processed_claims(save_path)
    print(f"Verified rows in output file: {final_rows}", flush=True)
    print(f"{'─'*70}\n", flush=True)

print("=" * 70, flush=True)
print(f"Check {error_log_path} for details on any parsing errors.", flush=True)
print("=" * 70, flush=True)
