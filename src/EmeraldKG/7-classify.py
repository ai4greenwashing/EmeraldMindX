from __future__ import annotations

import os
import json
import argparse
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import pandas as pd
from neo4j import GraphDatabase, Result
from graphdatascience import GraphDataScience
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from google import genai
import re, unicodedata
from rapidfuzz import fuzz, process, utils
from numpy import dot
from numpy.linalg import norm
import numpy as np
from dotenv import load_dotenv
from google.genai import types
import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.classification_utils import get_company_id, retrieve_evidence, retrieve_evidence_with_paths
from utils.classification_utils import classify_few, classify_zero, MODEL_NAME, classify_zero_python, classify_few_python

load_dotenv()
try:
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "emeraldmind"))
    driver.verify_connectivity()
    gds = GraphDataScience(driver)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Neo4j: {e}")
encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

POS_LABEL = "greenwashing"
NEG_LABEL = "not_greenwashing"
ABSTAIN_LABEL = "abstain"


def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return Path(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[2]
    

PROJECT_ROOT = get_git_root()
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_evaluation(
    dataset: str = "mixed",
    prompt: str = "few",
    evidence: str = "json",
    hops: str = "one",
    top_k_per_class: int = 10,
    similarity_threshold: float = 0.3,
    embeddings_dir: str = None,
    retry_errors_only: bool = True
) -> None:
    
    """
    Run evaluation with configurable parameters.
    Handles 'abstain' label by excluding it from metrics calculation.
    
    Args:
        dataset: Dataset size - 'mixed', 'small'
        prompt: Prompting strategy - 'few' for few-shot or 'zero' for zero-shot
        hops: Retrieval strategy - 'one' for single-hop or 'multi' for multi-hop with paths
        top_k_per_class: Number of most similar nodes to retrieve per class
        similarity_threshold: Minimum similarity threshold for node selection
        embeddings_dir: Directory containing claim_{i}.json files (if None, uses {dataset})
        retry_errors_only: If True, loads existing output and only retries rows with "error" in llm_raw_label
    """
    if embeddings_dir is None:
        embeddings_dir = f"{dataset}"
    
    csv_path = DATASETS_DIR / f"{dataset}.csv"

    if prompt == "few":
        if evidence == "python":
            classify_fn = classify_few_python
        else:
            classify_fn = classify_few
        prompt_type = "few"
    elif prompt == "zero":
        if evidence == "python":
            classify_fn = classify_zero_python
        else:
            classify_fn = classify_zero
        prompt_type = "zero"
    else:
        raise ValueError(f"Invalid prompt type: {prompt}. Must be 'few' or 'zero'")
    
    if hops == "one":
        retrieve_fn = retrieve_evidence
        hops_type = "one"
    elif hops == "multi":
        retrieve_fn = retrieve_evidence_with_paths
        hops_type = "multi"
    else:
        raise ValueError(f"Invalid hops type: {hops}. Must be 'one' or 'multi'")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"CSV path: {csv_path}")
    print(f"Prompt strategy: {prompt_type}")
    print(f"Evidence format: {evidence}")
    print(f"Retrieval strategy: {hops_type}")
    print(f"Top-K per class: {top_k_per_class}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Retry errors only: {retry_errors_only}")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(csv_path)
    
    if 'unique_id' not in df.columns:
        print("Warning: 'unique_id' column not found in CSV. Adding it based on row index.")
        df['unique_id'] = df.index
    
    required = {"company", "claim", "label"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain: {', '.join(required)}")

    out_filename = f"{dataset}_{prompt_type}_shot_{hops_type}_hop_{top_k_per_class}_{similarity_threshold}_{embeddings_dir}_{MODEL_NAME}_{evidence}.csv"
    out_path = RESULTS_DIR / out_filename
    existing_results = None
    rows_to_process = None
    
    if retry_errors_only and out_path.exists():
        print(f"\n{'='*60}")
        print(f"RETRY MODE: Loading existing results from {out_path}")
        print(f"{'='*60}\n")
        
        existing_results = pd.read_csv(out_path)
        print(f"Loaded {len(existing_results)} existing results")
        
        if 'llm_raw_label' in existing_results.columns:
            error_mask = existing_results['llm_raw_label'].astype(str).str.contains('error', case=False, na=False)
            error_rows = existing_results[error_mask]
            print(f"Found {len(error_rows)} rows with 'error' in llm_raw_label")
            
            if len(error_rows) == 0:
                print("\nNo error rows found. Nothing to retry.")
                return
            
            if 'unique_id' in error_rows.columns:
                error_ids = set(error_rows['unique_id'].tolist())
                rows_to_process = df[df['unique_id'].isin(error_ids)].copy()
                print(f"Retrying {len(rows_to_process)} rows based on unique_id match")
            else:
                error_indices = error_rows.index.tolist()
                rows_to_process = df.iloc[error_indices].copy()
                print(f"Retrying {len(rows_to_process)} rows based on index match")
        else:
            print("Warning: 'llm_raw_label' column not found in existing results.")
            print("Processing all rows as fallback.")
            rows_to_process = df
    else:
        if retry_errors_only:
            print(f"\nOutput file {out_path} does not exist. Processing all rows.")
        rows_to_process = df

    y_true: List[str] = []
    y_pred: List[str] = []
    rows_to_log: List[Dict[str, Any]] = []
    abstain_count = 0
    retry_count = 0
    

    try:
        for row in rows_to_process.itertuples():
            if pd.notna(row.company):
                company = str(row.company).strip() if pd.notna(row.company) else ""
            else:
                print(f"Skipping row {row.Index}: company name is missing.")
                rows_to_log.append(
                    {
                        "company": None,
                        "claim": str(row.claim),
                        "context": "(Skipped: company name missing)",
                        "llm_label": None,
                        "llm_type": None,
                        "llm_raw_label": "skipped_no_company_name",
                        "llm_reason": "Skipped due to missing company name.",
                        "cited_node_ids": [],
                        "gold_label": None,
                        "label_raw": (
                            str(row.label)
                            if pd.notna(row.label)
                            else None
                        ),
                        "abstained": False
                    }
                )
                continue

            claim = str(row.claim).strip() if pd.notna(row.claim) else "No claim text"
            raw_response_llm = ""
            
            
            if 'unique_id' in df.columns and pd.notna(row.unique_id):
                unique_id = int(row.unique_id)
            else:
                unique_id = row.Index if hasattr(row, 'Index') and row.Index >= 0 else getattr(row, 'Index', 0)
                if unique_id < 0:
                    unique_id = len(rows_to_log)
            
            retry_count += 1
            print(f"\n[RETRY {retry_count}/{len(rows_to_process)}] Processing row {row.Index}: unique_id={unique_id}, company={company}")
            
            label_raw = (
                str(row.label).strip().lower()
                if pd.notna(row.label)
                else "missing"
            )
            gold_label_normalized = label_raw
            if gold_label_normalized not in (POS_LABEL, NEG_LABEL):
                print(
                    f"Warning: Row {row.Index} has unexpected label: '{label_raw}'. Normalizing to '{NEG_LABEL}'."
                )
                gold_label_normalized = NEG_LABEL

            llm_raw_label = None
            pred_label_normalized = None
            abstained = False
            comp_id = get_company_id(company)
            if comp_id is None:
                print(f" {company} not found in Neo4j; predicting 'insufficient_data'.")
                llm_raw_label = "no_company_id"
                pred_label_normalized = ABSTAIN_LABEL
                llm_reason = "company not found in knowledge graph."
                cited_node_ids = []
                context_str = "(No company ID found)"
                raw_response="(No company ID found)"
                llm_type = "no_company_id"
            else:
                try:
                    evidence = retrieve_fn(
                        comp_id, 
                        claim, 
                        claim_idx=unique_id,
                        top_k_per_class=top_k_per_class,
                        similarity_threshold=similarity_threshold,
                        embeddings_dir=embeddings_dir
                    )
                except Exception as e:
                    print(f"Error during evidence retrieval for row {row.Index}: {e}")
                    evidence = []

                if not evidence:
                    llm_raw_label = "no_context"
                    pred_label_normalized = ABSTAIN_LABEL
                    llm_reason = "No relevant context found in knowledge graph."
                    cited_node_ids = []
                    context_str = "(No context found after retrieval attempts)"
                    raw_response = "(No context found)"
                    llm_type = "no_context"
                    print(
                        f"No context found for row {row.Index}, defaulting to {pred_label_normalized}"
                    )
                else:
                    try:
                        resp, context_str, raw_response_llm = classify_fn(evidence, company, claim)
                        llm_raw_label = (
                            resp.get("label", "no_label_from_llm").lower().strip()
                        )
                        llm_type = resp.get("type", "unknown_model").lower().strip()
                        raw_response = resp 
                        if raw_response is None:
                            raw_response = "(No response from LLM)"

                        pred_label_normalized = llm_raw_label
                        
                        # Handle abstain separately - don't normalize it
                        if pred_label_normalized == ABSTAIN_LABEL:
                            print(f"LLM abstained for row {row.Index}")
                            abstained = True
                            abstain_count += 1
                            # Don't include in metrics
                        elif pred_label_normalized not in (POS_LABEL, NEG_LABEL):
                            print(
                                f"Warning: LLM returned unexpected label '{llm_raw_label}' for row {row.Index}. Normalizing to '{ABSTAIN_LABEL}'."
                            )
                            pred_label_normalized = ABSTAIN_LABEL
                        
                        llm_reason = resp.get("reasoning", "")
                        cited_node_ids = resp.get("cited_node_ids", [])
                    except Exception as e:
                        print(f"Hard failure on Gemini for row {row.Index}: {e}")
                        llm_reason = f"LLM call failed: {e}"
                        cited_node_ids = []
                        llm_raw_label = "llm_error"
                        llm_type = "llm_error"
                        pred_label_normalized = ABSTAIN_LABEL
                        context_str = "(Error during LLM classification)"
                        raw_response = "(Error during LLM classification)"

            if not abstained and gold_label_normalized in (POS_LABEL, NEG_LABEL) and pred_label_normalized in (POS_LABEL, NEG_LABEL):
                y_true.append(gold_label_normalized)
                y_pred.append(pred_label_normalized)
            else:
                if abstained:
                    print(f"Row {row.Index} abstained - excluded from metrics")
                else:
                    print(
                        f"Skipping row {row.Index} for metrics due to non-binary label: Gold='{gold_label_normalized}', Pred='{pred_label_normalized}'"
                    )

            rows_to_log.append(
                {
                    "company": company,
                    "claim": claim,
                    "context": context_str,
                    "llm_label": pred_label_normalized,
                    "llm_type": llm_type,
                    "llm_raw_label": llm_raw_label,
                    "llm_reason": llm_reason,
                    "cited_node_ids": cited_node_ids,
                    "gold_label": gold_label_normalized,
                    "label_raw": label_raw,
                    "abstained": abstained,
                    "unique_id": unique_id,
                    "llm_full_response": raw_response_llm,
                }
            )
            print(
                f"{company} (Row {row.Index}): gold={gold_label_normalized}, pred={pred_label_normalized}{' [ABSTAINED]' if abstained else ''}"
            )
    finally:
        driver.close()

    # Merge results if we were in retry mode
    if existing_results is not None and len(rows_to_log) > 0:
        print(f"\n{'='*60}")
        print(f"MERGING RESULTS")
        print(f"{'='*60}\n")
        
        retried_df = pd.DataFrame(rows_to_log)
        
        # Create a mapping of unique_id to new results
        if 'unique_id' in retried_df.columns and 'unique_id' in existing_results.columns:
            # Use unique_id for merging
            print(f"Merging based on unique_id column")
            
            # Remove error rows from existing results
            error_ids = set(retried_df['unique_id'].tolist())
            non_error_results = existing_results[~existing_results['unique_id'].isin(error_ids)]
            
            print(f"Keeping {len(non_error_results)} non-error rows from existing results")
            print(f"Adding {len(retried_df)} newly processed rows")
            
            # Combine and sort by unique_id
            df_log = pd.concat([non_error_results, retried_df], ignore_index=True)
            df_log = df_log.sort_values('unique_id').reset_index(drop=True)
        else:
            # if no unique_id use index-based merging
            print(f"Merging based on DataFrame index")
            existing_results_copy = existing_results.copy()
            
            for new_row in rows_to_log:
                idx = new_row.get('unique_id', None)
                if idx is not None and idx < len(existing_results_copy):
                    for col, val in new_row.items():
                        if col in existing_results_copy.columns:
                            existing_results_copy.loc[idx, col] = val
            
            df_log = existing_results_copy
        
        print(f"Final dataset has {len(df_log)} rows")
        
        y_true = []
        y_pred = []
        abstain_count = 0
        
        for _, row in df_log.iterrows():
            if pd.notna(row.get('gold_label')) and pd.notna(row.get('llm_label')):
                gold = str(row['gold_label']).strip().lower()
                pred = str(row['llm_label']).strip().lower()
                abstained = row.get('abstained', False)
                
                if abstained or pred == ABSTAIN_LABEL:
                    abstain_count += 1
                elif gold in (POS_LABEL, NEG_LABEL) and pred in (POS_LABEL, NEG_LABEL):
                    y_true.append(gold)
                    y_pred.append(pred)
    else:
        df_log = pd.DataFrame(rows_to_log)

    if not y_true or not y_pred:
        print(
            "Error: No valid binary labels found for evaluation. Skipping metrics calculation."
        )
        df_log.to_csv(out_path, index=False)
        print(f"\nFull log written to {out_path.resolve()}")
        print(f"Total abstained: {abstain_count}")
        return

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=POS_LABEL,
        zero_division=0,
    )
    
    print("\nLogged dataframe preview:")
    print(df_log.head(3).to_markdown(index=False))

    df_log.to_csv(out_path, index=False)
    print(f"\nFull log written to {out_path.resolve()}")
    print("\nEvaluation Summary")
    print("==================")
    print(f"Configuration: {dataset} dataset, {prompt_type}-shot prompting, {hops_type}-hop retrieval")
    print(f"Total rows in output: {len(df_log)}")
    print(f"Rows retried: {retry_count}")
    print(f"Abstained: {abstain_count}")
    print(f"Evaluated: {len(y_true)} (excluding abstained)")
    print(f"Accuracy : {acc:.3f} (on {len(y_true)} non-abstained samples)")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 score : {f1:.3f}")
    print("\nDetailed report:\n")
    print(classification_report(y_true, y_pred, digits=3))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run greenwashing classification evaluation with configurable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["emerald_data", "green_claims"],
        default="emerald_data",
        help="Dataset to use (determines CSV file and default embeddings directory)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["few", "zero"],
        default="few",
        help="Prompting strategy: 'few' for few-shot or 'zero' for zero-shot"
    )
    parser.add_argument(
        "--evidence",
        type=str,
        choices=["json", "python"],
        default="json",
        help="Evidence format: 'json' for JSON context, 'python' for Python code context"
    )
    parser.add_argument(
        "--hops",
        type=str,
        choices=["one", "multi"],
        default="one",
        help="Retrieval strategy: 'one' for single-hop or 'multi' for multi-hop with paths"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        dest="top_k_per_class",
        help="Number of most similar nodes to retrieve per class"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        dest="similarity_threshold",
        help="Minimum similarity threshold for node selection"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Directory containing claim_{{i}}.json files (default: {{dataset}})"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        default=False,
        dest="retry_errors_only",
        help="Only retry rows with errors in existing output file"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_evaluation(
        dataset=args.dataset,
        prompt=args.prompt,
        evidence=args.evidence,
        hops=args.hops,
        top_k_per_class=args.top_k_per_class,
        similarity_threshold=args.similarity_threshold,
        embeddings_dir=args.embeddings_dir,
        retry_errors_only=args.retry_errors_only
    )