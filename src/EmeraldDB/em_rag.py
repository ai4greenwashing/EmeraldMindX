import sys
import os
import re
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import transformers
import subprocess
import chromadb
from chromadb.utils import embedding_functions
import warnings

transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
load_dotenv()

def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return root
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_git_root()
print(f"Project Root detected as: {PROJECT_ROOT}")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL = "27b"
MODEL_ID = f"google/gemma-3-{MODEL}-it"

zero_shot_prompt = """
You are an expert in ESG (Environmental, Social, and Governance) analysis.
Classify the following claim regarding its greenwashing status.
Output Format:
Label: [greenwashing | not_greenwashing | abstain]
Justification: [Reasoning]
Type: [Type 1 | Type 2 | Type 3 | Type 4 | N/A]
"""

few_shot_prompt = """
You are an expert in ESG analysis. Here are examples of greenwashing classification.
[Insert Examples Here]
Classify the following claim.
Output Format:
Label: [greenwashing | not_greenwashing | abstain]
Justification: [Reasoning]
Type: [Type 1 | Type 2 | Type 3 | Type 4 | N/A]
"""

class ReportParser:
    def __init__(self, db_path, embedding_model_name="sentence-transformers/all-mpnet-base-v2"):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(
            name="esg_reports", 
            embedding_function=self.ef
        )

    def query(self, query_text, n_results=5, company_filter=None):
        query_args = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if company_filter:
            query_args["where"] = {"company": company_filter}
        return self.collection.query(**query_args)

print(f"Loading Model: {MODEL_ID}")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

parser = ReportParser(
    db_path=DB_PATH,
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)

def approximate_token_count(text: str) -> int:
    return len(text) // 4

def call_llm(prompt: str):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, temperature=1e-5, max_new_tokens=512)

    generated = output_ids[0][input_len:]
    response_text = processor.decode(generated, skip_special_tokens=True)
    return response_text.strip(), MODEL_ID

def extract_label(text: str) -> str:
    match = re.search(
        r"Label:\s*(greenwashing|not_greenwashing|abstain)", text, re.IGNORECASE
    )
    return match.group(1).strip().lower() if match else None

def extract_type(text: str) -> str:
    match = re.search(r"Type:\s*(Type\s*[1-4]|N/?A)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_justification(text: str) -> str:
    match = re.search(
        r"Justification:\s*(.+?)(?:\n\s*\n|$)", text, re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else None

def normalize_company_name(name: str) -> str:
    reports_folder = Path(REPORTS_DIR)
    companies = {
        "_".join(f.stem.split("_")[:-1])
        for f in reports_folder.glob("*.pdf")
        if len(f.stem.split("_")) >= 2
    }
    
    if not companies:
        txt_companies = {
            "_".join(f.stem.split("_")[:-2])
            for f in reports_folder.glob("*.txt")
            if len(f.stem.split("_")) >= 3
        }
        companies.update(txt_companies)

    if not companies:
        return name

    input_emb = embedding_model.encode([name])[0]
    company_embs = embedding_model.encode(list(companies))
    sims = np.dot(company_embs, input_emb) / (
        np.linalg.norm(company_embs, axis=1) * np.linalg.norm(input_emb)
    )
    best_idx = np.argmax(sims)
    return list(companies)[best_idx]

EXPERIMENTS = [
    ("rag_green", zero_shot_prompt, "zero_shot"),
    ("rag_green", few_shot_prompt, "few_shot"),
    ("rag_emerald", zero_shot_prompt, "zero_shot"),
    ("rag_emerald", few_shot_prompt, "few_shot"),
]

TYPE_INCLUDED = True
DATASETS = {
    "green": "GreenClaims",
    "emerald": "EmeraldData",
}

for prefix, llm_guidelines, mode in EXPERIMENTS:
    dataset_key = "green" if "green" in prefix else "emerald"
    dataset_name = DATASETS[dataset_key]
    
    save_df = f"rag_{dataset_key}_{mode}.csv"
    SAVE_PATH = os.path.join(RESULTS_DIR, save_df)
    INPUT_PATH = os.path.join(DATA_DIR, f"{dataset_name}.csv")

    if not os.path.exists(INPUT_PATH):
        if os.path.exists(f"{dataset_name}.csv"):
             INPUT_PATH = f"{dataset_name}.csv"
        else:
            print(f"Skipping {dataset_name}: File not found at {INPUT_PATH}")
            continue

    if os.path.exists(SAVE_PATH):
        df = pd.read_csv(SAVE_PATH)
        print(f"Resuming from {SAVE_PATH} ({len(df)} rows).")
    else:
        df = pd.read_csv(INPUT_PATH)
        df["llm_model_pred"] = None
        df["llm_response"] = None
        df["predicted_label"] = None
        df["predicted_justification"] = None
        df["full_prompt"] = None
        df["snippets_used"] = None
        if TYPE_INCLUDED:
            df["predicted_type"] = None

    for index, row in df.iterrows():
        if pd.notna(row.get("predicted_label")):
            continue

        claim = row["claim"]
        company = row["company"]

        normalized_company = normalize_company_name(company)

        print(
            f"\nProcessing claim {index + 1}/{len(df)}: {claim} (Company: {normalized_company})"
        )

        results = parser.query(claim, n_results=8, company_filter=normalized_company)
        
        if results and results["documents"] and len(results["documents"]) > 0:
            context = [
                f"---Snippet {i+1}---\n{doc}"
                for i, doc in enumerate(results["documents"][0])
            ]
        else:
            context = []

        llm_prompt_base = (
            llm_guidelines
            + "\n\n<CLAIM>\n"
            + claim
            + "\n\n</CLAIM>\n"
            + "\n\n<CONTEXT>\n"
        )
        remaining_snippets = context.copy()

        while remaining_snippets:
            context_text = "\n\n".join(remaining_snippets)
            full_prompt = llm_prompt_base + context_text
            if approximate_token_count(full_prompt) <= 120_000:
                break
            remaining_snippets.pop()

        if not remaining_snippets:
            llm_response = None
            final_context_used = 0
            if not context: 
                print("No context found in DB.")
        else:
            context_text = "\n".join(remaining_snippets) + "\n\n</CONTEXT>\n"
            full_prompt = llm_prompt_base + context_text
            llm_response, model_used = call_llm(full_prompt)
            final_context_used = len(remaining_snippets)

        if llm_response:
            df.at[index, "llm_model_pred"] = model_used
            df.at[index, "llm_response"] = llm_response
            df.at[index, "predicted_label"] = extract_label(llm_response)
            df.at[index, "predicted_justification"] = extract_justification(
                llm_response
            )
            df.at[index, "full_prompt"] = full_prompt
            df.at[index, "snippets_used"] = final_context_used
            if TYPE_INCLUDED:
                df.at[index, "predicted_type"] = extract_type(llm_response)
        else:
            for col in [
                "llm_model_pred",
                "llm_response",
                "predicted_label",
                "predicted_justification",
                "full_prompt",
                "snippets_used",
            ]:
                df.at[index, col] = None
            if TYPE_INCLUDED:
                df.at[index, "predicted_type"] = None

        df.to_csv(SAVE_PATH, index=False, escapechar="\\")
        print(f"Progress saved to {SAVE_PATH}")

    print(f"\nFinished experiment: {save_df}")