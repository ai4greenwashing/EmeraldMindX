import pandas as pd
import os
from dotenv import load_dotenv
import tiktoken
import time
import re
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import google.generativeai as genai
import tiktoken

from utils.rag_prompts import zero_shot_prompt, few_shot_prompt


from dotenv import load_dotenv

load_dotenv()


def approximate_token_count(text: str):
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Token count approximation failed: {e}")
        return 0


MODEL_ID = f"google/gemma-3-27b-it"


model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")


DATASETS = ["green_claims", "emerald_data"]
LLM_GUIDELINES = {
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
}
TYPE_INCLUDED = True


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


def call_llm(prompt: str):

    MODEL_ID = f"google/gemma-3-27b-it"

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


for dataset in DATASETS:
    for mode_name, llm_prompt in LLM_GUIDELINES.items():
        save_name = f"baseline_{dataset.split('_')[0]}_{mode_name}"
        SAVE_PATH = f"your_save_path/{save_name}.csv"
        SOURCE_PATH = f"datasets/{dataset}.csv"

        df = pd.read_csv(SOURCE_PATH)

        if os.path.exists(SAVE_PATH):
            result_df = pd.read_csv(SAVE_PATH)
            print(f"Resuming from existing results file with {len(result_df)} rows.")
        else:
            result_df = df.copy()

            for col in [
                "llm_model_pred",
                "llm_response",
                "predicted_label",
                "predicted_justification",
                "full_prompt",
                "predicted_type",
            ]:
                if col not in result_df.columns:
                    result_df[col] = None

        for idx, row in result_df.iterrows():
            if pd.notna(row.get("llm_response")):
                print(f"Skipping row {idx+1}/{len(result_df)} (already processed).")
                continue

            claim = row["claim"]
            company = row["company"]

            print(f"Processing row {idx+1}/{len(result_df)}")

            full_prompt = llm_prompt + f"\n<CLAIM>:\n{claim}\n</CLAIM>"
            response_text, model_used = call_llm(full_prompt)

            if response_text:
                label = extract_label(response_text)
                justification = extract_justification(response_text)

                result_df.at[idx, "llm_response"] = response_text
                result_df.at[idx, "predicted_label"] = label
                result_df.at[idx, "predicted_justification"] = justification

                if TYPE_INCLUDED:
                    result_df.at[idx, "predicted_type"] = extract_type(response_text)
            else:
                result_df.at[
                    idx,
                    [
                        "llm_response",
                        "predicted_label",
                        "predicted_justification",
                    ],
                ] = [None] * 5
                if TYPE_INCLUDED:
                    result_df.at[idx, "predicted_type"] = None

            result_df.to_csv(SAVE_PATH, index=False)
            print(f"Saved results after processing row {idx+1}.\n")

print("\nAll combinations processed and saved in exp_results/")
