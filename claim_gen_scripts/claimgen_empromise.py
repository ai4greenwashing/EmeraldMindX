from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

INPUT_JSON = "Trainset_English.json"


OUTPUT_DIR = "path/to/claims"

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

CATEGORIES = ("environmental", "social", "governance", "n/a")
LABELS = ("greenwashing", "not_greenwashing")


GENERATION_PROMPT = """
You are an ESG Claim Architect and Greenwashing Auditor.

Task: Analyze the provided text from an ESG report. For each core finding, generate one greenwashing and one not_greenwashing claims.

PRIMARY DEFINITION: Greenwashing is defined as one of the following four types:

Type 1: Vague or Misleading Labels – Using terms like "eco-friendly," "green," or "conscious" without specific metrics or definitions.
Type 2: Legal Obligations as Achievements – Presenting mandatory legal compliance (e.g., following a new EU plastic law) as a voluntary sustainability initiative.
Type 3: Overgeneralization / Hidden Trade-offs – Claiming a whole product is "sustainable" based on one small part, or highlighting one green feature while ignoring a massive negative impact.
Type 4: Unsupported Claims (No Proof) – Making environmental claims that lack evidence, third-party verification, or use data out of context to sound better.

Not greenwashing claims are those that do not belong to any of the above categories.
Always mention the company name in the claim text. Do not use pronouns like "we" or "our". Be specific and concise.
Generate two claims per core finding: one that exemplifies greenwashing and one that exemplifies not_greenwashing. Do not generate more than two claims per core finding.

OUTPUT FORMAT:
For each core finding, output two claims in the following format:

Claim 1: [The claim text here]
Label: [greenwashing or not_greenwashing]
Justification: [A brief explanation of why this claim is categorized as greenwashing or not_greenwashing.]
Category: [Environmental, Social, Governance, or N/A]

Claim 2: [The claim text here]
Label: [greenwashing or not_greenwashing]
Justification: [A brief explanation of why this claim is categorized as greenwashing or not_greenwashing.]
Category: [Environmental, Social, Governance, or N/A]

Text to analyze:

"""


CLAIM_PATTERN = re.compile(
    r"Claim(?:\s+\d+)?\s*:\s*(.+?)\s*"
    r"\n\s*Label\s*:\s*(greenwashing|not_greenwashing)\s*"
    r"\n\s*Justification\s*:\s*(.+?)\s*"
    r"\n\s*Category\s*:\s*([A-Za-z/ ]+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_claims(text: str) -> List[Dict[str, str]]:
    claims = []
    for match in CLAIM_PATTERN.finditer(text or ""):
        category = match.group(4).strip().lower()
        claims.append(
            {
                "claim": match.group(1).strip(),
                "gold_label": match.group(2).strip().lower(),
                "justification": match.group(3).strip(),
                "category": category if category in CATEGORIES else "n/a",
            }
        )
    return claims


def company_and_year(record: dict) -> Dict[str, str]:
    url = str(record.get("URL", "") or "")
    stem = os.path.splitext(os.path.basename(urlparse(url).path))[0]
    year = ""
    match = re.search(r"(19|20)\d{2}", stem)
    if match:
        year = match.group()
        stem = stem.replace(year, "")
    return {"company": stem.strip("_- ").lower(), "year": year}


def generate(
    model: str,
    passages: List[str],
    batch_size: int,
    gpu_memory: float,
    max_model_len: int,
    tensor_parallel_size: int,
) -> List[str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": GENERATION_PROMPT + passage}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for passage in passages
    ]

    budget = max_model_len - sampling_params.max_tokens
    lengths = [len(tokenizer(p).input_ids) for p in prompts]
    too_long = sum(1 for n in lengths if n > budget)
    if too_long:
        print(f"  {too_long} passage(s) exceed the context budget and will be skipped")

    completions: List[str] = []
    total_batches = (len(prompts) - 1) // batch_size + 1
    for start in range(0, len(prompts), batch_size):
        window = list(range(start, min(start + batch_size, len(prompts))))
        keep = [i for i in window if lengths[i] <= budget]
        print(f"  batch {start // batch_size + 1}/{total_batches}")

        outputs = {}
        if keep:
            generated = llm.generate([prompts[i] for i in keep], sampling_params)
            outputs = {i: o.outputs[0].text.strip() for i, o in zip(keep, generated)}
        completions.extend(outputs.get(i, "") for i in window)

    return completions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labelled greenwashing claims from ESG passages."
    )
    parser.add_argument(
        "--input",
        default=INPUT_JSON,
        help="JSON list of records, each with a `data` field.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["environmental"],
        choices=list(CATEGORIES) + ["all"],
        help="Categories to keep; the rest go to the filtered-out file.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Limit the records processed (0 = all).",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{args.input}: expected a JSON list of records")

    records = [r for r in records if str(r.get("data", "")).strip()]
    if args.max_records > 0:
        records = records[: args.max_records]
    print(f"Records with a source passage: {len(records)}")

    completions = generate(
        args.model,
        [str(r["data"]) for r in records],
        args.batch_size,
        args.gpu_memory,
        args.max_model_len,
        args.tensor_parallel_size,
    )

    carry = (
        "page_number",
        "promise_status",
        "verification_timeline",
        "evidence_status",
        "evidence_quality",
    )

    columns = [
        "claim",
        "gold_label",
        "company",
        "year",
        "original_text",
        *carry,
        "source_url",
        "justification",
        "category",
    ]

    rows, unparsed = [], 0
    for record, completion in zip(records, completions):
        parsed = parse_claims(completion)
        if not parsed:
            unparsed += 1
            continue
        for claim in parsed:
            rows.append(
                {
                    **claim,
                    **company_and_year(record),
                    "original_text": record["data"],
                    "source_url": record.get("URL", ""),
                    **{key: record.get(key, "") for key in carry},
                }
            )

    if not rows:
        print("No parseable claims were generated.")
        return

    claims = pd.DataFrame(rows, columns=columns)
    keep = (
        claims
        if "all" in args.categories
        else claims[claims["category"].isin(args.categories)]
    )
    dropped = claims.drop(keep.index)

    os.makedirs(args.output_dir, exist_ok=True)
    keep.to_csv(os.path.join(args.output_dir, "claims.csv"), index=False)
    dropped.to_csv(
        os.path.join(args.output_dir, "claims_filtered_out.csv"), index=False
    )

    print(f"\nGenerated {len(claims)} claims, {unparsed} unparseable response(s)")
    print(f"Kept {len(keep)} in {args.categories}, set aside {len(dropped)}")
    print("\nLabel distribution of the kept claims:")
    print(keep["gold_label"].value_counts().to_string())
    print(f"\nOutputs in {os.path.abspath(args.output_dir)}")
    print(
        "\nThese claims are model-generated and need human review before use "
        "as evaluation data."
    )


if __name__ == "__main__":
    main()