from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple

import pandas as pd

from vectordb import EMBEDDING_MODEL, ReportStore

CLAIMS_CSV = "path/to/claims.csv"


CHROMADB_PATH = "path/to/chromadb"


REPORTS_DIR = "path/to/reports"

OUTPUT_CSV = "path/to/pipeline_outputs/rag.csv"

DEFAULT_MODEL = "google/gemma-3-27b-it"

TOP_M = 8
MAX_MODEL_LEN = 8192
MAX_OUTPUT_TOKENS = 512
BATCH_SIZE = 64


MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["greenwashing", "not_greenwashing", "abstain"],
        },
        "type": {
            "type": "string",
            "enum": ["Type 1", "Type 2", "Type 3", "Type 4", "N/A"],
        },
        "justification": {"type": "string"},
    },
    "required": ["label", "type", "justification"],
    "additionalProperties": False,
}


FEW_SHOT_PROMPT = """

You are an **ESG (Environmental, Social, and Governance) and Greenwashing Fact-Checker.**

**Task:** Given an ESG-related claim, fact-check it and determine whether it constitutes **greenwashing**, **not_greenwashing**, or if you should **abstain**.

### Fact-Checking Process

**When context is provided below:** Use the provided context as your PRIMARY source of truth. Cross-reference claims against specific information in the context.

**When no context is provided:** Use your internal knowledge to verify factual accuracy. You may only make determinations based on facts you can confidently recall.

**In BOTH cases:** Your decision MUST be based on verifiable facts that either prove or disprove the claim. NEVER judge based on:
- Vague wording or marketing language
- How "suspicious" or "too good to be true" something sounds  
- Absence of detail (this warrants abstain, not greenwashing)
- Your intuition or reasoning about what "could be" misleading

### Decision Criteria

**Greenwashing** - You have specific evidence that the claim is factually false or misleading:
1. **Type 1 - Factually false labels:** You can verify the label/certification is false, doesn't exist, or misrepresents actual standards.
2. **Type 2 - Legal obligations misrepresented as voluntary:** You can verify the action was legally mandated, not a voluntary initiative.
3. **Type 3 - Partial truth presented as whole:** You can verify only part of the product/service meets the stated criteria.
4. **Type 4 - Claims contradicted by evidence:** You have specific facts/evidence that directly contradict the claim.

**Not Greenwashing** - You can verify the claim is factually accurate based on available information (context or knowledge).

**Abstain** - You cannot verify factual accuracy either way. Use this when:
- No context is provided AND you lack sufficient internal knowledge
- Context is provided but doesn't contain information to verify/refute the specific claim
- You cannot confidently determine if the claim is true or false

### FEW-SHOT EXAMPLES

<EXAMPLE 1: No context provided, insufficient internal knowledge>
<CONTEXT>
[No context provided]
</CONTEXT>
<CLAIM>H&M promotes its new "Conscious Collection" as a sustainable fashion line.</CLAIM>
<OUTPUT>
{"label": "abstain", "type": "N/A", "justification": "No context provided. While I know H&M has a 'Conscious Collection,' I do not have specific, verifiable information about the materials, certifications, or manufacturing processes used in this collection to determine if the sustainability claims are factually accurate or false."}
</OUTPUT>
</EXAMPLE 1>
---
<EXAMPLE 2: No context provided, using internal knowledge>
<CONTEXT>
[No context provided]
</CONTEXT>
<CLAIM>McDonald's announces it is reducing plastic waste by introducing reusable cutlery in France.</CLAIM>
<OUTPUT>
{"label": "greenwashing", "type": "Type 2", "justification": "Based on internal knowledge: The French Anti-Waste Law (AGEC), passed in February 2020, mandates that fast-food restaurants must use reusable tableware for on-site dining as of January 1, 2023. McDonald's is legally required to provide reusable cutlery in France, making this a legal compliance issue rather than a voluntary sustainability initiative."}
</OUTPUT>
</EXAMPLE 2>
---
<EXAMPLE 3: Context provided, contradicts claim>
<CONTEXT>
Product manufacturing report: "Walmart's bamboo towel line uses bamboo viscose, produced through a chemical process involving carbon disulfide (CS2) and sodium hydroxide (NaOH). The manufacturing process generates significant chemical waste and requires substantial water usage."
</CONTEXT>
<CLAIM>Walmart advertises its "eco-friendly" bamboo towels as fully sustainable.</CLAIM>
<OUTPUT>
{"label": "greenwashing", "type": "Type 3", "justification": "The provided context shows that while bamboo is the raw material, the manufacturing process involves 'carbon disulfide (CS2) and sodium hydroxide (NaOH)' and 'generates significant chemical waste and requires substantial water usage.' The claim of 'fully sustainable' is factually inaccurate as it ignores the environmentally harmful manufacturing process documented in the context."}
</OUTPUT>
</EXAMPLE 3>
---
<EXAMPLE 4: Context provided, supports claim with evidence>
<CONTEXT>
Regulatory filing: "Canada's Competition Bureau issued a penalty to Keurig Canada in January 2022 for making false or misleading claims about the recyclability of its K-Cup pods. Investigation found that the majority of Canadian municipalities do not accept K-Cup pods in curbside recycling programs due to their multi-material composition."
</CONTEXT>
<CLAIM>Keurig states that all of its single-use coffee pods are "easily recyclable in any city recycling program."</CLAIM>
<OUTPUT>
{"label": "greenwashing", "type": "Type 4", "justification": "The context directly contradicts the claim, stating that 'the majority of Canadian municipalities do not accept K-Cup pods in curbside recycling programs' and that Keurig was penalized by Canada's Competition Bureau in 2022 'for making false or misleading claims about the recyclability' of these pods."}
</OUTPUT>
</EXAMPLE 4>
---
<EXAMPLE 5: Context provided, verifies claim>
<CONTEXT>
Energy audit report: "Tesla's Gigafactory 1 in Nevada operates with a 70 MW solar array installed on its roof, supplemented by renewable energy purchased through power purchase agreements (PPAs). As of Q4 2023, 100% of the facility's energy consumption comes from renewable sources, verified by third-party auditor SGS."
</CONTEXT>
<CLAIM>Tesla reports that its Gigafactory in Nevada now runs entirely on renewable energy.</CLAIM>
<OUTPUT>
{"label": "not_greenwashing", "type": "N/A", "justification": "The context verifies the claim with specific details: '70 MW solar array' on-site, renewable energy PPAs, and third-party verification by SGS confirming '100% of the facility's energy consumption comes from renewable sources.' The claim is factually accurate based on documented evidence."}
</OUTPUT>
</EXAMPLE 5>
---
<EXAMPLE 6: Context provided but insufficient>
<CONTEXT>
Press release: "GreenCorp announces its new line of eco-friendly cleaning products launching this quarter."
</CONTEXT>
<CLAIM>GreenCorp's new cleaning products are made from 100% biodegradable ingredients.</CLAIM>
<OUTPUT>
{"label": "abstain", "type": "N/A", "justification": "The provided context mentions 'eco-friendly cleaning products' but contains no specific information about ingredients, biodegradability testing, or product composition. Without this information, I cannot verify whether the '100% biodegradable ingredients' claim is factually accurate or false."}
</OUTPUT>
</EXAMPLE 6>

### Output Format

Respond with ONLY a single valid JSON object and nothing else (no preamble, no code fences, no trailing text). The JSON object MUST have exactly these three fields:

{
  "label": "greenwashing" | "not_greenwashing" | "abstain",
  "type": "Type 1" | "Type 2" | "Type 3" | "Type 4" | "N/A",
  "justification": "your reasoning here"
}

Rules:
- `label` must be exactly one of: "greenwashing", "not_greenwashing", "abstain".
- `type` must be exactly one of: "Type 1", "Type 2", "Type 3", "Type 4", "N/A". Use "N/A" whenever `label` is "not_greenwashing" or "abstain".
- `justification` is a single string explaining your reasoning.

"""


def assemble_prompt(claim: str, snippets: List[str]) -> str:
    return (
        FEW_SHOT_PROMPT.rstrip()
        + "\n\n<CONTEXT>\n"
        + "\n\n".join(snippets)
        + "\n</CONTEXT>\n\n<CLAIM>\n"
        + claim
        + "\n</CLAIM>\n"
    )


def clean(text: str) -> str:
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\n\t")
    return text.replace("\xa0", " ")


def approximate_tokens(text: str) -> int:
    return len(text) // 4


class CompanyMatcher:

    def __init__(self, reports_dir: str, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.companies = sorted(
            {
                re.sub(r"(_\d{4})+$", "", os.path.splitext(name)[0])
                for name in os.listdir(reports_dir)
                if name.lower().endswith((".pdf", ".txt"))
            }
        )
        if not self.companies:
            raise ValueError(f"No reports found in {reports_dir}")

        self.model = SentenceTransformer(model)
        self.embeddings = self.model.encode(self.companies, normalize_embeddings=True)
        print(f"Matching against {len(self.companies)} company name(s)")

    def match(self, name: str) -> str:
        import numpy as np

        query = self.model.encode([str(name)], normalize_embeddings=True)[0]
        return self.companies[int(np.argmax(self.embeddings @ query))]


def retrieve(
    store: ReportStore, matcher: CompanyMatcher, claim: str, company: str, top_m: int
) -> Tuple[str, List[str]]:
    matched = matcher.match(company)
    documents = store.query(claim, n_results=top_m, company=matched)
    snippets = [f"---Snippet {i + 1}---\n{clean(d)}" for i, d in enumerate(documents)]
    return matched, snippets


def fit_to_budget(claim: str, snippets: List[str]) -> List[str]:

    kept = list(snippets)
    while kept and approximate_tokens(assemble_prompt(claim, kept)) > MAX_PROMPT_TOKENS:
        kept.pop()
    return kept


def parse_verdict(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not text:
        return None, None, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, None, None
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return None, None, None

    if not isinstance(payload, dict):
        return None, None, None

    def text_field(key: str, lower: bool = False) -> Optional[str]:
        value = payload.get(key)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value.lower() if lower else value

    return (
        text_field("label", lower=True),
        text_field("type"),
        text_field("justification"),
    )


OUTPUT_COLUMNS = [
    "claim",
    "gold_label",
    "predicted_label",
    "justification",
    "predicted_type",
    "company",
    "matched_company",
    "snippets_used",
    "llm_model",
    "llm_response",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify claims against retrieved ESG report text."
    )
    parser.add_argument("--claims", default=CLAIMS_CSV)
    parser.add_argument("--db-path", default=CHROMADB_PATH)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--top-m", type=int, default=TOP_M)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory", type=float, default=0.7)
    arguments = parser.parse_args()

    claims = pd.read_csv(arguments.claims)
    for column in ("claim", "company"):
        if column not in claims.columns:
            raise ValueError(f"{arguments.claims}: missing column {column!r}")

    if os.path.exists(arguments.output):
        results = pd.read_csv(arguments.output)
        print(f"Resuming from {arguments.output} ({len(results)} rows)")
    else:
        results = claims.copy()
    for column in OUTPUT_COLUMNS:
        if column not in results.columns:
            results[column] = None
        results[column] = results[column].astype(object)

    pending = [
        i for i, row in results.iterrows() if pd.isna(row.get("predicted_label"))
    ]
    print(f"{len(pending)} of {len(results)} rows pending")
    if not pending:
        return

    store = ReportStore(arguments.db_path, arguments.embedding_model)
    matcher = CompanyMatcher(arguments.reports_dir)

    print("Retrieving context")
    prepared, without_context = [], 0
    for index in pending:
        claim = results.at[index, "claim"]
        matched, snippets = retrieve(
            store, matcher, claim, results.at[index, "company"], arguments.top_m
        )
        snippets = fit_to_budget(claim, snippets)
        if not snippets:
            without_context += 1
            continue
        prepared.append(
            {
                "index": index,
                "matched": matched,
                "snippets": len(snippets),
                "prompt": assemble_prompt(claim, snippets),
            }
        )
    print(
        f"  {len(prepared)} prompts built, "
        f"{without_context} row(s) had no retrievable context"
    )

    if not prepared:
        results.to_csv(arguments.output, index=False)
        return

    from vllm import LLM
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    print(f"Loading {arguments.model}")
    llm = LLM(
        model=arguments.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=arguments.gpu_memory,
        max_model_len=arguments.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=MAX_OUTPUT_TOKENS,
        min_tokens=10,
        repetition_penalty=1.1,
        structured_outputs=StructuredOutputsParams(json=VERDICT_SCHEMA),
    )

    os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
    unparsed = 0
    total_batches = (len(prepared) - 1) // arguments.batch_size + 1

    for start in range(0, len(prepared), arguments.batch_size):
        batch = prepared[start : start + arguments.batch_size]
        print(f"  batch {start // arguments.batch_size + 1}/{total_batches}")
        outputs = llm.generate([item["prompt"] for item in batch], sampling_params)

        for item, output in zip(batch, outputs):
            response = output.outputs[0].text.strip() if output.outputs else ""
            label, verdict_type, justification = parse_verdict(response)
            if label is None:
                unparsed += 1

            index = item["index"]
            results.at[index, "predicted_label"] = label
            results.at[index, "justification"] = justification
            results.at[index, "predicted_type"] = verdict_type
            results.at[index, "matched_company"] = item["matched"]
            results.at[index, "snippets_used"] = item["snippets"]
            results.at[index, "llm_model"] = arguments.model
            results.at[index, "llm_response"] = response

        results.to_csv(arguments.output, index=False)

    if unparsed:
        print(
            f"  {unparsed} response(s) could not be parsed; "
            f"those rows stay pending for the next run"
        )

    decided = (
        results["predicted_label"].isin(["greenwashing", "not_greenwashing"]).sum()
    )
    print(f"\nWrote {arguments.output}")
    print(
        f"  decided {decided}/{len(results)}, "
        f"abstained {(results['predicted_label'] == 'abstain').sum()}"
    )


if __name__ == "__main__":
    main()
