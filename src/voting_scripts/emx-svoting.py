"""EMX-SVOTING: confidence-weighted vote over a panel of LLM judges.

Each judge reports a confidence for both EMX-KGRAG and EMX-RAG. Confidences are
Each judge's pair is renormalized to sum to one, the pairs are summed per
pipeline across the panel, and the higher cumulative score wins. Scores within
TIE_TOLERANCE of each other, or a claim on which no judge produced a usable
pair, yield an abstention.

Example
-------
    python emx_svoting.py \
        --data-dir path/to/pipeline_outputs \
        --dataset emerald_data:emerald_data_kgrag.csv:emerald_data_rag.csv \
        --panel-size 5 \
        --output-dir path/to/results/svoting_5
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

import emx_common as common

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Directory containing one CSV per pipeline per dataset. Each CSV is read for
# the columns: claim, gold_label, predicted_label, justification — where
# predicted_label and gold_label are greenwashing / not_greenwashing / abstain,
# and justification is the free-text rationale the judges compare. Claims are
# matched across the two files by claim text.
DATA_DIR = "path/to/pipeline_outputs"

# Directory for judgments.csv, predictions.csv and metrics.csv.
OUTPUT_DIR = "path/to/results"

CONFIDENCE_FLOOR = 1e-4
TIE_TOLERANCE = 1e-6

CONFIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "conf_rag": {"type": "number", "minimum": CONFIDENCE_FLOOR, "maximum": 1.0},
        "conf_graphrag": {
            "type": "number",
            "minimum": CONFIDENCE_FLOOR,
            "maximum": 1.0,
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
    },
    "required": ["conf_rag", "conf_graphrag", "reason"],
    "additionalProperties": False,
}


ILORA_CONFIDENCE_PROMPT = """You are an expert evaluator of environmental sustainability claims. Your task is to determine which analysis pipeline produced the highest-quality justification for classifying an ESG claim, according to the **ILORA Evaluation Framework**.

### Claim
{claim}

### Pipeline Outputs

**RAG (Retrieval-Augmented Generation)**
- Label: {rag_label}
- Justification: {rag_justification}

**GraphRAG (Graph-based Retrieval-Augmented Generation)**
- Label: {graphrag_label}
- Justification: {graphrag_justification}

### ILORA Evaluation Framework
Evaluate the QUALITY of each justification along the following five criteria. For each criterion, mentally assign a score from 1 to 5 (1 = lowest quality, 5 = highest quality):

1. **Informativeness (I)** - Does the explanation provide new information, such as background knowledge or additional context that helps understand the decision?
2. **Logicality (L)** - Does the explanation follow a reasonable thought process? Is there a strong causal relationship between the explanation and the result?
3. **Objectivity (O)** - Is the explanation objective and free from excessive subjective emotion or bias?
4. **Readability (R)** - Does the explanation follow proper grammatical and structural rules? Are the sentences coherent and easy to understand?
5. **Accuracy (A)** - Does the generated explanation align with the actual label? Does the explanation accurately reflect the result?

Your confidence in each pipeline should reflect its aggregate quality across all five ILORA criteria - the higher the overall ILORA quality, the higher the confidence assigned to that pipeline.

### STRICT OUTPUT RULES
- Output ONLY a single JSON object. No preamble, no explanation, no markdown, no code fences.
- The JSON must have exactly three keys: "conf_rag", "conf_graphrag", and "reason".
- "conf_rag" and "conf_graphrag" must each be a number between 0 and 1 (inclusive), representing your confidence that each pipeline produced the best justification under the ILORA framework.
- The two confidence values MUST sum to exactly 1.0.
- A value near 1.0 for one pipeline means you are highly confident it is the best under ILORA; values spread across pipelines indicate uncertainty between them.
- "reason" must be no more than 50 words and should briefly reference the ILORA criteria that drove the confidence distribution.
- Any output that is not a bare JSON object with valid confidences summing to 1.0 will be considered invalid.
- No ties allowed - you must assign a higher confidence to one pipeline, even if the difference is small. You cannot assign 0 confidence to all pipelines; at least one must have a confidence greater than 0.

### Response:
{{"conf_rag": <0.0-1.0>, "conf_graphrag": <0.0-1.0>, "reason": "<max 50 words>"}}"""


def parse_confidences(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (conf_rag, conf_kgrag), or (None, None) if unusable."""
    payload = common.extract_json(text)
    if not payload:
        return None, None

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    conf_rag = as_float(payload.get("conf_rag"))
    conf_kgrag = as_float(payload.get("conf_graphrag"))

    if conf_rag is None or conf_kgrag is None:
        return None, None
    if conf_rag < 0 or conf_kgrag < 0 or (conf_rag + conf_kgrag) <= 0:
        return None, None
    return conf_rag, conf_kgrag


def collect_judgments(
    claims: List[common.Claim],
    panel: List[str],
    batch_size: int,
    gpu_memory: float,
) -> pd.DataFrame:
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    prompts = [common.build_prompt(ILORA_CONFIDENCE_PROMPT, c) for c in claims]
    claims, prompts = common.filter_by_context(claims, prompts, panel)

    rows = []
    for judge in panel:
        print(f"\n  Judge: {judge}")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=common.MODELS[judge].get("max_new_tokens", 512),
            structured_outputs=StructuredOutputsParams(json=CONFIDENCE_SCHEMA),
        )
        llm = common.load_judge(judge, gpu_memory)
        formatted = [common.apply_chat_template(p, judge) for p in prompts]
        completions = common.generate(llm, formatted, sampling_params, batch_size)
        common.unload_judge(llm)
        common.report_gpu(f"after unloading {judge}")

        parsed = 0
        for claim, completion in zip(claims, completions):
            conf_rag, conf_kgrag = parse_confidences(completion)
            parsed += conf_rag is not None
            rows.append(
                {
                    "dataset": claim.dataset,
                    "claim": claim.text,
                    "gold_label": claim.gold_label,
                    "kgrag_label": claim.label("kgrag"),
                    "rag_label": claim.label("rag"),
                    "judge": judge,
                    "conf_rag": conf_rag,
                    "conf_kgrag": conf_kgrag,
                    "raw_response": (completion or "")[:500],
                }
            )
        print(f"    confidences parsed: {parsed}/{len(claims)}")

    return pd.DataFrame(rows)


def aggregate(judgments: pd.DataFrame, panel: List[str]) -> pd.DataFrame:
    """Sum per-pipeline confidences across judges and take the argmax."""
    rows = []
    for (dataset, claim), group in judgments.groupby(["dataset", "claim"], sort=False):
        first = group.iloc[0]
        labels = {"kgrag": first["kgrag_label"], "rag": first["rag_label"]}

        scores = {"kgrag": 0.0, "rag": 0.0}
        contributing = 0
        per_judge: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for _, judgment in group.iterrows():
            conf_rag = judgment["conf_rag"]
            conf_kgrag = judgment["conf_kgrag"]
            if pd.isna(conf_rag) or pd.isna(conf_kgrag):
                per_judge[judgment["judge"]] = (None, None)
                continue

            # Renormalize so each judge contributes the same total weight.
            # The prompt already asks for a pair summing to 1.0, so this is a
            # no-op for compliant responses and a guard against the rest.
            conf_rag, conf_kgrag = float(conf_rag), float(conf_kgrag)
            total = conf_rag + conf_kgrag
            conf_rag, conf_kgrag = conf_rag / total, conf_kgrag / total

            scores["rag"] += conf_rag
            scores["kgrag"] += conf_kgrag
            contributing += 1
            per_judge[judgment["judge"]] = (conf_rag, conf_kgrag)

        if contributing == 0 or abs(scores["kgrag"] - scores["rag"]) <= TIE_TOLERANCE:
            winner, final_label = common.ABSTAIN, common.ABSTAIN
        else:
            winner = max(scores, key=scores.get)
            final_label = labels[winner]

        row = {
            "dataset": dataset,
            "claim": claim,
            "gold_label": first["gold_label"],
            "kgrag_label": labels["kgrag"],
            "rag_label": labels["rag"],
            "score_kgrag": round(scores["kgrag"], 6),
            "score_rag": round(scores["rag"], 6),
            "judges_contributing": contributing,
            "winner": winner,
            "final_label": final_label,
        }
        for judge in panel:
            conf_rag, conf_kgrag = per_judge.get(judge, (None, None))
            row[f"conf_rag_{judge}"] = conf_rag
            row[f"conf_kgrag_{judge}"] = conf_kgrag
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = common.base_arg_parser(
        "EMX-SVOTING: confidence-weighted vote over LLM judges.",
        default_data_dir=DATA_DIR,
        default_output_dir=OUTPUT_DIR,
    )
    args = parser.parse_args()

    panel = common.resolve_panel(args)
    os.makedirs(args.output_dir, exist_ok=True)
    judgments_path = os.path.join(args.output_dir, "judgments.csv")

    print(f"EMX-SVOTING | panel: {', '.join(panel)}")

    if args.reuse_judgments:
        judgments = pd.read_csv(judgments_path)
        print(f"  Reusing {len(judgments)} judgments from {judgments_path}")
    else:
        claims = common.load_claims(args.data_dir, args.datasets, args.max_claims)
        if not claims:
            print("No claims to process.")
            return
        judgments = collect_judgments(claims, panel, args.batch_size, args.gpu_memory)
        judgments.to_csv(judgments_path, index=False)
        print(f"\n  Wrote {judgments_path}")

    predictions = aggregate(judgments, panel)
    predictions.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    summary = common.summarize(
        predictions, {"method": "emx-svoting", "num_judges": len(panel)}
    )
    summary.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    common.print_summary(summary)
    print(f"\nOutputs in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
