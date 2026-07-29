"""EMX-VOTING: plurality vote over a panel of LLM judges.

Each judge inspects the (label, justification) pair produced by EMX-KGRAG and
EMX-RAG for a claim and votes for one of them. The pipeline with the most votes
supplies the final label; an unresolved vote yields an abstention.

Example
-------
    python emx_voting.py \
        --data-dir path/to/pipeline_outputs \
        --dataset emerald_data:emerald_data_kgrag.csv:emerald_data_rag.csv \
        --panel-size 5 \
        --output-dir path/to/results/voting_5
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional

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

MAX_NEW_TOKENS = 300

ILORA_VOTE_PROMPT = """Evaluate the QUALITY of the explanations below, according to the ILORA Evaluation Framework.
For each criterion, give a score from 1 to 5 (1 = lowest quality, 5 = highest quality).

CRITERIA:
1. Informativeness (I) - Does the explanation provide new information, such as background knowledge or additional context that helps understand the decision?
2. Logicality (L) - Does the explanation follow a reasonable thought process? Is there a strong causal relationship between the explanation and the result?
3. Objectivity (O) - Is the explanation objective and free from excessive subjective emotion or bias?
4. Readability (R) - Does the explanation follow proper grammatical and structural rules? Are the sentences coherent and easy to understand?
5. Accuracy (A) - Does the generated explanation align with the actual label? Does the explanation accurately reflect the result?

### Input
**Claim:** {claim}

**GraphRAG**
- Label: {graphrag_label}
- Justification: {graphrag_justification}

**RAG**
- Label: {rag_label}
- Justification: {rag_justification}

### Task
Pick the single best pipeline based on the ILORA CRITERIA.

### Response Format (JSON only, no other text):
```json
{{"winner": "<GraphRAG|RAG>", "reason": "<one sentence explanation>"}}
```"""


def parse_vote(text: str) -> tuple:
    """Return (pipeline, reason). The pipeline is None when unparseable."""
    payload = common.extract_json(text)
    if payload:
        winner = _normalize(str(payload.get("winner", "")))
        if winner:
            reason = str(payload.get("reason", ""))[:500]
            return winner, reason

    winner = _normalize(text or "")
    return winner, "fallback_parse" if winner else ""


def _normalize(value: str) -> Optional[str]:
    value = value.lower()
    # "rag" is a substring of "graphrag", so the graph variant is tested first.
    if "kgrag" in value or "graphrag" in value or "graph rag" in value:
        return "kgrag"
    if "rag" in value:
        return "rag"
    return None


def collect_judgments(
    claims: List[common.Claim],
    panel: List[str],
    batch_size: int,
    gpu_memory: float,
) -> pd.DataFrame:
    from vllm import SamplingParams

    prompts = [common.build_prompt(ILORA_VOTE_PROMPT, claim) for claim in claims]
    claims, prompts = common.filter_by_context(claims, prompts, panel)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    rows = []
    for judge in panel:
        print(f"\n  Judge: {judge}")
        llm = common.load_judge(judge, gpu_memory)
        formatted = [common.apply_chat_template(p, judge) for p in prompts]
        completions = common.generate(llm, formatted, sampling_params, batch_size)
        common.unload_judge(llm)
        common.report_gpu(f"after unloading {judge}")

        parsed = 0
        for claim, completion in zip(claims, completions):
            vote, reason = parse_vote(completion)
            parsed += vote is not None
            rows.append(
                {
                    "dataset": claim.dataset,
                    "claim": claim.text,
                    "gold_label": claim.gold_label,
                    "kgrag_label": claim.label("kgrag"),
                    "rag_label": claim.label("rag"),
                    "judge": judge,
                    "vote": vote or "",
                    "reason": reason,
                    "raw_response": (completion or "")[:500],
                }
            )
        print(f"    votes parsed: {parsed}/{len(claims)}")

    return pd.DataFrame(rows)


def aggregate(judgments: pd.DataFrame, panel: List[str]) -> pd.DataFrame:
    """Plurality vote per claim. Ties and empty tallies become abstentions."""
    rows = []
    for (dataset, claim), group in judgments.groupby(["dataset", "claim"], sort=False):
        first = group.iloc[0]
        labels = {"kgrag": first["kgrag_label"], "rag": first["rag_label"]}

        votes = [v for v in group["vote"] if v in common.PIPELINES]
        counts = Counter(votes)
        leaders = (
            [p for p, c in counts.items() if c == max(counts.values())]
            if counts
            else []
        )

        if len(leaders) == 1:
            winner = leaders[0]
            final_label = labels[winner]
        else:
            winner = common.ABSTAIN
            final_label = common.ABSTAIN

        row = {
            "dataset": dataset,
            "claim": claim,
            "gold_label": first["gold_label"],
            "kgrag_label": labels["kgrag"],
            "rag_label": labels["rag"],
            "votes_kgrag": counts.get("kgrag", 0),
            "votes_rag": counts.get("rag", 0),
            "votes_invalid": len(group) - len(votes),
            "winner": winner,
            "final_label": final_label,
        }
        by_judge: Dict[str, str] = dict(zip(group["judge"], group["vote"]))
        for judge in panel:
            row[f"vote_{judge}"] = by_judge.get(judge, "")
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = common.base_arg_parser(
        "EMX-VOTING: plurality vote over LLM judges.",
        default_data_dir=DATA_DIR,
        default_output_dir=OUTPUT_DIR,
    )
    args = parser.parse_args()

    panel = common.resolve_panel(args)
    os.makedirs(args.output_dir, exist_ok=True)
    judgments_path = os.path.join(args.output_dir, "judgments.csv")

    print(f"EMX-VOTING | panel: {', '.join(panel)}")

    if args.reuse_judgments:
        judgments = pd.read_csv(judgments_path).fillna({"vote": ""})
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
        predictions, {"method": "emx-voting", "num_judges": len(panel)}
    )
    summary.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    common.print_summary(summary)
    print(f"\nOutputs in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
