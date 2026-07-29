from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd

# Directory holding the per-pipeline CSVs, each with the columns
# claim, gold_label, predicted_label, justification.
DATA_DIR = "path/to/pipeline_outputs"

# Directory for <pipeline>/<dataset>.csv score files.
OUTPUT_DIR = "path/to/ilora"

DEFAULT_MODEL = "prometheus-eval/prometheus-13b-v1.0"

CRITERIA = ("Informativeness", "Logicality", "Objectivity", "Readability", "Accuracy")
SCORE_COLUMNS = (*CRITERIA, "OverallScore")

# Pipelines scored directly, mapped to the field of the dataset spec they read.
BASE_PIPELINES = {"baseline": "baseline", "emx-rag": "rag", "emx-kgrag": "kgrag"}

# Which scored pipeline a winner in predictions.csv corresponds to.
WINNER_TO_PIPELINE = {"rag": "emx-rag", "kgrag": "emx-kgrag"}


ILORA_PROMPT = """You are an expert evaluator of LLM-generated explanations.

Evaluate the QUALITY of the explanation according to the ILORA Evaluation Framework.
For each criterion, give a score from 1 to 5 (1 = lowest quality, 5 = highest quality).

CRITERIA:

1. Informativeness (I) - Does the explanation provide new information, such as background knowledge or additional context that helps understand the decision?
    Score 1: The explanation provides no new information beyond restating the claim.
    Score 2: The explanation provides minimal context, mostly redundant with the claim.
    Score 3: The explanation provides some useful context but remains shallow.
    Score 4: The explanation provides clear additional information that aids understanding.
    Score 5: The explanation provides rich, substantive context with concrete supporting evidence.

2. Logicality (L) - Does the explanation follow a reasonable thought process? Is there a strong causal relationship between the explanation and the result?
    Score 1: No logical connection between explanation and verdict; contradictory.
    Score 2: Weak, hand-wavy reasoning with major gaps.
    Score 3: Plausible reasoning but with noticeable gaps.
    Score 4: Sound reasoning with only minor weaknesses.
    Score 5: Rigorous causal chain from evidence to conclusion.

3. Objectivity (O) - Is the explanation objective and free from excessive subjective emotion or bias?
    Score 1: Heavily biased or emotionally loaded.
    Score 2: Noticeable subjective framing throughout.
    Score 3: Mostly neutral with occasional slanted language.
    Score 4: Objective tone with only minor lapses.
    Score 5: Fully neutral and evidence-based.

4. Readability (R) - Does the explanation follow proper grammatical and structural rules? Are the sentences coherent and easy to understand?
    Score 1: Broken or very hard to parse.
    Score 2: Frequent grammatical errors and awkward phrasing.
    Score 3: Readable but uneven in clarity.
    Score 4: Clear and well-structured with minor issues.
    Score 5: Polished, fluent, and easy to follow.

5. Accuracy (A) - Does the generated explanation align with the actual truth label? Does the explanation accurately reflect the result?
    Score 1: Explanation contradicts the truth label.
    Score 2: Mostly inconsistent with the truth label.
    Score 3: Partially aligned, with mixed signals.
    Score 4: Aligned with only minor inaccuracies.
    Score 5: Fully consistent with the truth label.

CONTEXT:
Claim: {claim}
Prediction: {prediction}
Justification: {justification}

Respond ONLY with a JSON object in this exact schema:
{{"informativeness": <int 1-5>, "logicality": <int 1-5>, "objectivity": <int 1-5>, "readability": <int 1-5>, "accuracy": <int 1-5>}}"""

ILORA_SCHEMA = {
    "type": "object",
    "properties": {
        "informativeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "logicality": {"type": "integer", "minimum": 1, "maximum": 5},
        "objectivity": {"type": "integer", "minimum": 1, "maximum": 5},
        "readability": {"type": "integer", "minimum": 1, "maximum": 5},
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": [
        "informativeness",
        "logicality",
        "objectivity",
        "readability",
        "accuracy",
    ],
    "additionalProperties": False,
}


def normalize(claim) -> str:

    return str(claim).strip().lower()


def parse_dataset_spec(spec: str) -> Dict[str, str]:

    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 4 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"Expected NAME:BASELINE_CSV:KGRAG_CSV:RAG_CSV, got {spec!r}"
        )
    return dict(zip(("name", "baseline", "kgrag", "rag"), parts))


def parse_selection_spec(spec: str) -> Dict[str, str]:

    label, _, path = spec.partition(":")
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            f"Expected LABEL:PREDICTIONS_CSV, got {spec!r}"
        )
    return {"label": label.strip(), "path": path.strip()}


def read_pipeline_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for column in ("claim", "predicted_label", "justification"):
        if column not in df.columns:
            raise ValueError(f"{path}: missing column {column!r}")
    df["claim_key"] = df["claim"].map(normalize)
    return df


def score_path(output_dir: str, pipeline: str, dataset: str) -> str:
    return os.path.join(output_dir, pipeline, f"{dataset}.csv")


def already_scored(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path)["claim"].dropna().map(normalize))
    except Exception:
        return set()


def append_rows(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        pd.DataFrame(rows).to_csv(handle, header=handle.tell() == 0, index=False)


def parse_scores(completion: str) -> Optional[Dict[str, float]]:
    try:
        payload = json.loads(completion)
    except (json.JSONDecodeError, TypeError):
        return None
    try:
        scores = {name: float(payload[name.lower()]) for name in CRITERIA}
    except (KeyError, TypeError, ValueError):
        return None
    scores["OverallScore"] = sum(scores[name] for name in CRITERIA) / len(CRITERIA)
    return scores


class Evaluator:

    def __init__(self, model: str, gpu_memory: float, max_model_len: int):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.sampling_params import SamplingParams, StructuredOutputsParams

        print(f"Loading evaluator: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(
            model=model,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(json=ILORA_SCHEMA),
        )

    def format(self, prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Models without a chat template receive the prompt unchanged.
            return prompt

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(
            [self.format(p) for p in prompts], self.sampling_params
        )
        return [o.outputs[0].text.strip() for o in outputs]


def score_pipeline(
    evaluator: Evaluator,
    frame: pd.DataFrame,
    pipeline: str,
    dataset: str,
    output_dir: str,
    batch_size: int,
    error_log: str,
) -> None:
    path = score_path(output_dir, pipeline, dataset)
    done = already_scored(path)

    pending = frame[
        ~frame["claim_key"].isin(done)
        & frame["justification"].notna()
        & frame["predicted_label"].notna()
    ]
    print(
        f"\n  {pipeline} / {dataset}: {len(pending)} to score, {len(done)} already done"
    )
    if pending.empty:
        return

    unparsed = 0
    records = pending.to_dict("records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            ILORA_PROMPT.format(
                claim=record["claim"],
                prediction=record["predicted_label"],
                justification=record["justification"],
            )
            for record in batch
        ]
        completions = evaluator.generate(prompts)

        rows = []
        for record, prompt, completion in zip(batch, prompts, completions):
            scores = parse_scores(completion)
            if scores is None:
                unparsed += 1
                log_failure(error_log, record["claim"], prompt, completion)
                continue
            rows.append(
                {
                    "claim": record["claim"],
                    "predicted_label": record["predicted_label"],
                    "justification": record["justification"],
                    **scores,
                }
            )

        if rows:
            append_rows(path, rows)
        print(f"    scored {min(start + batch_size, len(records))}/{len(records)}")

    if unparsed:
        print(f"    {unparsed} response(s) could not be parsed, logged to {error_log}")


def log_failure(error_log: str, claim: str, prompt: str, completion: str) -> None:
    os.makedirs(os.path.dirname(error_log), exist_ok=True)
    with open(error_log, "a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 70}\nCLAIM: {claim}\n{'=' * 70}\n")
        handle.write(f"PROMPT:\n{prompt}\n{'-' * 70}\n")
        handle.write(f"OUTPUT:\n{completion}\n")


def derive_pipeline(
    label: str,
    predictions: pd.DataFrame,
    dataset: str,
    output_dir: str,
) -> None:

    scored = {}
    for winner, pipeline in WINNER_TO_PIPELINE.items():
        path = score_path(output_dir, pipeline, dataset)
        if not os.path.exists(path):
            print(f"\n  {label} / {dataset}: {pipeline} has no scores yet, skipping")
            return
        frame = pd.read_csv(path)
        frame["claim_key"] = frame["claim"].map(normalize)
        scored[winner] = frame.drop_duplicates("claim_key").set_index("claim_key")

    rows, undecided, missing = [], 0, 0
    for _, prediction in predictions.iterrows():
        winner = str(prediction.get("winner", "")).strip().lower()
        if winner not in scored:
            undecided += 1
            continue
        key = normalize(prediction["claim"])
        if key not in scored[winner].index:
            missing += 1
            continue
        source = scored[winner].loc[key]
        rows.append(
            {
                "claim": source["claim"],
                "predicted_label": source["predicted_label"],
                "justification": source["justification"],
                "winner": winner,
                **{name: float(source[name]) for name in SCORE_COLUMNS},
            }
        )

    path = score_path(output_dir, label, dataset)
    if os.path.exists(path):
        os.remove(path)
    if rows:
        append_rows(path, rows)

    print(
        f"\n  {label} / {dataset}: {len(rows)} derived, "
        f"{undecided} undecided, {missing} unscored"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pipeline justifications under the ILORA rubric."
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        type=parse_dataset_spec,
        metavar="NAME:BASELINE_CSV:KGRAG_CSV:RAG_CSV",
    )
    parser.add_argument(
        "--selection",
        dest="selections",
        action="append",
        default=[],
        type=parse_selection_spec,
        metavar="LABEL:PREDICTIONS_CSV",
        help="A voting variant to derive, e.g. "
        "emx-voting:results/voting_5/predictions.csv",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory", type=float, default=0.8)
    parser.add_argument(
        "--derive-only",
        action="store_true",
        help="Skip scoring and only rebuild the derived pipelines.",
    )
    args = parser.parse_args()

    error_log = os.path.join(args.output_dir, "logs", "parse_failures.log")

    datasets = [spec["name"] for spec in args.datasets]

    # The pipeline CSVs are only needed for scoring; deriving reads the score
    # files written by an earlier run.
    if not args.derive_only:
        frames = {}
        for spec in args.datasets:
            frames[spec["name"]] = {
                pipeline: read_pipeline_csv(os.path.join(args.data_dir, spec[field]))
                for pipeline, field in BASE_PIPELINES.items()
            }
            print(
                f"{spec['name']}: "
                + ", ".join(f"{p}={len(f)}" for p, f in frames[spec["name"]].items())
            )

        evaluator = Evaluator(args.model, args.gpu_memory, args.max_model_len)
        for dataset, by_pipeline in frames.items():
            for pipeline, frame in by_pipeline.items():
                score_pipeline(
                    evaluator,
                    frame,
                    pipeline,
                    dataset,
                    args.output_dir,
                    args.batch_size,
                    error_log,
                )

    for selection in args.selections:
        predictions = pd.read_csv(selection["path"])
        for dataset in datasets:
            subset = (
                predictions[predictions["dataset"] == dataset]
                if "dataset" in predictions.columns
                else predictions
            )
            if subset.empty:
                print(
                    f"\n  {selection['label']} / {dataset}: no rows in "
                    f"{selection['path']}"
                )
                continue
            derive_pipeline(selection["label"], subset, dataset, args.output_dir)

    print(f"\nScores in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
