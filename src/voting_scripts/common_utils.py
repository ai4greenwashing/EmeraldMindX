from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

MODELS: Dict[str, dict] = {
    "prometheus-7b": {
        "name": "prometheus-eval/prometheus-7b-v2.0",
        "max_context": 4096,
        "max_model_len": 4096,
        "chat_template": "mistral",
        "max_new_tokens": 768,
    },
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "max_context": 32000,
        "max_model_len": 32768,
        "chat_template": "default",
        "max_new_tokens": 512,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_context": 32000,
        "max_model_len": 32768,
        "chat_template": "mistral",
        "max_new_tokens": 512,
    },
    "phi4-mini": {
        "name": "microsoft/Phi-4-mini-instruct",
        "max_context": 128000,
        "max_model_len": 131072,
        "chat_template": "phi4",
        "max_new_tokens": 768,
    },
    "gemma3-4b": {
        "name": "google/gemma-3-4b-it",
        "max_context": 128000,
        "max_model_len": 131072,
        "chat_template": "gemma",
        "max_new_tokens": 768,
    },
    "deepseek-llm-7b-chat": {
        "name": "deepseek-ai/deepseek-llm-7b-chat",
        "max_context": 4096,
        "max_model_len": 4096,
        "chat_template": "default",
        "max_new_tokens": 512,
    },
    "granite-4.1-8b": {
        "name": "ibm-granite/granite-4.1-8b",
        "max_context": 128000,
        "max_model_len": 131072,
        "chat_template": "default",
        "max_new_tokens": 512,
    },
    "gemma3-27b": {
        "name": "google/gemma-3-27b-it",
        "max_context": 4096,
        "max_model_len": 4096,
        "chat_template": "gemma",
        "max_new_tokens": 768,
    },
}

# Panel compositions reported in the paper.
PANELS: Dict[int, List[str]] = {
    1: ["prometheus-7b"],
    3: ["prometheus-7b", "qwen2.5-7b", "phi4-mini"],
    5: ["prometheus-7b", "qwen2.5-7b", "phi4-mini", "mistral-7b", "gemma3-4b"],
    7: [
        "prometheus-7b",
        "qwen2.5-7b",
        "phi4-mini",
        "mistral-7b",
        "gemma3-4b",
        "deepseek-llm-7b-chat",
        "granite-4.1-8b",
    ],
}


PIPELINES = ("kgrag", "rag")

# Surface forms shown to the judges inside the prompt. These strings are part of
# the prompt: changing them changes the model inputs.
PIPELINE_DISPLAY = {"kgrag": "GraphRAG", "rag": "RAG"}

LABELS = ("greenwashing", "not_greenwashing")
ABSTAIN = "abstain"

REQUIRED_COLUMNS = ("claim", "gold_label", "predicted_label", "justification")


@dataclass
class Claim:
    """A single claim together with the output of each base-level pipeline."""

    dataset: str
    text: str
    gold_label: str
    outputs: Dict[str, Dict[str, str]]

    def label(self, pipeline: str) -> str:
        return self.outputs[pipeline]["label"]

    def justification(self, pipeline: str) -> str:
        return self.outputs[pipeline]["justification"]


def parse_dataset_spec(spec: str) -> Dict[str, str]:
    """Parse ``NAME:KGRAG_CSV:RAG_CSV``."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected NAME:KGRAG_CSV:RAG_CSV, got {spec!r}"
        )
    name, kgrag_csv, rag_csv = (p.strip() for p in parts)
    if not all((name, kgrag_csv, rag_csv)):
        raise argparse.ArgumentTypeError(f"Empty field in dataset spec {spec!r}")
    return {"name": name, "kgrag": kgrag_csv, "rag": rag_csv}


def _read_pipeline_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    return df


def _as_text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _index_by_claim(df: pd.DataFrame) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for _, row in df.iterrows():
        text = _as_text(row["claim"])
        if not text:
            continue
        index[text.lower()] = {
            "text": text,
            "gold_label": _as_text(row["gold_label"]).lower(),
            "label": _as_text(row["predicted_label"]).lower(),
            "justification": _as_text(row["justification"]),
        }
    return index


def load_claims(
    data_dir: str,
    dataset_specs: Sequence[Dict[str, str]],
    max_claims: int = 0,
) -> List[Claim]:
    """Load claims present in both pipeline files, with agreeing gold labels."""
    import os

    claims: List[Claim] = []
    for spec in dataset_specs:
        indices = {}
        for pipeline in PIPELINES:
            path = os.path.join(data_dir, spec[pipeline])
            indices[pipeline] = _index_by_claim(_read_pipeline_csv(path))

        keys = sorted(set(indices["kgrag"]) & set(indices["rag"]))
        keys = [
            k
            for k in keys
            if indices["kgrag"][k]["gold_label"]
            and indices["kgrag"][k]["gold_label"] == indices["rag"][k]["gold_label"]
        ]
        if max_claims > 0:
            keys = keys[:max_claims]

        for key in keys:
            claims.append(
                Claim(
                    dataset=spec["name"],
                    text=indices["kgrag"][key]["text"],
                    gold_label=indices["kgrag"][key]["gold_label"],
                    outputs={
                        p: {
                            "label": indices[p][key]["label"],
                            "justification": indices[p][key]["justification"],
                        }
                        for p in PIPELINES
                    },
                )
            )

        print(f"  {spec['name']}: {len(keys)} claims")

    return claims


def build_prompt(template: str, claim: Claim) -> str:
    return template.format(
        claim=claim.text,
        rag_label=claim.label("rag"),
        rag_justification=claim.justification("rag"),
        graphrag_label=claim.label("kgrag"),
        graphrag_justification=claim.justification("kgrag"),
    )


def apply_chat_template(prompt: str, judge: str) -> str:
    template = MODELS[judge].get("chat_template", "default")
    if template == "mistral":
        return f"[INST] {prompt} [/INST]"
    if template == "gemma":
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    if template == "phi4":
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    return prompt


def filter_by_context(
    claims: Sequence[Claim],
    prompts: Sequence[str],
    panel: Sequence[str],
) -> tuple:
    """Drop claims whose prompt cannot fit the smallest context in the panel.

    Uses a 4 characters per token approximation, as in the reported runs.
    """
    budget = min(MODELS[j]["max_context"] for j in panel) * 4
    kept_claims, kept_prompts = [], []
    for claim, prompt in zip(claims, prompts):
        if len(prompt) <= budget:
            kept_claims.append(claim)
            kept_prompts.append(prompt)
    dropped = len(claims) - len(kept_claims)
    if dropped:
        print(f"  Dropped {dropped} claim(s) exceeding the panel context budget")
    return kept_claims, kept_prompts


def load_judge(judge: str, gpu_memory: float):
    from vllm import LLM

    config = MODELS[judge]
    print(f"  Loading {judge} ({config['name']})")
    return LLM(
        model=config["name"],
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory,
        max_model_len=config["max_model_len"],
        trust_remote_code=True,
    )


def unload_judge(llm) -> None:

    import contextlib

    for path in (
        ("llm_engine", "model_executor"),
        ("llm_engine", "engine_core"),
        ("engine", "model_executor"),
    ):
        with contextlib.suppress(Exception):
            obj = llm
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            if hasattr(obj, path[-1]):
                delattr(obj, path[-1])

    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()

    del llm
    gc.collect()

    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def report_gpu(tag: str) -> None:
    """Print current GPU allocation, to confirm a judge was torn down."""
    try:
        import torch

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"    [gpu] {tag}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )
    except Exception:
        pass


def generate(
    llm, prompts: Sequence[str], sampling_params, batch_size: int
) -> List[str]:
    """Run generation in batches and return the completion text per prompt."""
    texts: List[str] = []
    total_batches = (len(prompts) - 1) // batch_size + 1
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        print(f"    batch {start // batch_size + 1}/{total_batches}")
        for output in llm.generate(batch, sampling_params):
            texts.append(output.outputs[0].text)
    return texts


def extract_json(text: str) -> Optional[dict]:
    """Parse the first JSON object in a completion, tolerating code fences."""
    import re

    cleaned = re.sub(r"```(?:json)?", "", text or "")
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def compute_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict:
    """Accuracy, coverage and overall accuracy as defined in the paper.

    Accuracy is measured over decided claims, coverage is the share of decided
    claims, and overall accuracy is their product (correct decisions / total).
    """
    total = len(gold)
    decided = [(g, p) for g, p in zip(gold, predicted) if p in LABELS]
    n_decided = len(decided)
    correct = sum(1 for g, p in decided if g == p)

    tp = sum(1 for g, p in decided if g == "greenwashing" and p == "greenwashing")
    fp = sum(1 for g, p in decided if g == "not_greenwashing" and p == "greenwashing")
    fn = sum(1 for g, p in decided if g == "greenwashing" and p == "not_greenwashing")
    tn = sum(
        1 for g, p in decided if g == "not_greenwashing" and p == "not_greenwashing"
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total_claims": total,
        "abstains": total - n_decided,
        "accuracy": round(correct / n_decided * 100, 2) if n_decided else 0.0,
        "coverage": round(n_decided / total * 100, 2) if total else 0.0,
        "overall_accuracy": round(correct / total * 100, 2) if total else 0.0,
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize(predictions: pd.DataFrame, extra: Dict[str, object]) -> pd.DataFrame:
    """Per-dataset metrics plus an ALL row."""
    rows = []
    for dataset in list(predictions["dataset"].unique()) + ["ALL"]:
        subset = (
            predictions
            if dataset == "ALL"
            else predictions[predictions["dataset"] == dataset]
        )
        row = {"dataset": dataset, **extra}
        row.update(compute_metrics(subset["gold_label"], subset["final_label"]))
        counts = subset["winner"].value_counts()
        row["wins_kgrag"] = int(counts.get("kgrag", 0))
        row["wins_rag"] = int(counts.get("rag", 0))
        row["undecided"] = int(counts.get(ABSTAIN, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    header = (
        f"  {'Dataset':<24s}{'Acc.':>8s}{'Cov.':>8s}{'Overall':>9s}"
        f"{'Abst.':>7s}{'KGRAG':>7s}{'RAG':>6s}{'Undec.':>8s}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for _, r in summary.iterrows():
        print(
            f"  {r['dataset']:<24s}{r['accuracy']:>7.2f}%{r['coverage']:>7.2f}%"
            f"{r['overall_accuracy']:>8.2f}%{r['abstains']:>7d}"
            f"{r['wins_kgrag']:>7d}{r['wins_rag']:>6d}{r['undecided']:>8d}"
        )


def base_arg_parser(
    description: str,
    default_data_dir: str,
    default_output_dir: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory holding the pipeline output CSV files.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        type=parse_dataset_spec,
        metavar="NAME:KGRAG_CSV:RAG_CSV",
        help="Dataset to evaluate. Repeat for several datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory for judgments, predictions and metrics.",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        choices=sorted(PANELS),
        help="Use one of the predefined judge panels.",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        choices=sorted(MODELS),
        help="Explicit list of judges, overrides --panel-size.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--max-claims",
        type=int,
        default=0,
        help="Limit the claims per dataset (0 = all).",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.7,
        help="Fraction of GPU memory per judge.",
    )
    parser.add_argument(
        "--reuse-judgments",
        action="store_true",
        help="Skip inference and aggregate an existing judgments.csv.",
    )
    return parser


def resolve_panel(args: argparse.Namespace) -> List[str]:
    if args.judges:
        return list(args.judges)
    if args.panel_size:
        return list(PANELS[args.panel_size])
    return list(PANELS[5])
