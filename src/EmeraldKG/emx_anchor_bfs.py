#!/usr/bin/env python3
"""
EMX-Anchor-BFS: greenwashing classification over EmeraldGraphX with the EmeraldMind
graph retrieval and anchor entry.

The comparator row EMX-Anchor-BFS of the EmeraldMindX paper. It keeps the retrieval of
EM-KGRAG (``7-classify.py``) and changes only the entry point, so that it isolates the
effect of the guided beam search (``emx_kgrag.py``) from that of the new graph:

  Entry       Resolve the company via (Anchor)-[:REPRESENTS]->(Organization), keeping
              every linked Organization; fall back to the four-tier ``CompanyMatcher``.
  Retrieval   From each Organization, ``retrieve_evidence`` (one hop) or
              ``retrieve_evidence_with_paths`` (multi-hop BFS) matches the claim's
              extracted nodes against connected nodes of the same class and keeps the
              top-k per class above the similarity threshold.
  Merge       Deduplicate across Organizations (highest similarity wins), keep at most
              ``per_class_cap`` nodes per class and ``hard_ceiling`` nodes overall.

The evidence is passed to ``classify_few`` (Gemma 3 27B), as in EMX-KGRAG, and the
verdicts are written in the CSV format read by the voting scripts.

Retrieval matches the claim nodes produced by ``6a-parse_claims_to_nodes.py`` and
embedded by ``6b-generate_embeddings.py``: claim i of the input CSV is read from
``<embeddings-dir>/claim_<i>.json``.

Usage
-----
    python src/EmeraldKG/emx_anchor_bfs.py --claims datasets/green_claims.csv \
        --embeddings-dir claims_green_claims --output results/green_claims_anchor_bfs.csv
    python src/EmeraldKG/emx_anchor_bfs.py --claims datasets/green_claims.csv \
        --embeddings-dir claims_green_claims --hops multi
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.emx_classification_utils import (
    classify_few,
    driver,
    retrieve_evidence,
    retrieve_evidence_with_paths,
)
from emx_kgrag import (
    ABSTAIN_LABEL,
    OUTPUT_COLUMNS,
    VALID_LABELS,
    _abstain,
    _company_matcher,
    compute_metrics,
    get_anchor_org_ids,
)

logger = getLogger(__name__)


@dataclass(frozen=True)
class BFSConfig:
    """Retrieval settings; the defaults are those of EM-KGRAG (``7-classify.py``)."""

    hops: str = "one"                  # "one" or "multi"
    top_k_per_class: int = 3
    similarity_threshold: float = 0.2
    per_class_cap: int = 3             # nodes kept per class after the cross-org merge
    hard_ceiling: Optional[int] = 40   # nodes kept overall (None = no cap)
    embeddings_dir: str = "claims"


# ---------------------------------------------------------------------------
# Entry point and retrieval
# ---------------------------------------------------------------------------


def resolve_org_ids(company_name: str) -> list[int]:
    """Every Organization an Anchor represents, else the CompanyMatcher match."""
    org_ids = get_anchor_org_ids(company_name)
    if org_ids:
        logger.info("Anchor resolved '%s' to %d Organization node(s).", company_name, len(org_ids))
        return org_ids
    company_id = _company_matcher().get_company_id(company_name)
    if company_id is None:
        logger.warning("No match for company '%s'.", company_name)
        return []
    return [company_id]


def merge_evidence(
    all_evidence: list[list[dict]],
    per_class_cap: int = 3,
    hard_ceiling: Optional[int] = 40,
) -> list[dict]:
    """Merge per-Organization evidence: one entry per node, capped per class and overall."""
    best_by_node: dict[int, dict] = {}
    for evidence_list in all_evidence:
        for item in evidence_list:
            nid = item.get("node_id")
            if nid is None:
                continue
            existing = best_by_node.get(nid)
            if existing is None or item.get("similarity", 0) > existing.get("similarity", 0):
                best_by_node[nid] = item

    buckets: dict[str, list[dict]] = {}
    for item in best_by_node.values():
        labels = item.get("labels") or []
        buckets.setdefault(labels[0] if labels else "Unknown", []).append(item)

    capped: list[dict] = []
    for items in buckets.values():
        items.sort(key=lambda x: (-x.get("similarity", 0), x.get("node_id", 0)))
        capped.extend(items[:per_class_cap])

    # Final sort for determinism
    capped.sort(
        key=lambda x: (
            -x.get("similarity", 0),
            x.get("node_id", 0),
            x["labels"][0] if x.get("labels") else "",
        )
    )
    if hard_ceiling and len(capped) > hard_ceiling:
        logger.info("Hard ceiling %d applied (had %d).", hard_ceiling, len(capped))
        capped = capped[:hard_ceiling]

    logger.info("Merged evidence: %d nodes across %d classes.", len(capped), len(buckets))
    return capped


def retrieve_evidence_anchor_bfs(
    company_name: str,
    claim: str,
    claim_idx: int,
    config: BFSConfig = BFSConfig(),
) -> tuple[list[dict], list[int]]:
    """Retrieve the evidence for a claim. Returns (evidence, org_ids)."""
    org_ids = resolve_org_ids(company_name)
    retrieve_fn = retrieve_evidence if config.hops == "one" else retrieve_evidence_with_paths

    all_evidence: list[list[dict]] = []
    for org_id in org_ids:
        try:
            evidence = retrieve_fn(
                org_id,
                claim,
                claim_idx=claim_idx,
                top_k_per_class=config.top_k_per_class,
                similarity_threshold=config.similarity_threshold,
                embeddings_dir=config.embeddings_dir,
            )
        except Exception as e:
            logger.warning("Retrieval error for claim %d, Organization %d: %s", claim_idx, org_id, e)
            continue
        if evidence:
            all_evidence.append(evidence)

    if not all_evidence:
        return [], org_ids
    return merge_evidence(all_evidence, config.per_class_cap, config.hard_ceiling), org_ids

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_claim(company: str, claim: str, claim_idx: int, config: BFSConfig) -> Dict[str, Any]:
    """Retrieve evidence for one claim and classify it; returns the verdict fields."""
    evidence, org_ids = retrieve_evidence_anchor_bfs(company, claim, claim_idx, config)

    if not org_ids:
        return _abstain("(No company ID found)", "Company not found in knowledge graph.")
    if not evidence:
        return _abstain("(No context found)", "No relevant context found in KG.")

    similarities = json.dumps({str(e["node_id"]): e["similarity"] for e in evidence})
    try:
        resp, context_str, _, _ = classify_few(evidence, company, claim)
    except Exception as e:
        logger.error("Classification error for '%s': %s", claim, e)
        return _abstain(None, f"LLM call failed: {e}", similarities)

    raw_label = str(resp.get("label", "")).lower().strip()
    return {
        "predicted_label": raw_label if raw_label in VALID_LABELS else ABSTAIN_LABEL,
        "context": context_str,
        "justification": resp.get("reasoning", ""),
        "similarities": similarities,
        "cited_node_ids": resp.get("cited_node_ids", []),
    }


def run_pipeline(df: pd.DataFrame, config: BFSConfig) -> tuple[pd.DataFrame, list[float]]:
    """Classify every claim in ``df``; returns the output rows and per-claim seconds.

    The row index of ``df`` selects the claim's embeddings file.
    """
    rows: list[dict] = []
    durations: list[float] = []

    for claim_idx, row in tqdm(df.iterrows(), total=len(df)):
        company = str(row.get("company", "")).strip() if pd.notna(row.get("company")) else ""
        claim = str(row.get("claim", "")).strip() if pd.notna(row.get("claim")) else ""
        gold = row.get("gold_label")
        gold_label = str(gold).strip().lower() if pd.notna(gold) else None

        # Failed rows are kept as abstains so they count against coverage.
        start = time.perf_counter()
        if not claim or not company:
            logger.warning("Row %s: missing claim or company.", claim_idx)
            verdict = _abstain("(Skipped)", "Missing claim or company.")
        else:
            try:
                verdict = classify_claim(company, claim, int(claim_idx), config)
            except Exception as e:
                logger.error("Pipeline error for row %s: %s", claim_idx, e)
                verdict = _abstain(None, f"Pipeline error: {e}")
        durations.append(time.perf_counter() - start)

        rows.append({"company": company, "claim": claim, "gold_label": gold_label, **verdict})

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), durations

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    defaults = BFSConfig()
    p = argparse.ArgumentParser(
        description="EMX-Anchor-BFS: greenwashing classification with anchor entry and "
                    "EmeraldMind graph retrieval over EmeraldGraphX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--claims", default="datasets/green_claims.csv",
                   help="Input CSV with company, claim and gold_label columns.")
    p.add_argument("--embeddings-dir", default=defaults.embeddings_dir,
                   help="Directory with the claim_<i>.json node embeddings of the input CSV.")
    p.add_argument("--output", default="anchor_bfs.csv", help="Output CSV.")
    p.add_argument("--hops", choices=["one", "multi"], default=defaults.hops)
    p.add_argument("--top-k", type=int, default=defaults.top_k_per_class, dest="top_k_per_class",
                   help="Nodes kept per class by the retrieval from each Organization.")
    p.add_argument("--threshold", type=float, default=defaults.similarity_threshold,
                   dest="similarity_threshold", help="Similarity threshold.")
    p.add_argument("--per-class-cap", type=int, default=defaults.per_class_cap,
                   help="Nodes kept per class after the cross-Organization merge.")
    p.add_argument("--hard-ceiling", type=int, default=defaults.hard_ceiling,
                   help="Nodes kept overall after the merge (0 = no cap).")
    p.add_argument("-r", "--rows", type=int, default=None, help="Limit to the first N rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.claims)
    if args.rows is not None:
        df = df.head(args.rows)
    if "gold_label" not in df.columns:
        logger.warning("Input CSV has no 'gold_label' column; metrics will be empty.")

    config = BFSConfig(
        hops=args.hops,
        top_k_per_class=args.top_k_per_class,
        similarity_threshold=args.similarity_threshold,
        per_class_cap=args.per_class_cap,
        hard_ceiling=args.hard_ceiling or None,
        embeddings_dir=args.embeddings_dir,
    )

    try:
        results, durations = run_pipeline(df, config)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        metrics = compute_metrics(results)
        if durations:
            metrics["mean_seconds_per_claim"] = statistics.mean(durations)
            metrics["median_seconds_per_claim"] = statistics.median(durations)
        print(json.dumps(metrics, indent=2))
        print(f"Results written to {args.output}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
