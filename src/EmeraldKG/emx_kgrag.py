#!/usr/bin/env python3
"""
EMX-KGRAG: greenwashing classification over EmeraldGraphX with guided beam search.

Implements Algorithm 1 ("Context Retrieval via Guided Beam Search") of the
EmeraldMindX paper:

  Entry       Resolve the company via (Anchor)-[:REPRESENTS]->(Organization);
              fall back to the four-tier ``CompanyMatcher``.
  Phase 1     Personalized PageRank from the Organization node(s) gives the
              top-``ppr`` candidates V_ppr. A weighted beam search then expands
              paths over undirected edges for up to ``k`` hops, scoring each step
              by edge-type weight plus node-type bonus and keeping ``beam`` paths
              per hop. Every expanded path is kept in P_all. V_ppr does not
              restrict the traversal.
  Phase 2     Keep the best path per terminal node, drop terminals with cosine
              similarity to the claim below ``theta``, and rank the rest by
                  s_norm = alpha * s/|path| + beta * PPR~(v_t) + gamma * sim
              where PPR~(v) = PPR(v) / max_{V_ppr} PPR if v in V_ppr, else 0.
              The top-``n`` paths form the evidence subgraph H.
  Safety net  KPIObservation / Penalty nodes one hop from the company are added
              when the beam search surfaced none.

The evidence is passed to ``classify_few`` (Gemma 3 27B) and the verdicts are
written in the CSV format read by the voting scripts.

Usage
-----
    python src/EmeraldKG/emx_kgrag.py --claims datasets/green_claims.csv \
        --output results/green_claims_kgrag.csv
    python src/EmeraldKG/emx_kgrag.py --claims datasets/green_claims.csv \
        --sweep --output-dir results/beam_search_sweep
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field, replace
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.emx_classification_utils import (
    ALWAYS_INCLUDE_CLASSES,
    CompanyMatcher,
    classify_few,
    cosine_similarity,
    driver,
    encoder,
    gds,
)

basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)

POS_LABEL = "greenwashing"
NEG_LABEL = "not_greenwashing"
ABSTAIN_LABEL = "abstain"
VALID_LABELS = {POS_LABEL, NEG_LABEL}

# Columns read by src/voting_scripts (see results/emx_pipelines/*_kgrag.csv).
OUTPUT_COLUMNS = [
    "company", "claim", "gold_label", "similarities", "context",
    "predicted_label", "justification", "cited_node_ids",
]

# ---------------------------------------------------------------------------
# Heuristic: edge-type weights and node-type bonuses (UpdateScores in Alg. 1)
# ---------------------------------------------------------------------------

EDGE_WEIGHTS: dict[str, float] = {
    # High value for greenwashing detection
    "reportsKPI": 1.0,
    "subjectToPenalty": 1.0,
    "contradictedBy": 1.0,
    "contradictedByMedia": 0.9,
    "claims": 0.9,
    "verifiedBy": 0.9,
    "violatesRegulation": 0.9,
    "targetsScienceBased": 0.8,
    "setsGoal": 0.8,
    "holdsCertification": 0.8,
    "adoptsStandard": 0.7,
    "subjectToRegulation": 0.7,
    # Medium value
    "offsetsWith": 0.6,
    "publishesReport": 0.6,
    "takesPartIn": 0.6,
    "investsIn": 0.5,
    "governedBy": 0.5,
    "identifies": 0.5,
    # Lower value
    "partOf": 0.4,
    "ownsFacility": 0.4,
    "worksAt": 0.3,
    "locatedIn": 0.2,
    "isIn": 0.2,
    # REPRESENTS is used for entry only, not scored
    "REPRESENTS": 0.0,
}

DEFAULT_EDGE_WEIGHT = 0.3

NODE_BONUSES: dict[str, float] = {
    "KPIObservation": 0.5,
    "Penalty": 0.4,
    "Controversy": 0.3,
    "SustainabilityClaim": 0.3,
    "ThirdPartyVerification": 0.2,
    "ScienceBasedTarget": 0.2,
    "MediaReport": 0.2,
}

# Node properties left out of the text that is embedded for similarity.
_SKIP_PROPS = ("valid_from", "valid_to", "is_current", "embedding")


@dataclass(frozen=True)
class RetrievalConfig:
    """Hyperparameters of Algorithm 1; defaults follow the paper's Table (tab:hyper)."""

    top_k_per_class: int = 3          # beam search returns top_k_per_class * 5 paths
    similarity_threshold: float = 0.2  # theta
    max_hops: int = 5                  # k
    beam_width: int = 50               # beam
    ppr_top_n: int = 100               # ppr
    ppr_damping: float = 0.85
    score_weights: Tuple[float, float, float] = (0.4, 0.2, 0.4)  # (alpha, beta, gamma)


# Grid of the parameter-tuning experiment (Section "Guided Beam Search Parameter
# Tuning"): every (alpha, beta, gamma) sums to 1.0.
SCORE_CONFIGS: list[tuple[str, tuple[float, float, float]]] = [
    ("graph_heavy",   (0.55, 0.30, 0.15)),
    ("graph_leaning", (0.45, 0.25, 0.30)),
    ("balanced",      (0.40, 0.20, 0.40)),
    ("sim_leaning",   (0.30, 0.20, 0.50)),
    ("ppr_heavy",     (0.25, 0.50, 0.25)),
    ("sim_heavy",     (0.20, 0.15, 0.65)),
]
SWEEP_BEAM_WIDTHS = [25, 50, 75, 100]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RX_PUNCT = re.compile(r"[^\w\s]")


def _clean_name(n: Any) -> str:
    """Normalise a company name the way Anchor.clean_name is stored."""
    if not isinstance(n, str) or not n.strip():
        return ""
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = _RX_PUNCT.sub(" ", n.lower()).strip()
    return " ".join(n.split())


def node_similarity(node_class: str, props: dict, claim_embedding: np.ndarray) -> float:
    """Cosine similarity between the claim and a node's textual rendering."""
    text_parts = [f"{node_class}:"]
    for key, value in sorted(props.items()):
        if value is not None and key not in _SKIP_PROPS:
            text_parts.append(f"{key}: {value}")
    node_emb = encoder.encode(" | ".join(text_parts), normalize_embeddings=True)
    return float(cosine_similarity(claim_embedding, node_emb))


@dataclass(order=True)
class PathState:
    """A partial path in the beam, ordered by negative score (max-heap via min-heap)."""
    neg_score: float
    node_ids: list[int] = field(compare=False)
    edge_types: list[str] = field(compare=False)
    visited: set[int] = field(compare=False)
    # Metadata for the terminal node
    terminal_labels: list[str] = field(compare=False, default_factory=list)
    terminal_props: dict = field(compare=False, default_factory=dict)

    @property
    def score(self) -> float:
        return -self.neg_score

    @property
    def depth(self) -> int:
        return len(self.node_ids) - 1

    @property
    def last_node(self) -> int:
        return self.node_ids[-1]

# ---------------------------------------------------------------------------
# Entry point and Personalized PageRank
# ---------------------------------------------------------------------------


_matcher: CompanyMatcher | None = None


def _company_matcher() -> CompanyMatcher:
    """Fallback matcher, built on first use (it loads and embeds every Organization)."""
    global _matcher
    if _matcher is None:
        _matcher = CompanyMatcher(driver, encoder)
    return _matcher


def get_anchor_org_ids(company_name: str) -> list[int]:
    """Look up the Organization node IDs an Anchor node represents."""
    with driver.session() as sess:
        result = sess.run(
            """
            MATCH (a:Anchor)-[:REPRESENTS]->(o:Organization)
            WHERE a.clean_name = $clean OR a.name = $name
            RETURN DISTINCT id(o) AS oid
            """,
            clean=_clean_name(company_name), name=company_name,
        ).data()
    return [r["oid"] for r in result]


def run_ppr(org_ids: list[int], top_n: int = 100, damping: float = 0.85) -> dict[int, float]:
    """Personalized PageRank from ``org_ids``; returns {node_id: score} for the top-N nodes."""
    graph_name = "_ppr_temp_graph"

    try:
        gds.graph.drop(gds.graph.get(graph_name))
    except Exception:
        pass

    try:
        G, _ = gds.graph.project(graph_name, "*", "*")
        result = gds.pageRank.stream(
            G,
            sourceNodes=org_ids,
            dampingFactor=damping,
            maxIterations=20,
        )
        top_rows = result.nlargest(top_n, "score")
        ppr_scores = {
            int(row["nodeId"]): float(row["score"])
            for _, row in top_rows.iterrows()
        }
        logger.info("PPR returned %d candidates (top-%d).", len(ppr_scores), top_n)
    except Exception as e:
        logger.warning("PPR failed: %s. Falling back to empty candidates.", e)
        ppr_scores = {}
    finally:
        try:
            gds.graph.drop(gds.graph.get(graph_name))
        except Exception:
            pass

    return ppr_scores

# ---------------------------------------------------------------------------
# Guided beam search (Algorithm 1)
# ---------------------------------------------------------------------------


def weighted_beam_search(
    org_ids: list[int],
    ppr_candidates: dict[int, float],
    claim_embedding: np.ndarray,
    max_hops: int = 5,
    beam_width: int = 50,
    top_k_results: int = 20,
    similarity_threshold: float = 0.2,
    score_weights: tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> list[dict]:
    """Return the top paths from ``org_ids`` as evidence dicts (format of
    ``retrieve_evidence_with_paths``, consumed by ``format_entities_as_json``)."""
    alpha, beta, gamma = score_weights

    # Phase 1: heuristic beam search
    beam: list[PathState] = []
    for oid in org_ids:
        heapq.heappush(beam, PathState(neg_score=0.0, node_ids=[oid], edge_types=[], visited={oid}))

    all_paths: list[PathState] = []  # P_all

    with driver.session() as session:
        for hop in range(max_hops):
            next_beam: list[PathState] = []

            for state in beam:
                neighbors = session.run(
                    """
                    MATCH (n)-[r]-(m)
                    WHERE id(n) = $node_id AND NOT id(m) IN $visited
                    RETURN id(m) AS m_id, labels(m) AS labels,
                           type(r) AS rel_type, properties(m) AS props
                    """,
                    node_id=state.last_node,
                    visited=list(state.visited),
                ).data()

                for neighbor in neighbors:
                    m_id = neighbor["m_id"]
                    labels = neighbor["labels"]
                    rel_type = neighbor["rel_type"]

                    # Anchor nodes are entry points only
                    if "Anchor" in labels:
                        continue

                    node_class = labels[0] if labels else "Unknown"
                    new_score = (
                        state.score
                        + EDGE_WEIGHTS.get(rel_type, DEFAULT_EDGE_WEIGHT)
                        + NODE_BONUSES.get(node_class, 0.0)
                    )
                    new_state = PathState(
                        neg_score=-new_score,
                        node_ids=state.node_ids + [m_id],
                        edge_types=state.edge_types + [rel_type],
                        visited=state.visited | {m_id},
                        terminal_labels=sorted(labels),
                        terminal_props=neighbor["props"],
                    )
                    all_paths.append(new_state)
                    heapq.heappush(next_beam, new_state)

            beam = heapq.nsmallest(beam_width, next_beam)
            logger.debug("  Hop %d: beam size=%d, explored=%d", hop + 1, len(beam), len(all_paths))

    if not all_paths:
        return []

    # Phase 2: semantic filtering and composite scoring.
    # Keep the max-score path per terminal node.
    best_by_terminal: dict[int, PathState] = {}
    for state in all_paths:
        terminal = state.last_node
        if terminal not in best_by_terminal or state.score > best_by_terminal[terminal].score:
            best_by_terminal[terminal] = state

    max_ppr = max(ppr_candidates.values()) if ppr_candidates else 1.0
    scored: list[tuple[float, float, PathState]] = []

    for terminal_id, state in best_by_terminal.items():
        labels = state.terminal_labels
        node_class = labels[0] if labels else "Unknown"
        sim = node_similarity(node_class, state.terminal_props or {}, claim_embedding)
        if sim < similarity_threshold:
            continue

        ppr_score = ppr_candidates.get(terminal_id, 0.0) / max_ppr if max_ppr > 0 else 0.0
        depth = state.depth if state.depth > 0 else 1
        s_norm = alpha * (state.score / depth) + beta * ppr_score + gamma * sim
        scored.append((s_norm, sim, state))

    scored.sort(key=lambda x: -x[0])

    evidence: list[dict] = []
    for _, sim, state in scored[:top_k_results]:
        evidence.append({
            "node_id": state.last_node,
            "labels": state.terminal_labels,
            "properties": state.terminal_props or {},
            "similarity": sim,
            "embedding_source": "hybrid PPR + beam search",
            "paths": [_fetch_path_details(state.node_ids, state.edge_types)],
        })
    return evidence


def _fetch_path_details(node_ids: list[int], edge_types: list[str]) -> dict:
    """Fetch the full properties of every node on a path."""
    path_nodes = []
    with driver.session() as session:
        for nid in node_ids:
            result = session.run(
                """
                MATCH (n) WHERE id(n) = $nid
                RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
                """,
                nid=nid,
            ).data()
            if result:
                path_nodes.append(result[0])

    return {
        "path_nodes": path_nodes,
        "rel_types": edge_types,
        "path_length": len(edge_types),
    }


def always_include_fallback(
    company_id: int,
    claim_embedding: np.ndarray,
    existing_node_ids: set[int],
    top_k: int = 3,
    similarity_threshold: float = 0.2,
) -> list[dict]:
    """Top-k 1-hop ALWAYS_INCLUDE_CLASSES neighbours of the company not already in evidence."""
    extra_evidence: list[dict] = []

    with driver.session() as session:
        for class_name in sorted(ALWAYS_INCLUDE_CLASSES):
            query = f"""
            MATCH (company:Organization)-[r]-(node:{class_name})
            WHERE id(company) = $company_id
              AND NOT id(node) IN $exclude
            RETURN DISTINCT
                id(node) AS node_id,
                node,
                labels(node) AS labels,
                type(r) AS rel_type
            ORDER BY node_id
            """
            try:
                result = session.run(
                    query, company_id=company_id, exclude=list(existing_node_ids),
                ).data()
            except Exception as e:
                logger.warning("ALWAYS_INCLUDE query failed for %s: %s", class_name, e)
                continue

            candidates = []
            for record in result:
                props = dict(record["node"])
                sim = node_similarity(class_name, props, claim_embedding)
                if sim >= similarity_threshold:
                    candidates.append({**record, "props": props, "similarity": sim})

            candidates.sort(key=lambda x: (-x["similarity"], x["node_id"]))

            for item in candidates[:top_k]:
                extra_evidence.append({
                    "node_id": item["node_id"],
                    "labels": sorted(item["labels"]),
                    "properties": item["props"],
                    "similarity": item["similarity"],
                    "embedding_source": "ALWAYS_INCLUDE fallback",
                    "paths": [{
                        "path_nodes": [
                            {"id": company_id, "labels": ["Organization"], "properties": {}},
                            {"id": item["node_id"], "labels": item["labels"],
                             "properties": item["props"]},
                        ],
                        "rel_types": [item["rel_type"]],
                        "path_length": 1,
                    }],
                })

    return extra_evidence


def retrieve_evidence_beam_search(
    company_name: str,
    claim: str,
    config: RetrievalConfig = RetrievalConfig(),
) -> tuple[list[dict], int | None]:
    """Retrieve the evidence subgraph H for a claim. Returns (evidence, company_id)."""
    org_ids = get_anchor_org_ids(company_name)
    if org_ids:
        logger.info("Anchor resolved '%s' to %d Organization node(s).", company_name, len(org_ids))
    else:
        company_id = _company_matcher().get_company_id(company_name)
        if company_id is None:
            logger.warning("No match for company '%s'.", company_name)
            return [], None
        org_ids = [company_id]
    company_id = org_ids[0]

    ppr_candidates = run_ppr(org_ids, top_n=config.ppr_top_n, damping=config.ppr_damping)
    claim_embedding = encoder.encode(claim, normalize_embeddings=True)

    evidence = weighted_beam_search(
        org_ids=org_ids,
        ppr_candidates=ppr_candidates,
        claim_embedding=claim_embedding,
        max_hops=config.max_hops,
        beam_width=config.beam_width,
        top_k_results=config.top_k_per_class * 5,
        similarity_threshold=config.similarity_threshold,
        score_weights=config.score_weights,
    )

    covered_classes = {e["labels"][0] for e in evidence if e["labels"]}
    missing_always = [c for c in ALWAYS_INCLUDE_CLASSES if c not in covered_classes]
    if missing_always:
        logger.info("ALWAYS_INCLUDE fallback for: %s", missing_always)
        evidence.extend(always_include_fallback(
            company_id=company_id,
            claim_embedding=claim_embedding,
            existing_node_ids={e["node_id"] for e in evidence},
            top_k=config.top_k_per_class,
            similarity_threshold=config.similarity_threshold,
        ))

    # Final sort for determinism
    evidence.sort(
        key=lambda x: (
            -x["similarity"],
            x["node_id"] if x["node_id"] is not None else 0,
            x["labels"][0] if x["labels"] else "",
        )
    )
    logger.info("Total evidence nodes: %d", len(evidence))
    return evidence, company_id

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _abstain(context: Optional[str], reason: str, similarities: str = "{}") -> Dict[str, Any]:
    return {"predicted_label": ABSTAIN_LABEL, "context": context, "justification": reason,
            "similarities": similarities, "cited_node_ids": []}


def classify_claim(company: str, claim: str, config: RetrievalConfig) -> Dict[str, Any]:
    """Retrieve evidence for one claim and classify it; returns the verdict fields."""
    evidence, company_id = retrieve_evidence_beam_search(company, claim, config)

    if company_id is None:
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


def run_pipeline(df: pd.DataFrame, config: RetrievalConfig) -> tuple[pd.DataFrame, list[float]]:
    """Classify every claim in ``df``; returns the output rows and per-claim seconds."""
    rows: list[dict] = []
    durations: list[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        company = str(row.get("company", "")).strip() if pd.notna(row.get("company")) else ""
        claim = str(row.get("claim", "")).strip() if pd.notna(row.get("claim")) else ""
        gold = row.get("gold_label")
        gold_label = str(gold).strip().lower() if pd.notna(gold) else None

        # Failed rows are kept as abstains so they count against coverage.
        start = time.perf_counter()
        if not claim or not company:
            logger.warning("Row %s: missing claim or company.", row.name)
            verdict = _abstain("(Skipped)", "Missing claim or company.")
        else:
            try:
                verdict = classify_claim(company, claim, config)
            except Exception as e:
                logger.error("Pipeline error for row %s: %s", row.name, e)
                verdict = _abstain(None, f"Pipeline error: {e}")
        durations.append(time.perf_counter() - start)

        rows.append({"company": company, "claim": claim, "gold_label": gold_label, **verdict})

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), durations


def compute_metrics(results: pd.DataFrame) -> Dict[str, Any]:
    """Accuracy over non-abstained claims, coverage, and overall accuracy = accuracy * coverage.

    Only rows whose gold label is greenwashing / not_greenwashing are evaluable.
    """
    evaluable = results[results["gold_label"].isin(VALID_LABELS)]
    committed = evaluable[evaluable["predicted_label"].isin(VALID_LABELS)]
    correct = int((committed["predicted_label"] == committed["gold_label"]).sum())

    total, non_abstain = len(evaluable), len(committed)
    accuracy = correct / non_abstain if non_abstain else 0.0
    coverage = non_abstain / total if total else 0.0
    return {
        "total": total,
        "non_abstain": non_abstain,
        "abstain": total - non_abstain,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "overall_accuracy": accuracy * coverage,
    }


def run_sweep(df: pd.DataFrame, base: RetrievalConfig, beam_widths: List[int], output_dir: Path) -> None:
    """Grid search over beam widths x SCORE_CONFIGS (paper Table tab:ablation)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(beam_widths) * len(SCORE_CONFIGS)
    ablation_rows: list[dict[str, Any]] = []

    for run_idx, (beam_width, (cfg_name, weights)) in enumerate(
        ((b, c) for b in beam_widths for c in SCORE_CONFIGS), start=1
    ):
        config = replace(base, beam_width=beam_width, score_weights=weights)
        run_csv = output_dir / f"beam{beam_width}_{cfg_name}.csv"
        print(f"\n[{run_idx}/{total_runs}] beam_width={beam_width} config={cfg_name} "
              f"weights={weights} -> {run_csv}")

        results, durations = run_pipeline(df, config)
        results.to_csv(run_csv, index=False)
        m = compute_metrics(results)

        ablation_rows.append({
            "beam_width": beam_width,
            "config": cfg_name,
            "alpha": weights[0],
            "beta": weights[1],
            "gamma": weights[2],
            **m,
            "mean_seconds_per_claim": statistics.mean(durations) if durations else 0.0,
            "output_csv": str(run_csv),
        })
        print(f"    accuracy={m['accuracy']:.4f}  coverage={m['coverage']:.4f}  "
              f"overall_accuracy={m['overall_accuracy']:.4f}  "
              f"(correct={m['correct']}/{m['non_abstain']} of {m['total']})")

    ablation_df = pd.DataFrame(ablation_rows).sort_values("overall_accuracy", ascending=False)
    ablation_csv = output_dir / "ablation_metrics.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print("\n" + ablation_df.to_string(index=False))
    print(f"\nAblation metrics saved to: {ablation_csv}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    defaults = RetrievalConfig()
    p = argparse.ArgumentParser(
        description="EMX-KGRAG: greenwashing classification with guided beam search over EmeraldGraphX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--claims", default="datasets/green_claims.csv",
                   help="Input CSV with company, claim and gold_label columns.")
    p.add_argument("--output", default="kgrag.csv", help="Output CSV (single run).")
    p.add_argument("--top-k", type=int, default=defaults.top_k_per_class, dest="top_k_per_class",
                   help="Per-class k; the beam search keeps top_k * 5 paths.")
    p.add_argument("--threshold", type=float, default=defaults.similarity_threshold,
                   dest="similarity_threshold", help="Similarity threshold theta.")
    p.add_argument("--max-hops", type=int, default=defaults.max_hops, help="Max path length k.")
    p.add_argument("--beam-width", type=int, default=defaults.beam_width)
    p.add_argument("--ppr-top-n", type=int, default=defaults.ppr_top_n)
    p.add_argument("--ppr-damping", type=float, default=defaults.ppr_damping)
    p.add_argument("--weights", type=float, nargs=3, default=list(defaults.score_weights),
                   metavar=("ALPHA", "BETA", "GAMMA"), help="Composite score weights.")
    p.add_argument("--sweep", action="store_true",
                   help="Run the beam-width x weights grid search instead of a single run.")
    p.add_argument("--beam-widths", type=int, nargs="+", default=SWEEP_BEAM_WIDTHS,
                   help="Beam widths for --sweep.")
    p.add_argument("--output-dir", default="beam_search_sweep", help="Output directory for --sweep.")
    p.add_argument("-r", "--rows", type=int, default=None, help="Limit to the first N rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.claims)
    if args.rows is not None:
        df = df.head(args.rows)
    if "gold_label" not in df.columns:
        logger.warning("Input CSV has no 'gold_label' column; metrics will be empty.")

    config = RetrievalConfig(
        top_k_per_class=args.top_k_per_class,
        similarity_threshold=args.similarity_threshold,
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        ppr_top_n=args.ppr_top_n,
        ppr_damping=args.ppr_damping,
        score_weights=tuple(args.weights),
    )

    try:
        if args.sweep:
            run_sweep(df, config, args.beam_widths, Path(args.output_dir))
        else:
            results, _ = run_pipeline(df, config)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(args.output, index=False)
            print(json.dumps(compute_metrics(results), indent=2))
            print(f"Results written to {args.output}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
