"""Unit tests for the guided beam search (Algorithm 1) in src/EmeraldKG/emx_kgrag.py.

Run from the repository root:  python -m pytest tests -q
"""

from __future__ import annotations

import random

import numpy as np

CLAIM = "the claim"


def _search(mod, enc, ppr=None, org_ids=(0,), **kw):
    params = dict(max_hops=5, beam_width=50, top_k_results=15, similarity_threshold=0.2,
                  score_weights=(0.4, 0.2, 0.4))
    params.update(kw)
    return mod.weighted_beam_search(list(org_ids), ppr or {}, enc.encode(CLAIM), **params)


def _terminals(evidence):
    return [e["node_id"] for e in evidence]


def _random_graph(g, n=200, extra_edges=150, seed=0):
    rng = random.Random(seed)
    classes = ["KPIObservation", "Goal", "Location", "Facility", "SustainabilityClaim",
               "Initiative", "Penalty", "Organization"]
    rels = ["reportsKPI", "setsGoal", "locatedIn", "ownsFacility", "claims",
            "takesPartIn", "subjectToPenalty", "partnersWith"]
    g.node(0, "Organization")
    for i in range(1, n):
        g.node(i, rng.choice(classes), value=rng.randint(0, 99))
        g.edge(rng.randrange(0, i), i, rng.choice(rels))
    for _ in range(extra_edges):
        a, b = rng.sample(range(n), 2)
        g.edge(a, b, rng.choice(rels))
    return rng


# --- R3.C1: role of V_ppr ---------------------------------------------------


def test_ppr_candidate_set_does_not_change_traversal(kgrag):
    """With beta = 0, H is identical for any V_ppr (empty, small, large, disjoint)."""
    mod, g, enc = kgrag
    rng = _random_graph(g)
    weights = (0.5, 0.0, 0.5)
    reference = _search(mod, enc, ppr={}, score_weights=weights, similarity_threshold=-1.0)
    assert reference
    for size in (1, 10, 100, 199):
        ppr = {v: rng.random() for v in rng.sample(range(200), size)}
        assert _search(mod, enc, ppr=ppr, score_weights=weights, similarity_threshold=-1.0) == reference


def test_ppr_count_acts_only_through_truncated_beta_term(kgrag):
    """PPR~(v) = PPR(v)/max PPR on V_ppr and 0 outside, so |V_ppr| can reorder H via beta."""
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "KPIObservation", name="alpha")
    g.node(2, "KPIObservation", name="bravo")
    g.edge(0, 1, "reportsKPI")
    g.edge(0, 2, "reportsKPI")
    enc.sim_overrides = {"name: alpha": 0.5, "name: bravo": 0.5}

    # Identical path score and similarity: only the PPR term separates them.
    only_bravo = _search(mod, enc, ppr={0: 1.0, 2: 0.5}, top_k_results=1)
    only_alpha = _search(mod, enc, ppr={0: 1.0, 1: 0.5}, top_k_results=1)
    assert _terminals(only_bravo) == [2]
    assert _terminals(only_alpha) == [1]


# --- Composite score (Eq. 1) -------------------------------------------------


def test_composite_score_trades_path_score_against_similarity(kgrag):
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "KPIObservation", name="kpi")   # s = 1.0 + 0.5 = 1.5
    g.node(2, "Goal", name="goal")             # s = 0.8
    g.edge(0, 1, "reportsKPI")
    g.edge(0, 2, "setsGoal")
    enc.sim_overrides = {"name: kpi": 0.3, "name: goal": 0.9}

    # 0.4*1.5 + 0.4*0.3 = 0.72  >  0.4*0.8 + 0.4*0.9 = 0.68
    assert _terminals(_search(mod, enc, score_weights=(0.4, 0.0, 0.4))) == [1, 2]
    # 0.2*1.5 + 0.6*0.3 = 0.48  <  0.2*0.8 + 0.6*0.9 = 0.70
    assert _terminals(_search(mod, enc, score_weights=(0.2, 0.0, 0.6))) == [2, 1]


def test_path_score_is_depth_normalised(kgrag):
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "KPIObservation", name="near")   # s = 1.5, depth 1 -> 1.5
    g.node(2, "Initiative", name="init")
    g.node(3, "KPIObservation", name="far")    # s = 0.6 + 1.5 = 2.1, depth 2 -> 1.05
    g.edge(0, 1, "reportsKPI")
    g.edge(0, 2, "takesPartIn")
    g.edge(2, 3, "reportsKPI")
    enc.sim_overrides = {"name: near": 0.5, "name: far": 0.5, "name: init": 0.0}

    assert _terminals(_search(mod, enc, score_weights=(1.0, 0.0, 0.0))) == [1, 3]


def test_similarity_threshold_filters_terminals(kgrag):
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "Goal", name="keep")
    g.node(2, "Goal", name="drop")
    g.edge(0, 1, "setsGoal")
    g.edge(0, 2, "setsGoal")
    enc.sim_overrides = {"name: keep": 0.25, "name: drop": 0.15}

    evidence = _search(mod, enc, similarity_threshold=0.2)
    assert _terminals(evidence) == [1]
    assert all(e["similarity"] >= 0.2 for e in evidence)


# --- Traversal ---------------------------------------------------------------


def test_traversal_is_undirected(kgrag):
    """A node pointing *into* the organisation is reachable."""
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "Facility", name="plant")
    g.edge(1, 0, "ownedBy")
    enc.sim_overrides = {"name: plant": 0.9}

    assert _terminals(_search(mod, enc)) == [1]


def test_paths_are_simple_rooted_and_bounded(kgrag):
    mod, g, enc = kgrag
    _random_graph(g, seed=1)
    for k in (1, 2, 3, 5):
        evidence = _search(mod, enc, max_hops=k, similarity_threshold=-1.0, top_k_results=10_000)
        assert evidence
        terminals = _terminals(evidence)
        assert len(terminals) == len(set(terminals))          # one path per terminal
        for e in evidence:
            (path,) = e["paths"]
            ids = [n["id"] for n in path["path_nodes"]]
            assert ids[0] == 0 and ids[-1] == e["node_id"]   # rooted at the anchor
            assert len(ids) == len(set(ids))                  # simple path
            assert 1 <= path["path_length"] == len(path["rel_types"]) == len(ids) - 1 <= k


def test_anchor_nodes_are_never_traversed(kgrag):
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "Anchor", name="anchor")
    g.node(2, "Organization", name="sibling")
    g.edge(1, 0, "REPRESENTS")
    g.edge(1, 2, "REPRESENTS")
    enc.sim_overrides = {"name: anchor": 0.9, "name: sibling": 0.9}

    assert _search(mod, enc) == []


def test_all_expanded_paths_survive_beam_pruning(kgrag):
    """P_all keeps a hop-1 node that the beam pruned; its descendants are never reached."""
    mod, g, enc = kgrag
    g.node(0, "Organization")
    g.node(1, "KPIObservation", name="kpi")     # s = 1.5, wins the beam
    g.node(2, "Location", name="site")          # s = 0.2, pruned at beam = 1
    g.node(3, "Facility", name="behind")        # only reachable through node 2
    g.edge(0, 1, "reportsKPI")
    g.edge(0, 2, "locatedIn")
    g.edge(2, 3, "locatedIn")
    enc.sim_overrides = {"name: kpi": 0.3, "name: site": 0.95, "name: behind": 0.95}

    terminals = _terminals(_search(mod, enc, beam_width=1))
    assert 2 in terminals
    assert 3 not in terminals
    assert 3 in _terminals(_search(mod, enc, beam_width=2))


def test_neighbour_expansions_bounded_by_beam(kgrag):
    """At most |org_ids| + (k-1)*beam neighbourhood expansions per claim."""
    mod, g, enc = kgrag
    _random_graph(g, n=400, extra_edges=600, seed=2)
    for beam, k in ((1, 5), (5, 3), (10, 5)):
        g.neighbour_queries = 0
        _search(mod, enc, beam_width=beam, max_hops=k)
        assert g.neighbour_queries <= 1 + (k - 1) * beam


# --- Safety net --------------------------------------------------------------


def test_always_include_adds_missing_penalty(kgrag):
    mod, g, enc = kgrag
    g.node(0, "Organization")
    goals = [g.node(i, "Goal", name=f"goal{chr(96 + i)}") for i in range(1, 7)]
    for i in goals:
        g.edge(0, i, "setsGoal")
    g.node(10, "Penalty", name="fine")
    g.edge(0, 10, "subjectToPenalty")
    enc.sim_overrides = {**{f"name: goal{chr(96 + i)}": 0.9 for i in goals}, "name: fine": 0.21}

    # top_k_per_class = 1 -> top_n = 5 Goals, so the beam output contains no Penalty.
    config = mod.RetrievalConfig(top_k_per_class=1)
    mod.run_ppr = lambda org_ids, top_n, damping: {}
    evidence, company_id = mod.retrieve_evidence_beam_search("acme", CLAIM, config)

    assert company_id == 0
    fallback = [e for e in evidence if e["embedding_source"] == "ALWAYS_INCLUDE fallback"]
    assert [e["node_id"] for e in fallback] == [10]
    assert sum(1 for e in evidence if e["labels"] == ["Goal"]) == 5
