"""Unit tests for the EMX-Anchor-BFS comparator in src/EmeraldKG/emx_anchor_bfs.py.

Run from the repository root:  python -m pytest tests -q
"""

from __future__ import annotations


def _ev(node_id, label, sim):
    return {"node_id": node_id, "labels": [label], "similarity": sim}


def _ids(evidence):
    return [e["node_id"] for e in evidence]


# --- Merge across Organizations ------------------------------------------------


def test_merge_keeps_highest_similarity_per_node(anchor_bfs):
    mod, _, _ = anchor_bfs
    merged = mod.merge_evidence([[_ev(1, "Goal", 0.4)], [_ev(1, "Goal", 0.7), _ev(2, "Goal", 0.5)]])
    assert _ids(merged) == [1, 2]
    assert merged[0]["similarity"] == 0.7


def test_merge_caps_per_class_and_overall(anchor_bfs):
    mod, _, _ = anchor_bfs
    org_a = [_ev(i, "Goal", 0.9 - i / 100) for i in range(5)]
    org_b = [_ev(10 + i, "KPIObservation", 0.5 - i / 100) for i in range(5)]

    merged = mod.merge_evidence([org_a, org_b], per_class_cap=3, hard_ceiling=None)
    assert _ids(merged) == [0, 1, 2, 10, 11, 12]

    assert _ids(mod.merge_evidence([org_a, org_b], per_class_cap=3, hard_ceiling=4)) == [0, 1, 2, 10]


# --- Entry and retrieval -------------------------------------------------------


def test_retrieval_runs_from_every_anchor_linked_org(anchor_bfs):
    mod, _, _ = anchor_bfs
    calls = []

    def fake_retrieve(org_id, claim, claim_idx, top_k_per_class, similarity_threshold, embeddings_dir):
        calls.append((org_id, claim_idx, top_k_per_class, similarity_threshold, embeddings_dir))
        return [_ev(100 + org_id, "Goal", 0.5 + org_id / 10)]

    mod.get_anchor_org_ids = lambda name: [1, 2]
    mod.retrieve_evidence = fake_retrieve
    mod.retrieve_evidence_with_paths = None   # hops="one" must not use it

    evidence, org_ids = mod.retrieve_evidence_anchor_bfs("acme", "the claim", 7, mod.BFSConfig())
    assert org_ids == [1, 2]
    assert calls == [(1, 7, 3, 0.2, "claims"), (2, 7, 3, 0.2, "claims")]
    assert _ids(evidence) == [102, 101]


def test_multi_hop_uses_path_retrieval_and_matcher_fallback(anchor_bfs):
    """No Anchor matches in the fake graph, so CompanyMatcher resolves to Organization 0."""
    mod, _, _ = anchor_bfs
    mod.retrieve_evidence = None
    mod.retrieve_evidence_with_paths = lambda org_id, claim, **kw: [_ev(5, "Penalty", 0.3)]

    evidence, org_ids = mod.retrieve_evidence_anchor_bfs(
        "acme", "the claim", 0, mod.BFSConfig(hops="multi"))
    assert org_ids == [0]
    assert _ids(evidence) == [5]


def test_retrieval_errors_abstain(anchor_bfs):
    mod, _, _ = anchor_bfs

    def failing(*args, **kw):
        raise RuntimeError("boom")

    mod.get_anchor_org_ids = lambda name: [1]
    mod.retrieve_evidence = failing
    verdict = mod.classify_claim("acme", "the claim", 0, mod.BFSConfig())
    assert verdict["predicted_label"] == "abstain"
    assert verdict["context"] == "(No context found)"


def test_unknown_llm_label_becomes_abstain(anchor_bfs):
    mod, _, _ = anchor_bfs
    mod.get_anchor_org_ids = lambda name: [1]
    mod.retrieve_evidence = lambda org_id, claim, **kw: [_ev(3, "Goal", 0.6)]

    mod.classify_few = lambda ev, company, claim: ({"label": "Greenwashing ", "reasoning": "r"}, "ctx", "", {})
    assert mod.classify_claim("acme", "the claim", 0, mod.BFSConfig())["predicted_label"] == "greenwashing"

    mod.classify_few = lambda ev, company, claim: ({"label": "unclear"}, "ctx", "", {})
    assert mod.classify_claim("acme", "the claim", 0, mod.BFSConfig())["predicted_label"] == "abstain"
