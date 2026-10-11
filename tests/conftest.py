"""Load emx_kgrag against an in-memory fake Neo4j, so tests need no database, GPU or LLM.

``utils.classification_utils`` connects to Neo4j and loads Gemma at import time,
so it is replaced by a stub exposing the names emx_kgrag imports.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "EmeraldKG" / "emx_kgrag.py"


class FakeGraph:
    """Directed labelled multigraph answering the Cypher queries emx_kgrag issues."""

    def __init__(self):
        self.nodes: dict[int, tuple[list[str], dict]] = {}
        self.edges: list[tuple[int, int, str]] = []
        self.neighbour_queries = 0

    def node(self, nid: int, label: str, **props) -> int:
        self.nodes[nid] = ([label], {"name": f"n{nid}", **props})
        return nid

    def edge(self, src: int, dst: int, rel: str) -> None:
        self.edges.append((src, dst, rel))

    def _incident(self, n: int):
        # Undirected: both outgoing and incoming relationships.
        for a, b, r in self.edges:
            if a == n:
                yield b, r
            if b == n:
                yield a, r

    def run(self, query: str, **kw):
        if "Anchor" in query:
            return _Result([])
        if "NOT id(m) IN $visited" in query:
            self.neighbour_queries += 1
            visited = set(kw["visited"])
            return _Result([
                {"m_id": m, "labels": list(self.nodes[m][0]), "rel_type": r,
                 "props": dict(self.nodes[m][1])}
                for m, r in self._incident(kw["node_id"]) if m not in visited
            ])
        if "RETURN id(n) AS id" in query:
            labels, props = self.nodes[kw["nid"]]
            return _Result([{"id": kw["nid"], "labels": list(labels), "properties": dict(props)}])
        if "company:Organization" in query:
            cls = query.split("node:")[1].split(")")[0]
            exclude = set(kw["exclude"])
            rows = [
                {"node_id": m, "node": dict(self.nodes[m][1]), "labels": list(self.nodes[m][0]),
                 "rel_type": r}
                for m, r in self._incident(kw["company_id"])
                if m not in exclude and self.nodes[m][0][0] == cls
            ]
            return _Result(sorted(rows, key=lambda x: x["node_id"]))
        raise AssertionError(f"unexpected query: {query}")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        return self.graph

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self):
        self.graph = FakeGraph()

    def session(self):
        return _Session(self.graph)

    def close(self):
        pass


class HashEncoder:
    """Deterministic unit vectors; ``sim_overrides`` pins a node text's similarity to the claim."""

    CLAIM = "the claim"

    def __init__(self):
        self.sim_overrides: dict[str, float] = {}

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        v = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(float) - 127.5
        return v / np.linalg.norm(v)

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        if text == self.CLAIM:
            return np.eye(32)[0]
        for marker, sim in self.sim_overrides.items():
            if marker in text:
                # Unit vector at angle acos(sim) from the claim axis.
                return np.eye(32)[0] * sim + np.eye(32)[1] * np.sqrt(max(0.0, 1 - sim * sim))
        return self._vec(text)


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture
def kgrag():
    """Fresh emx_kgrag module bound to an empty FakeGraph; returns (module, graph, encoder)."""
    driver, encoder = FakeDriver(), HashEncoder()
    stub = types.ModuleType("utils.classification_utils")
    stub.ALWAYS_INCLUDE_CLASSES = ["KPIObservation", "Penalty"]
    stub.classify_few = None
    stub.cosine_similarity = _cosine
    stub.driver = driver
    stub.encoder = encoder
    stub.gds = None
    stub.get_company_id = lambda name: 0
    sys.modules["utils"] = types.ModuleType("utils")
    sys.modules["utils.classification_utils"] = stub

    spec = importlib.util.spec_from_file_location("emx_kgrag", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["emx_kgrag"] = module
    spec.loader.exec_module(module)
    yield module, driver.graph, encoder
    for name in ("emx_kgrag", "utils", "utils.classification_utils"):
        sys.modules.pop(name, None)
