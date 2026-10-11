"""Retrieval, formatting and classification helpers used by the EmeraldMindX pipelines.

classification_utils.py serves the EmeraldMind scripts in src/EmeraldKG (1-7).
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json

import torch
import numpy as np
import re, unicodedata
from rapidfuzz import fuzz, process
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM)

from dotenv import load_dotenv

from utils.globals import (
    ALWAYS_INCLUDE_CLASSES,
    FEW_TMPL
)

# MODEL = "google/gemma-3-27b-it"

# # variable used in names
# MODEL_NAME="gemma3-27b" 
MODEL = "google/gemma-3-27b-it"

# variable used in names
MODEL_NAME="gemma3-27b" 

load_dotenv()

try:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "s3cr3tpass")),
    )
    driver.verify_connectivity()
    gds = GraphDataScience(driver)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Neo4j: {e}")
encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.bfloat16
).eval()


_RX_PUNCT = re.compile(r"[\\W_]+", re.U)

# Globals
_RX_PUNCT = re.compile(r"[^\w\s]")
FUZZY_MIN = 65
EMBED_MIN = 0.7


def _clean_name(n: Any) -> str:
    """
    Normalise company names for fuzzy matching.
    Returns '' for None / non‑str inputs.
    Leaves corporate suffixes (Bank, Bancorp, etc.) intact!
    """
    if not isinstance(n, str) or not n.strip():
        return ""

    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()

    # Strip punctuation, lowercase, and remove extra whitespace
    n = _RX_PUNCT.sub(" ", n.lower()).strip()
    return " ".join(n.split())


class CompanyMatcher:
    def __init__(self, driver, encoder):
        self.driver = driver
        self.encoder = encoder

        # Caches
        self.orgs = []  # List of tuples: (oid, raw_name, clean_name)
        self.db_embeddings = None  # NumPy matrix of shape (N, Embedding_Dim)

        self._load_and_precompute()

    def _load_and_precompute(self):
        print("INFO: Loading organizations and precomputing embeddings...")
        with self.driver.session() as sess:
            rows = sess.run(
                "MATCH (o:Organization) RETURN id(o) AS id, o.name AS name"
            ).data()

        clean_names = []
        for row in rows:
            if row["name"]:
                clean = _clean_name(row["name"])
                self.orgs.append((row["id"], row["name"], clean))
                # Encode the cleaned names, not the raw names!
                clean_names.append(clean)

        # Batch encode ALL database names at once
        if clean_names:
            self.db_embeddings = self.encoder.encode(
                clean_names, normalize_embeddings=True
            )
        print(f"INFO: Loaded {len(self.orgs)} organizations.")

    def get_company_id(
        self, name: str, *, fuzzy_min: int = FUZZY_MIN, embed_min: float = EMBED_MIN
    ) -> int | None:
        if not name or not self.orgs:
            return None

        target_clean = _clean_name(name)

        # 1. EXACT CLEAN MATCH
        exact_match = next(
            (oid for oid, raw, clean in self.orgs if clean == target_clean), None
        )
        if exact_match is not None:
            return exact_match

        # 2. CONTAINMENT MATCH
        containment_candidates = [
            (oid, raw, clean)
            for oid, raw, clean in self.orgs
            if target_clean and clean and target_clean in clean
        ]

        if len(containment_candidates) == 1:
            return containment_candidates[0][0]

        elif len(containment_candidates) > 1:
            best_candidate = None
            best_contain_score = -1

            for oid, raw, clean in containment_candidates:
                score = fuzz.token_sort_ratio(target_clean, clean)
                if score > best_contain_score:
                    best_contain_score, best_candidate = score, oid

            if best_candidate and best_contain_score >= fuzzy_min:
                return best_candidate

        # 3. FUZZY MATCH
        clean_names = [clean for _, _, clean in self.orgs]

        best_match = process.extractOne(
            target_clean,
            clean_names,
            scorer=fuzz.token_set_ratio,
            score_cutoff=fuzzy_min,
        )

        if best_match:
            best_idx = best_match[2]
            return self.orgs[best_idx][0]

        # 4. EMBEDDING MATCH
        if self.db_embeddings is not None:
            # Encode the cleaned target to properly compare against the cleaned db_embeddings
            target_emb = self.encoder.encode(target_clean, normalize_embeddings=True)

            # Vectorized dot product (Cosine similarity since vectors are normalized)
            similarities = np.dot(self.db_embeddings, target_emb)
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            if best_sim >= embed_min:
                return self.orgs[best_idx][0]

        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def load_claim_nodes(
    claim_idx: int, claims_dir: str = "big_dataset"
) -> Optional[List[Dict[str, Any]]]:
    """
    Load nodes with embeddings from claim_{i}.json file.

    Args:
        claim_idx: The claim index
        claims_dir: Directory containing claim_{i}.json files

    Returns:
        List of nodes with embeddings, or None if file not found
    """
    claim_file = Path(claims_dir) / f"claim_{claim_idx}.json"

    if not claim_file.exists():
        print(f"Warning: Claim file not found: {claim_file}")
        return None

    try:
        with open(claim_file, "r") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])

        # Verify embeddings exist
        nodes_with_embeddings = [n for n in nodes if "embedding" in n]

        if len(nodes_with_embeddings) < len(nodes):
            print(
                f"Warning: Only {len(nodes_with_embeddings)}/{len(nodes)} nodes have embeddings in {claim_file.name}"
            )

        return nodes_with_embeddings
    except Exception as e:
        print(f"Error loading {claim_file}: {e}")
        return None


def _safe_json(text: str) -> Dict[str, Any]:
    def _clean_text(s: str) -> str:
        """Remove markdown and extra whitespace."""
        s = s.strip()
        # Remove markdown code blocks
        if s.startswith("```"):
            s = re.sub(r"^```[a-z]*\n?", "", s)
            s = re.sub(r"```$", "", s)
        # Remove <OUTPUT> tags if present
        s = re.sub(r"<OUTPUT>\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*</OUTPUT>", "", s, flags=re.IGNORECASE)
        return s.strip()

    def _extract_cited_nodes(s: str) -> List[int]:
        """Extract cited node IDs as a list of integers."""
        match = re.search(r"Cited Nodes:\s*\[(.*?)\]", s, re.IGNORECASE)
        if not match:
            # Try without brackets just in case
            match = re.search(r"Cited Nodes:\s*([^\n<]+)", s, re.IGNORECASE)

        if match:
            nodes_str = match.group(1).strip()
            if nodes_str.lower() in ["", "none", "n/a", "null"]:
                return []

            # Extract numbers from the string
            ids = re.findall(r"\d+", nodes_str)
            return [int(id_str) for id_str in ids]
        return []

    def _extract_label(s: str) -> str | None:
        """Extract label value."""
        patterns = [
            # Standard format: Label: greenwashing
            r"Label:\s*\*{0,2}(greenwashing|not_greenwashing|abstain)\*{0,2}",
            # With markdown: Label: **greenwashing**
            r"Label:\s*\*\*(greenwashing|not_greenwashing|abstain)\*\*",
            # Standalone with asterisks
            r"\*\*(greenwashing|not_greenwashing|abstain)\*\*",
            # Just the word after "Label:"
            r"Label:\s*(greenwashing|not_greenwashing|abstain)",
            # Quoted
            r'Label:\s*["\']?(greenwashing|not_greenwashing|abstain)["\']?',
        ]

        for pattern in patterns:
            match = re.search(pattern, s, re.IGNORECASE | re.MULTILINE)
            if match:
                label = match.group(1).lower().strip()
                # Normalize variants
                if label in ["greenwashing", "not_greenwashing", "abstain"]:
                    return label

        return None

    def _extract_type(s: str) -> str | None:
        """Extract type value."""
        patterns = [
            # Standard format: Type: Type 1
            r"Type:\s*\*{0,2}(Type\s*[1-4]|N/A)\*{0,2}",
            # With markdown
            r"Type:\s*\*\*(Type\s*[1-4]|N/A)\*\*",
            # Just the type
            r"Type:\s*(Type\s*[1-4]|N/A)",
            # Quoted
            r'Type:\s*["\']?(Type\s*[1-4]|N/A)["\']?',
            # Without "Type" prefix: Type: 1
            r"Type:\s*\*{0,2}([1-4])\*{0,2}(?!\d)",
        ]

        for pattern in patterns:
            match = re.search(pattern, s, re.IGNORECASE | re.MULTILINE)
            if match:
                type_val = match.group(1).strip()
                # Normalize
                if type_val.upper() == "N/A":
                    return "N/A"
                # If just a number, add "Type" prefix
                if type_val.isdigit():
                    return f"Type {type_val}"
                # Normalize spacing: "Type1" -> "Type 1"
                type_val = re.sub(
                    r"Type\s*(\d)", r"Type \1", type_val, flags=re.IGNORECASE
                )
                return type_val

        return None

    def _extract_justification(s: str) -> str:
        """Extract justification text - handles multi-line responses."""

        # Find the start of justification
        match = re.search(r"Justification:\s*", s, re.IGNORECASE | re.MULTILINE)
        if not match:
            return ""

        start_pos = match.end()

        # Extract everything after "Justification:"
        remaining_text = s[start_pos:]

        # Find where it ends (next field label or end of string)
        end_markers = [
            r"\n\s*Label:",
            r"\n\s*Type:",
            r"\n\s*Cited Nodes:",  # <--- ADD THIS!
            r"</OUTPUT>",
            r"</",  # Any closing tag
        ]

        end_pos = len(remaining_text)
        for marker in end_markers:
            marker_match = re.search(marker, remaining_text, re.IGNORECASE)
            if marker_match:
                end_pos = min(end_pos, marker_match.start())

        justification = remaining_text[:end_pos].strip()

        # Clean up
        justification = justification.strip("\"'")
        justification = re.sub(
            r"[ \t]+", " ", justification
        )  # Normalize spaces on same line
        justification = justification.strip()

        return justification if len(justification) > 3 else ""

    def _try_structured_parse(s: str) -> Dict[str, Any] | None:
        """Try to extract all fields in order."""
        result = {}

        # Extract each field
        label = _extract_label(s)
        type_val = _extract_type(s)
        justification = _extract_justification(s)
        cited_nodes = _extract_cited_nodes(s)  # <--- ADD THIS

        # Must have at least label to be valid
        if label:
            result["label"] = label
            result["type"] = type_val if type_val else "N/A"
            result["reasoning"] = justification if justification else ""
            result["cited_node_ids"] = cited_nodes  # <--- ADD THIS
            return result

        return None

    def _try_json_fallback(s: str) -> Dict[str, Any] | None:
        """
        Fallback: try to parse as JSON (in case LLM ignored format instructions).
        """
        try:
            # Remove markdown
            s = re.sub(r"^```json\s*", "", s)
            s = re.sub(r"^```\s*", "", s)
            s = re.sub(r"```$", "", s)

            # Try direct JSON parse
            data = json.loads(s.strip())

            if isinstance(data, dict):
                # Map JSON keys to expected keys
                result = {}

                # Label
                for key in ["label", "Label", "LABEL"]:
                    if key in data:
                        result["label"] = str(data[key]).lower()
                        break

                # Type
                for key in ["type", "Type", "TYPE"]:
                    if key in data:
                        result["type"] = str(data[key])
                        break

                # Justification
                for key in [
                    "justification",
                    "Justification",
                    "JUSTIFICATION",
                    "reasoning",
                ]:
                    if key in data:
                        result["reasoning"] = str(data[key])
                        break

                if "label" in result:
                    result.setdefault("type", "N/A")
                    result.setdefault("reasoning", "")
                    return result

        except (json.JSONDecodeError, ValueError):
            pass

        return None

    # Validate input
    if not text or not isinstance(text, str):
        return {
            "label": "error_parsing",
            "type": "N/A",
            "reasoning": "Empty or invalid response text",
        }

    original_text = text
    text = _clean_text(text)

    # Strategy 1: Structured text parsing (primary)
    result = _try_structured_parse(text)
    if result:
        return result

    # Strategy 2: Try with original text
    result = _try_structured_parse(original_text)
    if result:
        return result

    # Strategy 3: JSON fallback (in case LLM went rogue)
    result = _try_json_fallback(text)
    if result:
        return result

    result = _try_json_fallback(original_text)
    if result:
        return result

    # All strategies failed
    return {
        "label": "error_parsing",
        "type": "N/A",
        "reasoning": f"Could not parse LLM response. Raw: {original_text[:300]}",
    }


def format_entities_as_json(entities: List[Dict[str, Any]], company_name: str) -> str:
    """
    Format entities with their paths as a structured JSON optimized for LLM consumption.
    Handles both path-based evidence (from retrieve_evidence_with_paths) and
    direct connection evidence (from retrieve_evidence).
    """

    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    def get_node_display_name(node_type: str, properties: Dict[str, Any]) -> str:
        """Get a display name using class and first meaningful attribute."""
        # Get first non-internal property value
        for k, v in sorted(properties.items()):
            if (
                k not in ["embedding", "id", "valid_from", "valid_to", "is_current"]
                and v is not None
            ):
                # Truncate long values
                str_v = str(v)
                return f"{node_type}({str_v})"
        return node_type

    def format_path(path_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format path with complete node information."""
        path_nodes = path_data.get("path_nodes", [])
        rel_types = path_data.get("rel_types", [])
        path_length = path_data.get("path_length", 0)

        # Format each node with full properties
        formatted_nodes = []
        for node_data in path_nodes:
            node_props = node_data.get("properties", {})
            node_labels = node_data.get("labels", [])

            # Filter out internal properties
            filtered_props = {}
            for k, v in sorted(node_props.items()):
                if (
                    k not in ["embedding", "id", "valid_from", "valid_to", "is_current"]
                    and v is not None
                ):
                    filtered_props[k] = convert_to_serializable(v)

            formatted_nodes.append(
                {
                    "id": node_data.get("id"),
                    "type": node_labels[0] if node_labels else "Node",
                    "properties": filtered_props,
                }
            )

        # Create a readable summary using class + first attribute
        path_summary_parts = []
        for i, node in enumerate(formatted_nodes):
            node_type = node["type"]
            props = node["properties"]
            display_name = get_node_display_name(node_type, props)
            path_summary_parts.append(display_name)

            if i < len(rel_types):
                path_summary_parts.append(f"-[{rel_types[i]}]->")

        return {
            "summary": " ".join(path_summary_parts),
            "hops": path_length,
            "relationships": rel_types,
            "nodes": formatted_nodes,
        }

    def format_direct_connection(
        entity_label: str,
        entity_props: Dict[str, Any],
        rel_type: str,
        company_name: str,
        node_id: Any,
    ) -> Dict[str, Any]:  # <--- Add node_id arg
        # Filter entity properties
        filtered_entity_props = {}
        for k, v in sorted(entity_props.items()):
            if (
                k not in ["embedding", "id", "valid_from", "valid_to", "is_current"]
                and v is not None
            ):
                filtered_entity_props[k] = convert_to_serializable(v)

        # Create summary
        entity_display = get_node_display_name(entity_label, filtered_entity_props)
        summary = f"Organization({company_name}) -[{rel_type}]-> {entity_display}"

        return {
            "summary": summary,
            "hops": 1,
            "relationships": [rel_type],
            "nodes": [
                {"type": "Organization", "properties": {"name": company_name}},
                {
                    "id": node_id,
                    "type": entity_label,
                    "properties": filtered_entity_props,
                },
            ],
        }

    entities_by_type = {}
    for entity in entities:
        labels = entity.get("labels", [])
        label = labels[0] if labels else "Unknown"

        if label not in entities_by_type:
            entities_by_type[label] = []

        props = entity.get("properties", {})
        similarity = entity.get("similarity", 0.0)
        paths = entity.get("paths", [])  # May be empty or missing
        rel_type = entity.get("rel_type", None)  # For direct connections
        node_id = entity.get("node_id")

        # Filter properties
        filtered_props = {}
        for k, v in sorted(props.items()):
            if (
                k not in ["embedding", "id", "valid_from", "valid_to", "is_current"]
                and v is not None
            ):
                filtered_props[k] = convert_to_serializable(v)

        # CHANGED: Parse all paths into a list instead of just picking one
        connections = []
        if paths:
            # Has path data: format every path returned
            connections = [format_path(p) for p in paths]
        elif rel_type is not None:
            # Has direct connection: format as a single-item list for consistency
            connections = [
                format_direct_connection(
                    label, filtered_props, rel_type, company_name, node_id
                )
            ]

        entity_obj = {
            "properties": filtered_props,
            "connections": connections,  # CHANGED: Plural to reflect the array of paths
        }

        entities_by_type[label].append(entity_obj)

    # Sort entities within each type by properties (for deterministic ordering)
    for label in entities_by_type:
        entities_by_type[label].sort(key=lambda x: str(x["properties"]))

    # Build final structure with clear hierarchy
    formatted_data = {
        "company": company_name,
        "evidence_summary": {
            "total_entities": len(entities),
            "entity_types": {
                label: len(ents) for label, ents in entities_by_type.items()
            },
        },
        "evidence_by_type": entities_by_type,
    }

    return json.dumps(formatted_data, indent=2, ensure_ascii=False)


import numpy as np
from typing import List, Dict, Any

def _build_node_text(node_class: str, node_dict: dict) -> str:
    """Helper to build text representation of a node deterministically."""
    sorted_props = sorted(node_dict.items())
    node_text_parts = [f"{node_class}:"]
    for key, value in sorted_props:
        if value is not None and key not in ["valid_from", "valid_to", "is_current", "embedding"]:
            node_text_parts.append(f"{key}: {value}")
    return " | ".join(node_text_parts)

def retrieve_evidence_with_paths(
    company_id: int,
    claim: str,
    claim_idx: int = 0,
    top_k_per_class: int = 3,
    similarity_threshold: float = 0.3,
    max_hops: int = 3,
    embeddings_dir: str = "big_dataset",
) -> List[Dict[str, Any]]:
    
    print(f"\n=== Retrieving evidence (k={max_hops} hops) for company_id={company_id}, claim_idx={claim_idx} ===")

    json_nodes = load_claim_nodes(claim_idx, embeddings_dir)
    json_nodes = json_nodes or []
    print(f"Loaded {len(json_nodes)} nodes with embeddings from claim_{claim_idx}.json")
    
    # Sort JSON nodes for determinism
    json_nodes = sorted(
        json_nodes,
        key=lambda n: (
            n.get("class", "Unknown"),
            str(n.get("properties", {}).get("name", "")),
            str(n.get("properties", {}).get("description", "")),
            str(n.get("properties", {})),
        ),
    )

    # 1. Identify unique classes to fetch
    unique_json_classes = {
        n.get("class", "Unknown") for n in json_nodes 
        if n.get("class", "Unknown") not in ALWAYS_INCLUDE_CLASSES
    }
    missing_always_include = sorted(list(ALWAYS_INCLUDE_CLASSES))

    # Cache for graph nodes to avoid re-querying and re-encoding
    class_to_graph_nodes = {}
    
    with driver.session() as session:
        # --- STAGE 1: BATCH FETCH & ENCODE NODES ---
        
        # 1A. Fetch and encode for JSON classes (k = max_hops)
        for node_class in unique_json_classes:
            query = f"""
            MATCH (company:Organization)-[*1..{max_hops}]-(node:{node_class})
            WHERE id(company) = $company_id
            RETURN DISTINCT id(node) as node_id, node, labels(node) as labels
            """
            result = session.run(query, company_id=company_id).data()
            if not result:
                class_to_graph_nodes[node_class] = []
                continue
                
            print(f"  Found {len(result)} unique {node_class} nodes, batch encoding...")
            texts_to_encode = [_build_node_text(node_class, dict(r["node"])) for r in result]
            
            # Batch encode all nodes of this class at once
            batch_embeddings = encoder.encode(texts_to_encode, normalize_embeddings=True)
            
            class_to_graph_nodes[node_class] = [
                {
                    "node_id": r["node_id"],
                    "node": r["node"],
                    "labels": r["labels"],
                    "embedding": batch_embeddings[i]
                }
                for i, r in enumerate(result)
            ]

        # 1B. Fetch and encode for ALWAYS_INCLUDE classes (k = 1)
        for class_name in missing_always_include:
            query = f"""
            MATCH (company:Organization)-[*1..1]-(node:{class_name})
            WHERE id(company) = $company_id
            RETURN DISTINCT id(node) as node_id, node, labels(node) as labels
            """
            result = session.run(query, company_id=company_id).data()
            if not result:
                class_to_graph_nodes[class_name] = []
                continue
                
            texts_to_encode = [_build_node_text(class_name, dict(r["node"])) for r in result]
            batch_embeddings = encoder.encode(texts_to_encode, normalize_embeddings=True)
            
            class_to_graph_nodes[class_name] = [
                {
                    "node_id": r["node_id"],
                    "node": r["node"],
                    "labels": r["labels"],
                    "embedding": batch_embeddings[i]
                }
                for i, r in enumerate(result)
            ]

        # --- STAGE 2: CALCULATE SIMILARITIES ---
        target_nodes_map = {}  # Deduplicate target nodes: node_id -> dict
        
        # 2A. Process JSON nodes
        for idx, json_node in enumerate(json_nodes):
            node_class = json_node.get("class", "Unknown")
            if node_class in ALWAYS_INCLUDE_CLASSES: continue
                
            graph_nodes = class_to_graph_nodes.get(node_class, [])
            if not graph_nodes: continue
                
            json_embedding = np.array(json_node["embedding"], dtype=np.float32)
            
            # Vectorized similarity calculation (much faster than looping)
            graph_embeddings = np.array([gn["embedding"] for gn in graph_nodes])
            similarities = np.dot(graph_embeddings, json_embedding) # Assuming embeddings are normalized
            
            # Filter and sort
            valid_indices = np.where(similarities >= similarity_threshold)[0]
            node_similarities = [
                {**graph_nodes[i], "similarity": similarities[i]} 
                for i in valid_indices
            ]
            
            node_similarities.sort(key=lambda x: (-x["similarity"], x["node_id"]))
            for item in node_similarities[:top_k_per_class]:
                # Keep the highest similarity if node is matched by multiple JSON nodes
                if item["node_id"] not in target_nodes_map or item["similarity"] > target_nodes_map[item["node_id"]]["similarity"]:
                    target_nodes_map[item["node_id"]] = {
                        **item,
                        "embedding_source": f"similar to JSON node #{idx+1}"
                    }

        # 2B. Process ALWAYS_INCLUDE classes
        if missing_always_include:
            claim_embedding = np.array(encoder.encode(claim, normalize_embeddings=True), dtype=np.float32)
            
            for class_name in missing_always_include:
                k_limit = 3 if class_name == "KPIObservation" else top_k_per_class
                graph_nodes = class_to_graph_nodes.get(class_name, [])
                if not graph_nodes: continue
                    
                graph_embeddings = np.array([gn["embedding"] for gn in graph_nodes])
                similarities = np.dot(graph_embeddings, claim_embedding)
                
                valid_indices = np.where(similarities >= similarity_threshold)[0]
                node_similarities = [
                    {**graph_nodes[i], "similarity": similarities[i]} 
                    for i in valid_indices
                ]
                
                node_similarities.sort(key=lambda x: (-x["similarity"], x["node_id"]))
                for item in node_similarities[:k_limit]:
                    if item["node_id"] not in target_nodes_map or item["similarity"] > target_nodes_map[item["node_id"]]["similarity"]:
                        target_nodes_map[item["node_id"]] = {
                            **item,
                            "embedding_source": "similar to claim (ALWAYS_INCLUDE)"
                        }

        # --- STAGE 3: BATCH FETCH PATHS ---
        evidence_paths = []
        if target_nodes_map:
            print(f"\nBatch fetching paths for {len(target_nodes_map)} prioritized nodes...")
            
            # Using Cypher CALL {} subquery to limit paths per UNWIND row (Requires Neo4j 4.1+)
            batch_paths_query = f"""
            MATCH (company:Organization) WHERE id(company) = $company_id
            UNWIND $target_ids AS target_id
            MATCH (node) WHERE id(node) = target_id
            CALL {{
                WITH company, node
                MATCH path = (company)-[*1..{max_hops}]-(node)
                RETURN path
                ORDER BY length(path) ASC
                LIMIT 10
            }}
            RETURN 
                target_id,
                [rel in relationships(path) | type(rel)] as rel_types,
                [n in nodes(path) | {{id: id(n), labels: labels(n), properties: properties(n)}}] as path_nodes,
                length(path) as path_length
            """
            
            path_results = session.run(
                batch_paths_query, 
                company_id=company_id, 
                target_ids=list(target_nodes_map.keys())
            ).data()

            # Group paths by target_id
            paths_by_node = {}
            for row in path_results:
                t_id = row["target_id"]
                if t_id not in paths_by_node:
                    paths_by_node[t_id] = []
                paths_by_node[t_id].append({
                    "path_nodes": row["path_nodes"],
                    "rel_types": row["rel_types"],
                    "path_length": row["path_length"],
                })

            # Assemble final payload
            for t_id, t_data in target_nodes_map.items():
                node_paths = paths_by_node.get(t_id, [])
                if node_paths:
                    evidence_paths.append({
                        "node_id": t_id,
                        "labels": sorted(t_data["labels"]),
                        "properties": dict(t_data["node"]),
                        "similarity": t_data["similarity"],
                        "embedding_source": t_data["embedding_source"],
                        "paths": node_paths,
                    })

    print(f"\nTotal evidence nodes retrieved: {len(evidence_paths)}")

    # Final sort for determinism
    evidence_paths.sort(
        key=lambda x: (
            -x["similarity"],
            x["node_id"] if x["node_id"] is not None else 0,
            x["labels"][0] if x["labels"] else "",
        )
    )
    
    return evidence_paths

def retrieve_evidence(
    company_id: int,
    claim: str,
    claim_idx: int = 0,
    top_k_per_class: int = 3,
    similarity_threshold: float = 0.3,
    embeddings_dir: str = "big_dataset",
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence by:
    1. Load nodes with embeddings from claim_{i}.json
    2. For each JSON node, find 10 most similar nodes in Neo4j graph (same class, connected to company)
    3. For ALWAYS_INCLUDE_CLASSES not in JSON, find 10 most similar to the claim in graph
    4. Return full Neo4j nodes with all attributes

    Args:
        company_id: The Neo4j node ID of the company
        claim: The claim text
        year: The year for temporal filtering
        claim_idx: The claim index to load nodes from
        top_k_per_class: Number of most similar nodes to retrieve per JSON node/class
        similarity_threshold: Minimum cosine similarity threshold
        embeddings_dir: Directory containing claim_{i}.json files

    Returns:
        List of evidence dictionaries with full node information from Neo4j (deterministically ordered)
    """
    print(
        f"\n=== Retrieving evidence for company_id={company_id}, claim_idx={claim_idx} ==="
    )

    # Load nodes with embeddings from JSON
    json_nodes = load_claim_nodes(claim_idx, embeddings_dir)

    if not json_nodes:
        print(f"No nodes found in claim_{claim_idx}.json")
        json_nodes = []
    else:
        print(
            f"Loaded {len(json_nodes)} nodes with embeddings from claim_{claim_idx}.json"
        )

        # Sort JSON nodes deterministically by class, then by a stable property
        json_nodes = sorted(
            json_nodes,
            key=lambda n: (
                n.get("class", "Unknown"),
                str(n.get("properties", {}).get("name", "")),
                str(n.get("properties", {}).get("description", "")),
                str(n.get("properties", {})),
            ),
        )

    # Track which classes we've seen in JSON
    json_classes = set()
    evidence = []

    with driver.session() as session:
        # Process each JSON node - find similar nodes in graph
        for idx, json_node in enumerate(json_nodes):
            node_class = json_node.get("class", "Unknown")

            json_classes.add(node_class)

            # Get the embedding from JSON node
            json_embedding = np.array(json_node["embedding"], dtype=np.float32)

            print(
                f"\n[{idx+1}/{len(json_nodes)}] Processing JSON node of class: {node_class}"
            )

            # Query Neo4j: get ALL nodes of this class connected to company
            # ORDER BY for deterministic results
            query = f"""
            MATCH (company:Organization)-[r]-(node:{node_class})
            WHERE id(company) = $company_id
            RETURN 
                id(node) as node_id,
                node,
                type(r) as rel_type,
                labels(node) as labels
            ORDER BY node_id
            """

            try:
                result = session.run(query, company_id=company_id).data()

                if not result:
                    print(
                        f"  No {node_class} nodes found in graph connected to company"
                    )
                    continue

                print(
                    f"  Found {len(result)} {node_class} nodes in graph, calculating similarities..."
                )

                # Calculate similarity between JSON embedding and each graph node
                node_similarities = []

                for graph_node in result:
                    node = graph_node["node"]
                    node_id = graph_node["node_id"]

                    # Create text representation of graph node for embedding
                    # Sort properties for deterministic text representation
                    node_dict = dict(node)
                    sorted_props = sorted(node_dict.items())

                    node_text_parts = [f"{node_class}:"]
                    for key, value in sorted_props:
                        if value is not None and key not in [
                            "valid_from",
                            "valid_to",
                            "is_current",
                            "embedding",
                        ]:
                            node_text_parts.append(f"{key}: {value}")
                    node_text = " | ".join(node_text_parts)

                    # Generate embedding for this graph node
                    graph_node_embedding = encoder.encode(
                        node_text, normalize_embeddings=True
                    )

                    # Calculate similarity with JSON node embedding
                    similarity = cosine_similarity(json_embedding, graph_node_embedding)

                    if similarity >= similarity_threshold:
                        node_similarities.append(
                            {
                                "node_id": node_id,
                                "node": node,
                                "labels": graph_node["labels"],
                                "rel_type": graph_node["rel_type"],
                                "similarity": similarity,
                            }
                        )

                # Stable sort: by similarity (descending), then by node_id (ascending) for determinism
                node_similarities.sort(key=lambda x: (-x["similarity"], x["node_id"]))
                top_nodes = node_similarities[:top_k_per_class]

                print(
                    f"  Selected {len(top_nodes)} most similar nodes (out of {len(node_similarities)} above threshold)"
                )

                # Add to evidence
                for item in top_nodes:
                    evidence_dict = {
                        "node_id": item["node_id"],
                        "labels": sorted(item["labels"]),  # Sort labels for determinism
                        "properties": dict(item["node"]),
                        "rel_type": item["rel_type"],
                        "similarity": item["similarity"],
                        "embedding_source": f"similar to JSON node #{idx+1}",
                    }
                    evidence.append(evidence_dict)

                    node_name = dict(item["node"]).get(
                        "name",
                        dict(item["node"]).get(
                            "description", dict(item["node"]).get("target", "unnamed")
                        ),
                    )
                    print(
                        f"    {node_class}: {str(node_name)[:50]}... (similarity={item['similarity']:.3f})"
                    )

            except Exception as e:
                print(f"  Error querying Neo4j for {node_class}: {e}")
                continue

        # Handle ALWAYS_INCLUDE_CLASSES not in JSON
        # Sort for deterministic processing order
        missing_always_include = sorted(
            [cls for cls in ALWAYS_INCLUDE_CLASSES]
        )

        if missing_always_include:
            print(
                f"\n=== Processing ALWAYS_INCLUDE_CLASSES not in JSON: {missing_always_include} ==="
            )

            # Generate claim embedding for these classes
            claim_embedding = encoder.encode(claim, normalize_embeddings=True)
            temp = top_k_per_class
            for class_name in missing_always_include:
                if class_name == "KPIObservation":
                    top_k_per_class = 3
                else:
                    top_k_per_class = temp

                print(f"\nProcessing ALWAYS_INCLUDE class: {class_name}")

                # Query Neo4j for all nodes of this class connected to company
                # ORDER BY for deterministic results
                query = f"""
                MATCH (company:Organization)-[r]-(node:{class_name})
                WHERE id(company) = $company_id
                RETURN 
                    id(node) as node_id,
                    node,
                    type(r) as rel_type,
                    labels(node) as labels
                ORDER BY node_id
                """

                try:
                    result = session.run(query, company_id=company_id).data()

                    if not result:
                        print(f"  No {class_name} nodes found in graph")
                        continue

                    print(f"  Found {len(result)} {class_name} nodes in graph")

                    # Calculate similarity with claim
                    node_similarities = []

                    for graph_node in result:
                        node = graph_node["node"]
                        node_id = graph_node["node_id"]

                        # Create text representation with sorted properties
                        node_dict = dict(node)
                        sorted_props = sorted(node_dict.items())

                        node_text_parts = [f"{class_name}:"]
                        for key, value in sorted_props:
                            if value is not None and key not in [
                                "valid_from",
                                "valid_to",
                                "is_current",
                                "embedding",
                            ]:
                                node_text_parts.append(f"{key}: {value}")
                        node_text = " | ".join(node_text_parts)

                        # Generate embedding
                        graph_node_embedding = encoder.encode(
                            node_text, normalize_embeddings=True
                        )

                        # Calculate similarity with CLAIM
                        similarity = cosine_similarity(
                            claim_embedding, graph_node_embedding
                        )

                        if similarity >= similarity_threshold:
                            node_similarities.append(
                                {
                                    "node_id": node_id,
                                    "node": node,
                                    "labels": graph_node["labels"],
                                    "rel_type": graph_node["rel_type"],
                                    "similarity": similarity,
                                }
                            )

                    # Stable sort: by similarity (descending), then by node_id (ascending)
                    node_similarities.sort(
                        key=lambda x: (-x["similarity"], x["node_id"])
                    )
                    top_nodes = node_similarities[:top_k_per_class]

                    print(
                        f"  Selected {len(top_nodes)} most similar to claim (out of {len(node_similarities)} above threshold)"
                    )

                    # Add to evidence
                    for item in top_nodes:
                        evidence_dict = {
                            "node_id": item["node_id"],
                            "labels": sorted(
                                item["labels"]
                            ),  # Sort labels for determinism
                            "properties": dict(item["node"]),
                            "rel_type": item["rel_type"],
                            "similarity": item["similarity"],
                            "embedding_source": "similar to claim (ALWAYS_INCLUDE)",
                        }
                        evidence.append(evidence_dict)

                        node_name = dict(item["node"]).get(
                            "name",
                            dict(item["node"]).get(
                                "description",
                                dict(item["node"]).get("target", "unnamed"),
                            ),
                        )
                        print(
                            f"    {class_name}: {str(node_name)[:50]}... (similarity={item['similarity']:.3f})"
                        )

                except Exception as e:
                    print(f"  Error querying Neo4j for {class_name}: {e}")
                    continue

    print(f"\nTotal evidence nodes retrieved from Neo4j: {len(evidence)}")

    # Final sort of evidence for deterministic output
    # Sort by: similarity (desc), then node_id (asc), then class (asc)
    evidence.sort(
        key=lambda x: (
            -x["similarity"],
            x["node_id"] if x["node_id"] is not None else 0,
            x["labels"][0] if x["labels"] else "",
        )
    )
    top_k_per_class = temp
    return evidence

def classify_few(
    evidence, company: str, claim: str
) -> Tuple[Dict[str, Any], str, str, Dict[int, float]]:
    """
    Classify a claim using the LLM with retrieved evidence.
    Returns: (parsed_json, context_string, raw_response, node_similarities_dict)
    """
    # 1. Extract similarities safely BEFORE formatting the LLM context
    node_similarities = {}
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if node_id is not None:
                    # CAST TO NATIVE FLOAT HERE to prevent JSON errors later
                    node_similarities[node_id] = float(item.get("similarity", 0.0))

    # Check what type of evidence we received
    if isinstance(evidence, str):
        # It's already a stringified JSON from the counterfactual script
        ctx = evidence
    elif isinstance(evidence, dict):
        # It's a raw dictionary from the counterfactual script
        ctx = json.dumps(evidence, indent=2)
    else:
        # It's raw Neo4j records from the baseline RAG pipeline
        ctx = (
            format_entities_as_json(evidence, company)
            if evidence
            else json.dumps({"company": company, "entities": []}, indent=2)
        )

    prompt = FEW_TMPL.format(company=company, claim=claim, context=ctx)

    # FIXED: Format for Gemma 3 Multimodal chat template
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that classifies claims.",
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    try:
        # Use processor, not tokenizer
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2200,
                do_sample=False,
                temperature=0.0,
            )

        # FIXED: Decode only the NEW tokens using the processor
        generation = generation[0][input_len:]
        response = processor.decode(generation, skip_special_tokens=True)

        # Return the similarities as the 4th value
        return _safe_json(response), ctx, response, node_similarities

    except Exception as e:
        print(f"Error calling {MODEL_NAME}: {e}")
        return {"label": "error", "reasoning": str(e)}, ctx, "", node_similarities
