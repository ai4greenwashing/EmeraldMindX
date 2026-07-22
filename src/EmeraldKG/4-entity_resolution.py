#!/usr/bin/env python3

import json
import pathlib
import argparse
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import re
import requests
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Entity classes that need deduplication
ENTITIES_NEEDING_RESOLUTION = {
    "Organization",
    "Person", 
    "Facility",
    "Product",
    "Material",
    "Location",
    "Community",
    "Authority",
    "Country",
    "Regulation",
    "Initiative",
    "ClaimKeyword",
    "Standard",
    "Certification",
    "Project",
    "Goal"
}

# Observation classes that are inherently unique (no deduplication needed)
OBSERVATION_CLASSES = {
    "KPIObservation",
    "SustainabilityClaim",
    "ThirdPartyVerification",
    "Controversy",
    "Penalty",
    "MediaReport",
    "Investment",
    "CarbonOffsetProject",
    "ScienceBasedTarget",
    "Emission",
    "Waste"
}


@dataclass
class EntityCluster:
    entity_class: str
    canonical_idx: int  
    instances: List[int] = field(default_factory=list)  
    non_temporal_signature: str = "" 


@dataclass
class ResolutionStats:
    total_entities: int = 0
    clusters_formed: int = 0
    llm_comparisons: int = 0
    llm_matches: int = 0
    temporal_versions_preserved: int = 0
    entities_by_class: Dict[str, int] = field(default_factory=dict)
    clusters_by_class: Dict[str, int] = field(default_factory=dict)
    blocks_by_class: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_entities": self.total_entities,
                "clusters_formed": self.clusters_formed,
                "entities_merged": self.total_entities - self.clusters_formed,
                "reduction_percentage": ((self.total_entities - self.clusters_formed) / self.total_entities * 100) if self.total_entities > 0 else 0,
                "temporal_versions_preserved": self.temporal_versions_preserved
            },
            "llm_matching": {
                "total_comparisons": self.llm_comparisons,
                "matches_found": self.llm_matches,
                "match_rate": (self.llm_matches / self.llm_comparisons * 100) if self.llm_comparisons > 0 else 0
            },
            "by_entity_class": {
                entity_class: {
                    "entities": self.entities_by_class.get(entity_class, 0),
                    "clusters": self.clusters_by_class.get(entity_class, 0),
                    "blocks": self.blocks_by_class.get(entity_class, 0),
                    "reduction": self.entities_by_class.get(entity_class, 0) - self.clusters_by_class.get(entity_class, 0)
                }
                for entity_class in set(list(self.entities_by_class.keys()) + list(self.clusters_by_class.keys()))
            }
        }


def load_schema_sets(schema: Dict) -> Tuple[Set[str], Set[str], Dict[str, List[Tuple[str, str]]]]:
    entity_classes = {node["class"] for node in schema.get("nodes", [])}
    edge_labels = {edge["label"] for edge in schema.get("edges", [])}
    
    edge_directions = {}
    for edge in schema.get("edges", []):
        label = edge["label"]
        from_class = edge.get("source_class")
        to_class = edge.get("target_class")
        if label not in edge_directions:
            edge_directions[label] = []
        if from_class and to_class:
            edge_directions[label].append((from_class, to_class))
    
    return entity_classes, edge_labels, edge_directions


def validate_triple(triple: Dict, entity_classes: Set[str], edge_labels: Set[str], 
                     edge_directions: Dict[str, List[Tuple[str, str]]]) -> Tuple[bool, List[str]]:
    errors = []
    
    if not isinstance(triple, dict):
        return False, ["Not a dict"]
    
    if not {"subject", "predicate", "object"}.issubset(triple.keys()):
        return False, ["Missing required keys"]
    
    subj = triple.get("subject")
    if not isinstance(subj, dict):
        errors.append("Subject not a dict")
    elif "class" not in subj or "properties" not in subj:
        errors.append("Subject missing class or properties")
    elif subj["class"] not in entity_classes:
        errors.append(f"Invalid subject class: {subj.get('class')}")
    else:
        props = subj.get("properties", {})
        if "valid_from" not in props:
            errors.append("Subject missing valid_from")
        if "valid_to" not in props:
            errors.append("Subject missing valid_to")
        if "is_current" not in props:
            errors.append("Subject missing is_current")
    
    obj = triple.get("object")
    if not isinstance(obj, dict):
        errors.append("Object not a dict")
    elif "class" not in obj or "properties" not in obj:
        errors.append("Object missing class or properties")
    elif obj["class"] not in entity_classes:
        errors.append(f"Invalid object class: {obj.get('class')}")
    else:
        props = obj.get("properties", {})
        if "valid_from" not in props:
            errors.append("Object missing valid_from")
        if "valid_to" not in props:
            errors.append("Object missing valid_to")
        if "is_current" not in props:
            errors.append("Object missing is_current")
    
    pred = triple.get("predicate")
    if not isinstance(pred, str):
        errors.append("Predicate not a string")
    elif pred not in edge_labels:
        errors.append(f"Invalid predicate: {pred}")
    else:
        if pred in edge_directions and subj.get("class") and obj.get("class"):
            valid_directions = edge_directions[pred]
            direction_ok = any(
                (from_cls == subj["class"] and to_cls == obj["class"])
                for from_cls, to_cls in valid_directions
            )
            if not direction_ok:
                errors.append(f"Invalid direction: {subj['class']} -{pred}-> {obj['class']}")
    
    if "temporal_metadata" not in triple:
        errors.append("Missing temporal_metadata")
    else:
        tm = triple["temporal_metadata"]
        if not isinstance(tm, dict):
            errors.append("temporal_metadata not a dict")
        else:
            for key in ["valid_from", "valid_to", "recorded_at"]:
                if key not in tm:
                    errors.append(f"temporal_metadata missing {key}")
    
    return len(errors) == 0, errors


def triples_to_graph(triples: List[Dict]) -> Dict[str, Any]:
    nodes = []
    edges = []
    node_map = {}  # (class, properties_tuple) -> node_index
    
    for triple in triples:
        subj = triple["subject"]
        obj = triple["object"]
        pred = triple["predicate"]
        
        subj_props = tuple(sorted((k, str(v)) for k, v in subj["properties"].items()))
        subj_key = (subj["class"], subj_props)
        if subj_key not in node_map:
            node_map[subj_key] = len(nodes)
            nodes.append(subj)
        
        subj_idx = node_map[subj_key]
       
        obj_props = tuple(sorted((k, str(v)) for k, v in obj["properties"].items()))
        obj_key = (obj["class"], obj_props)
        if obj_key not in node_map:
            node_map[obj_key] = len(nodes)
            nodes.append(obj)
        obj_idx = node_map[obj_key]
        
        # Create edge
        edge = {
            "subject": subj_idx,
            "predicate": pred,
            "object": obj_idx
        }
        
        if "temporal_metadata" in triple:
            edge["temporal_metadata"] = triple["temporal_metadata"]
        
        edges.append(edge)
    
    return {"nodes": nodes, "edges": edges}


def get_non_temporal_signature(node: Dict[str, Any]) -> str:
    props = node.get("properties", {})
    non_temporal = {
        k: v for k, v in props.items() 
        if k not in ["valid_from", "valid_to", "is_current", "recorded_at"]
    }
    signature_parts = [f"{k}:{v}" for k, v in sorted(non_temporal.items()) if v is not None]
    return "|".join(signature_parts)


def get_embedding_text(node: Dict[str, Any]) -> str:
    entity_class = node["class"]
    props = node.get("properties", {})
    parts = [f"Type: {entity_class}"]
    temporal_fields = {"valid_from", "valid_to", "is_current", "recorded_at"}
    
    for key, value in sorted(props.items()):
        if key not in temporal_fields and value is not None and value != "":
            parts.append(f"{key}: {value}")
    return ". ".join(parts)


_embedding_cache: Dict[str, np.ndarray] = {}

def get_embedding(text: str, use_cache: bool = True) -> Optional[np.ndarray]:
    if use_cache and text in _embedding_cache:
        return _embedding_cache[text]
    
    try:
        resp = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text:latest", "input": text},
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        
        embeddings = result.get("embeddings") or result.get("embedding")
        if embeddings and len(embeddings) > 0:
            embedding = np.array(embeddings[0])
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            if use_cache:
                _embedding_cache[text] = embedding
            
            return embedding
        else:
            logger.warning("No embeddings in response")
            return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Embedding API call failed: {e}")
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product > 0 else 0.0


def embedding_based_blocking(nodes: List[Dict[str, Any]], 
                             similarity_threshold: float = 0.85) -> Dict[str, List[List[int]]]:
    logger.info(f"Starting embedding-based blocking with threshold {similarity_threshold}")
    
    by_class = defaultdict(list)
    for idx, node in enumerate(nodes):
        entity_class = node["class"]
        if entity_class in ENTITIES_NEEDING_RESOLUTION:
            by_class[entity_class].append(idx)
    
    blocks = {}
    
    for entity_class, indices in by_class.items():
        logger.info(f"  Processing {entity_class}: {len(indices)} entities")
        
        if len(indices) == 0:
            continue
        embeddings = []
        valid_indices = []
        
        for idx in indices:
            text = get_embedding_text(nodes[idx])
            emb = get_embedding(text)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)
            else:
                logger.warning(f"Failed to get embedding for node {idx}")
        
        if not embeddings:
            logger.warning(f"No embeddings generated for {entity_class}")
            continue
        embeddings = np.array(embeddings)
        class_blocks = []
        assigned = set()
        
        for i, idx in enumerate(valid_indices):
            if idx in assigned:
                continue
            block = [idx]
            assigned.add(idx)
            for j, other_idx in enumerate(valid_indices):
                if other_idx in assigned:
                    continue
                
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= similarity_threshold:
                    block.append(other_idx)
                    assigned.add(other_idx)
            
            class_blocks.append(block)
        
        blocks[entity_class] = class_blocks
        logger.info(f"    Created {len(class_blocks)} blocks for {entity_class}")
    
    return blocks


def call_ollama_for_match(node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
    props1 = {k: v for k, v in node1.get("properties", {}).items() 
              if k not in ["valid_from", "valid_to", "is_current", "recorded_at"]}
    props2 = {k: v for k, v in node2.get("properties", {}).items() 
              if k not in ["valid_from", "valid_to", "is_current", "recorded_at"]}
    
    temp1 = {k: v for k, v in node1.get("properties", {}).items() 
             if k in ["valid_from", "valid_to", "is_current"]}
    temp2 = {k: v for k, v in node2.get("properties", {}).items() 
             if k in ["valid_from", "valid_to", "is_current"]}
    
    prompt = f"""You are an entity resolution expert for temporal knowledge graphs. Determine if these two entities refer to the same real-world entity.

Entity 1:
Type: {node1['class']}
Attributes: {json.dumps(props1, indent=2)}
Temporal info: {json.dumps(temp1, indent=2)}

Entity 2:
Type: {node2['class']}
Attributes: {json.dumps(props2, indent=2)}
Temporal info: {json.dumps(temp2, indent=2)}

IMPORTANT: Two entities are the SAME if they represent:
1. The exact same real-world entity (e.g., same organization, same person, same location)
2. Even if they have different temporal validity periods (they might be different versions over time)

Two entities are DIFFERENT if:
1. They represent fundamentally different real-world entities
2. Names are similar but refer to different entities (e.g., "Paris, France" vs "Paris, Texas")

Consider:
- Are the identifying attributes (name, location, etc.) referring to the same entity?
- Could these be temporal versions of the same entity?
- Or are these completely different entities that happen to have similar names?

Answer with ONLY "yes" if they are the same real-world entity (possibly different temporal versions), or "no" if they are different entities."""
    
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:latest", "prompt": prompt, "stream": False},
            timeout=60
        )
        resp.raise_for_status()
        response = resp.json().get("response", "").strip().lower()
        
        is_match = "yes" in response and "no" not in response
        return is_match
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama call failed: {e}, defaulting to no match")
        return False


def resolve_entities_in_block(block_indices: List[int], 
                              nodes: List[Dict[str, Any]],
                              stats: ResolutionStats) -> List[EntityCluster]:
    if len(block_indices) <= 1:
        if len(block_indices) == 1:
            cluster = EntityCluster(
                entity_class=nodes[block_indices[0]]["class"],
                canonical_idx=block_indices[0],
                instances=[block_indices[0]],
                non_temporal_signature=get_non_temporal_signature(nodes[block_indices[0]])
            )
            return [cluster]
        return []
    
    clusters = []
    assigned = set()
    
    for i, idx1 in enumerate(block_indices):
        if idx1 in assigned:
            continue
        
        node1 = nodes[idx1]
        cluster_indices = [idx1]
        assigned.add(idx1)
        
        for idx2 in block_indices[i+1:]:
            if idx2 in assigned:
                continue
            node2 = nodes[idx2]
            stats.llm_comparisons += 1
            
            if call_ollama_for_match(node1, node2):
                cluster_indices.append(idx2)
                assigned.add(idx2)
                stats.llm_matches += 1
        
        cluster = EntityCluster(
            entity_class=node1["class"],
            # Use first as canonical
            canonical_idx=cluster_indices[0],  
            instances=cluster_indices,
            non_temporal_signature=get_non_temporal_signature(node1)
        )
        
        clusters.append(cluster)
        
        if len(cluster_indices) > 1:
            stats.temporal_versions_preserved += len(cluster_indices)
            logger.debug(f"Cluster formed: {len(cluster_indices)} versions of {node1['class']}")
    
    return clusters


def create_resolved_graph(original_graph: Dict[str, Any], 
                         clusters: List[EntityCluster],
                         observation_indices: List[int]) -> Dict[str, Any]:
    nodes = original_graph["nodes"]
    edges = original_graph["edges"]
    new_nodes = []
    old_to_new_mapping = {}
    for obs_idx in observation_indices:
        old_to_new_mapping[obs_idx] = len(new_nodes)
        new_nodes.append(nodes[obs_idx])
    for cluster in clusters:
        versions = [nodes[idx] for idx in cluster.instances]
        
        if len(versions) == 1:
            old_to_new_mapping[cluster.instances[0]] = len(new_nodes)
            new_nodes.append(versions[0])
        else:
            consolidated = {
                "class": cluster.entity_class,
                "properties": {},
                "temporal_versions": []
            }
            
            all_props = {}
            temporal_fields = {"valid_from", "valid_to", "is_current"}
            
            for version in versions:
                for key, value in version["properties"].items():
                    if key not in temporal_fields:
                        if key not in all_props and value is not None and value != "":
                            all_props[key] = value
            
            consolidated["properties"] = all_props
            
            for version in versions:
                temp_version = {
                    "valid_from": version["properties"].get("valid_from"),
                    "valid_to": version["properties"].get("valid_to"),
                    "is_current": version["properties"].get("is_current"),
                    "properties": version["properties"]  # Full properties for this version
                }
                consolidated["temporal_versions"].append(temp_version)
            
            new_idx = len(new_nodes)
            for old_idx in cluster.instances:
                old_to_new_mapping[old_idx] = new_idx
            
            new_nodes.append(consolidated)
    
    edge_set = set()
    new_edges = []
    
    for edge in edges:
        old_subj = edge["subject"]
        old_obj = edge["object"]
        
        if old_subj not in old_to_new_mapping or old_obj not in old_to_new_mapping:
            logger.warning(f"Edge references unmapped node: {old_subj} -> {old_obj}")
            continue
        
        new_subj = old_to_new_mapping[old_subj]
        new_obj = old_to_new_mapping[old_obj]
        
        edge_key = (new_subj, edge["predicate"], new_obj)
        
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            new_edge = {
                "subject": new_subj,
                "predicate": edge["predicate"],
                "object": new_obj
            }
            
            if "temporal_metadata" in edge:
                new_edge["temporal_metadata"] = edge["temporal_metadata"]
            
            new_edges.append(new_edge)
    
    return {"nodes": new_nodes, "edges": new_edges}


def resolve_entities(input_file: pathlib.Path,
                    schema_file: pathlib.Path,
                    output_file: pathlib.Path,
                    similarity_threshold: float = 0.85):
    """
    Main entity resolution pipeline:
    1. Load and validate triples
    2. Convert to graph format
    3. Embedding-based blocking
    4. LLM-based entity resolution within blocks
    5. Create resolved graph with temporal versions preserved
    """
    logger.info(f"Loading schema from {schema_file}")
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    entity_classes, edge_labels, edge_directions = load_schema_sets(schema)
    logger.info(f"Loading triples from {input_file}")
    triples = json.loads(input_file.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(triples)} triples")
    logger.info("Validating triples...")
    valid_triples = []
    invalid_count = 0
    
    for triple in triples:
        is_valid, errors = validate_triple(triple, entity_classes, edge_labels, edge_directions)
        if is_valid:
            valid_triples.append(triple)
        else:
            invalid_count += 1
    
    logger.info(f"Validation complete: {len(valid_triples)} valid, {invalid_count} invalid")
    
    if not valid_triples:
        logger.error("No valid triples to process")
        return
    
    logger.info("Converting triples to graph format...")
    graph = triples_to_graph(valid_triples)
    logger.info(f"Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    
    stats = ResolutionStats(total_entities=0)
    
    entity_indices = []
    observation_indices = []
    
    for idx, node in enumerate(graph["nodes"]):
        if node["class"] in ENTITIES_NEEDING_RESOLUTION:
            entity_indices.append(idx)
            entity_class = node["class"]
            stats.entities_by_class[entity_class] = stats.entities_by_class.get(entity_class, 0) + 1
        else:
            observation_indices.append(idx)
    
    stats.total_entities = len(entity_indices)
    logger.info(f"Entities: {len(entity_indices)}, Observations: {len(observation_indices)}")
    
    logger.info("\n=== Stage 1: Embedding-Based Blocking ===")
    blocks = embedding_based_blocking(graph["nodes"], similarity_threshold)
    
    for entity_class, block_list in blocks.items():
        stats.blocks_by_class[entity_class] = len(block_list)
    total_blocks = sum(len(block_list) for block_list in blocks.values())
    logger.info(f"Created {total_blocks} blocks across {len(blocks)} entity classes")
    
    logger.info("\n=== Stage 2: LLM-Based Entity Resolution ===")
    all_clusters = []
    for entity_class, block_list in blocks.items():
        logger.info(f"\nResolving {entity_class}: {len(block_list)} blocks")
        class_clusters = 0
        for block_idx, block_indices in enumerate(block_list):
            if len(block_indices) > 1:
                logger.info(f"  Block {block_idx + 1}/{len(block_list)}: {len(block_indices)} entities")
                clusters = resolve_entities_in_block(block_indices, graph["nodes"], stats)
                all_clusters.extend(clusters)
                stats.clusters_formed += len(clusters)
                class_clusters += len(clusters)
            else:
                cluster = EntityCluster(
                    entity_class=entity_class,
                    canonical_idx=block_indices[0],
                    instances=block_indices,
                    non_temporal_signature=get_non_temporal_signature(graph["nodes"][block_indices[0]])
                )
                all_clusters.append(cluster)
                stats.clusters_formed += 1
                class_clusters += 1
        stats.clusters_by_class[entity_class] = class_clusters
    
    logger.info(f"\nResolution complete: {stats.clusters_formed} clusters formed")
    logger.info(f"LLM comparisons: {stats.llm_comparisons}, matches: {stats.llm_matches}")
    logger.info(f"Temporal versions preserved: {stats.temporal_versions_preserved}")
    
    logger.info("\n=== Stage 3: Creating Resolved Graph ===")
    resolved_graph = create_resolved_graph(graph, all_clusters, observation_indices)
    logger.info(
        f"Resolved graph: {len(graph['nodes'])} → {len(resolved_graph['nodes'])} nodes, "
        f"{len(graph['edges'])} → {len(resolved_graph['edges'])} edges"
    )
    
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(
        json.dumps(resolved_graph, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"\nSaved resolved graph to {output_file}")
    
    stats_file = output_file.parent / f"{output_file.stem}_stats.json"
    stats_data = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "parameters": {
            "similarity_threshold": similarity_threshold
        },
        "input_statistics": {
            "total_triples": len(triples),
            "valid_triples": len(valid_triples),
            "invalid_triples": invalid_count,
            "original_nodes": len(graph['nodes']),
            "original_edges": len(graph['edges']),
            "entity_nodes": len(entity_indices),
            "observation_nodes": len(observation_indices)
        },
        "output_statistics": {
            "resolved_nodes": len(resolved_graph['nodes']),
            "resolved_edges": len(resolved_graph['edges'])
        },
        "resolution_statistics": stats.to_dict()
    }
    
    stats_file.write_text(
        json.dumps(stats_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"Saved statistics to {stats_file}")
    
    logger.info(
        f"\n{'='*60}\n"
        f"ENTITY RESOLUTION SUMMARY\n"
        f"{'='*60}\n"
        f"Input: {len(triples)} triples\n"
        f"Valid: {len(valid_triples)} triples\n"
        f"Original graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges\n"
        f"Resolved graph: {len(resolved_graph['nodes'])} nodes, {len(resolved_graph['edges'])} edges\n"
        f"Entities resolved: {len(entity_indices)} → {stats.clusters_formed} clusters\n"
        f"Reduction: {len(entity_indices) - stats.clusters_formed} entities merged\n"
        f"Temporal versions preserved: {stats.temporal_versions_preserved}\n"
        f"LLM comparisons: {stats.llm_comparisons} (matches: {stats.llm_matches})\n"
        f"{'='*60}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Entity resolution using embedding-based blocking and LLM matching"
    )
    parser.add_argument(
        "--input_file",
        type=pathlib.Path,
        required=True,
        help="Path to all_validated_triples.json"
    )
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        required=True,
        help="Path to graph_schema.json"
    )
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        required=True,
        help="Output path for resolved graph"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Embedding similarity threshold for blocking (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    resolve_entities(
        args.input_file,
        args.schema,
        args.output_file,
        args.similarity_threshold
    )


if __name__ == "__main__":
    main()