#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import islice

from neo4j import GraphDatabase, Driver
from collections import defaultdict

MERGE_BY_IDENTITY_CLASSES = [
    "Organization",
]

_RE_NON_WORD = re.compile(r"[^\w]")

def _cypher_safe(s: str) -> str:
    return _RE_NON_WORD.sub("_", s)

def _sanitize_props(d: dict) -> dict:
    return {_cypher_safe(k): v for k, v in d.items()}

def _normalize_props(props: dict) -> dict:
    normalized = {}
    for k, v in props.items():
        if v is None:
            normalized[k] = ""
        elif isinstance(v, dict):
            normalized[k] = json.dumps(v)
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                normalized[k] = json.dumps(v)
            else:
                normalized[k] = v
        else:
            normalized[k] = v
    return normalized

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def _json_dumps_deterministic(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _canonicalize_value(v):
    if v is None:
        return ""
    if isinstance(v, dict):
        return _json_dumps_deterministic(v)
    if isinstance(v, list):
        if not v:
            return []
        if all(isinstance(x, dict) for x in v):
            canon = [_json_dumps_deterministic(x) for x in v]
            canon.sort()
            return canon
        return v
    return v

def _canonicalize_props(props: dict) -> dict:
    canon = {}
    for k, v in props.items():
        safe_k = _cypher_safe(k)
        canon[safe_k] = _canonicalize_value(v)
    return canon


def load_schema(schema_path: Path) -> dict:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    node_index = {}
    for node_def in schema["nodes"]:
        node_index[node_def["class"]] = {
            "properties": set(node_def.get("properties", [])),
            "identity_keys": node_def.get("identity_keys", [])
        }
    edge_index = {}
    for edge_def in schema["edges"]:
        label = edge_def["label"]
        if label not in edge_index:
            edge_index[label] = []
        edge_index[label].append({
            "source_class": edge_def["source_class"],
            "target_class": edge_def["target_class"]
        })
    
    return {
        "nodes": node_index,
        "edges": edge_index,
        "raw": schema
    }

def generate_node_key(node_class: str, properties: dict, schema: dict) -> str:
    node_schema = schema["nodes"].get(node_class)
    
    if node_class in MERGE_BY_IDENTITY_CLASSES and node_schema:
        identity_keys = node_schema["identity_keys"]
        
        if identity_keys:
            key_parts = []
            for key in identity_keys:
                safe_key = _cypher_safe(key)
                value = properties.get(safe_key)
                if value not in (None, ""):
                    if isinstance(value, (list, dict)):
                        key_parts.append(f"{safe_key}={_json_dumps_deterministic(value)}")
                    else:
                        key_parts.append(f"{safe_key}={value}")
            
            if key_parts:
                return f"{node_class}::{':'.join(key_parts)}"
    
    prop_str = _json_dumps_deterministic(properties)
    return f"{node_class}::{prop_str}"

def ingest_nodes_batch_unwind(nodes: Dict[str, dict], driver: Driver, batch_size: int = 5000):
    by_label = defaultdict(list)
    for node in nodes.values():
        label = node["class"]
        props = _normalize_props(node["properties"])
        props = _sanitize_props(props)
        props["_node_key"] = node["key"]
        by_label[label].append(props)

    total = sum(len(v) for v in by_label.values())
    processed = 0
    print(f"Ingesting {total} nodes grouped by label...")

    with driver.session() as session:
        for label, rows in by_label.items():
            # chunk rows
            for i in range(0, len(rows), batch_size):
                batch_rows = rows[i : i + batch_size]
                cypher = (
                    f"UNWIND $rows AS r\n"
                    f"MERGE (n:`{label}` {{_node_key: r._node_key}})\n"
                    f"SET n += r"
                )
                # execute in a single write transaction
                def _tx_work(tx):
                    tx.run(cypher, rows=batch_rows)
                session.execute_write(_tx_work)

                processed += len(batch_rows)
                print(f"  Progress: {processed}/{total} nodes", end="\r")

    print(f"\nCreated/merged {processed} nodes")
    return processed


def validate_node(node: dict, schema: dict) -> Tuple[bool, str]:
    # Validate a node against schema
    node_class = node.get("class")
    
    if not node_class:
        return False, "Missing 'class' field"
    
    if node_class not in schema["nodes"]:
        return False, f"Unknown node class '{node_class}'"
    
    properties = node.get("properties", {})
    if not isinstance(properties, dict):
        return False, "Properties must be a dictionary"
    
    return True, ""

def validate_edge(edge: dict, schema: dict) -> Tuple[bool, str]:
    # Validate an edge against schema
    predicate = edge.get("predicate")
    
    if not predicate:
        return False, "Missing 'predicate' field"
    
    subject = edge.get("subject", {})
    obj = edge.get("object", {})
    
    subj_class = subject.get("class")
    obj_class = obj.get("class")
    
    if not subj_class or not obj_class:
        return False, "Subject or object missing 'class' field"
    
    if predicate not in schema["edges"]:
        return False, f"Unknown edge predicate '{predicate}'"
    
    valid_combinations = schema["edges"][predicate]
    is_valid = any(
        combo["source_class"] == subj_class and combo["target_class"] == obj_class
        for combo in valid_combinations
    )
    
    if not is_valid:
        return False, f"Invalid edge: {subj_class} -[{predicate}]-> {obj_class}"
    
    return True, ""


def validate_and_extract(edge_list: List[dict], schema: dict) -> Tuple[Dict[str, dict], List[dict], List[str]]:
    nodes = {}
    edges = []
    errors = []
    
    for i, item in enumerate(edge_list):
        subj = item.get("subject", {})
        is_valid, error = validate_node(subj, schema)
        if not is_valid:
            errors.append(f"Edge {i} subject: {error}")
            continue
        
        obj = item.get("object", {})
        is_valid, error = validate_node(obj, schema)
        if not is_valid:
            errors.append(f"Edge {i} object: {error}")
            continue
        
        is_valid, error = validate_edge(item, schema)
        if not is_valid:
            errors.append(f"Edge {i}: {error}")
            continue
        
        subj_class = subj.get("class")
        subj_props_raw = subj.get("properties", {})
        subj_props = _canonicalize_props(subj_props_raw)
        subj_key = generate_node_key(subj_class, subj_props, schema)
        
        if subj_key not in nodes:
            nodes[subj_key] = {
                "class": subj_class,
                "properties": subj_props.copy(),
                "key": subj_key
            }
        
        obj_class = obj.get("class")
        obj_props_raw = obj.get("properties", {})
        obj_props = _canonicalize_props(obj_props_raw)
        obj_key = generate_node_key(obj_class, obj_props, schema)
        
        if obj_key not in nodes:
            nodes[obj_key] = {
                "class": obj_class,
                "properties": obj_props.copy(),
                "key": obj_key
            }
        
        edges.append({
            "subject_key": subj_key,
            "predicate": item.get("predicate"),
            "object_key": obj_key
        })
    
    return nodes, edges, errors


# database operations 
def clear_database(driver: Driver):
    """Clear all nodes and relationships from database."""
    print("Clearing database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared")

def setup_indexes(driver: Driver, schema: dict):
    print("\nCreating indexes...")
    
    with driver.session() as session:
        for node_class, node_def in schema["nodes"].items():
            identity_keys = node_def["identity_keys"]
            
            if identity_keys:
                for key in identity_keys:
                    safe_key = _cypher_safe(key)
                    try:
                        session.run(
                            f"CREATE INDEX IF NOT EXISTS "
                            f"FOR (n:`{node_class}`) ON (n.{safe_key})"
                        )
                    except Exception as e:
                        print(f"Could not create index on {node_class}.{safe_key}: {e}")
        
        try:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n) ON (n._node_key)")
        except Exception:
            pass
    
    print("Indexes created")

from collections import defaultdict

def ingest_edges_batch_unwind(edges: List[dict], driver: Driver, batch_size: int = 5000):
    by_pred = defaultdict(list)
    for e in edges:
        pred = _cypher_safe(e["predicate"])
        by_pred[pred].append({"sub": e["subject_key"], "obj": e["object_key"]})

    total = len(edges)
    processed = 0
    failed = 0

    with driver.session() as session:
        for pred, rows in by_pred.items():
            for i in range(0, len(rows), batch_size):
                batch_rows = rows[i : i + batch_size]
                cypher = (
                    "UNWIND $rows AS r\n"
                    "MATCH (a {_node_key: r.sub}), (b {_node_key: r.obj})\n"
                    f"MERGE (a)-[rel:`{pred}`]->(b)\n"
                    "RETURN count(rel) as created"
                )
                def _tx_work(tx):
                    res = tx.run(cypher, rows=batch_rows)
                    return res.single().get("created", 0)
                created = session.execute_write(_tx_work)
                processed += len(batch_rows)
                print(f"  Progress: {processed}/{total} edges, created in batch: {created}", end="\r")

    print(f"\nEdges processed (input count): {total}")
    return total


def ingest_nodes_batch(nodes: Dict[str, dict], driver: Driver, batch_size: int = 5000):
    node_list = list(nodes.values())
    total = len(node_list)
    ingested = 0
    
    print(f"\n[DEBUG] First 3 nodes to ingest:")
    for i, node in enumerate(node_list[:3]):
        print(f"  {i}: {node['class']} - key: {node['key'][:80]}...")
    
    with driver.session() as session:
        for batch in batched(node_list, batch_size):
            def _batch_tx(tx):
                for node in batch:
                    label = node["class"]
                    props = _normalize_props(node["properties"])
                    props = _sanitize_props(props)
                    node_key = node["key"]
                    
                    props["_node_key"] = node_key
                    
                    cypher = (
                        f"MERGE (n:`{label}` {{_node_key: $_node_key}}) "
                        "SET n += $props"
                    )
                    tx.run(cypher, _node_key=node_key, props=props)
            
            session.execute_write(_batch_tx)
            ingested += len(batch)
            print(f"  Progress: {ingested}/{total} nodes", end="\r")
    
    print(f"\nCreated {total} nodes")
    return total

def ingest_edges_batch(edges: List[dict], driver: Driver, batch_size: int = 5000):
    total = len(edges)
    ingested = 0
    failed = 0
    
    print(f"\n[DEBUG] Starting edge ingestion. Total edges to process: {total}")
    print(f"[DEBUG] First 3 edges to process:")
    for i, edge in enumerate(edges[:3]):
        print(f"  {i}: {edge['subject_key'][:60]}... -[{edge['predicate']}]-> {edge['object_key'][:60]}...")
    
    with driver.session() as session:
        for batch_idx, batch in enumerate(batched(edges, batch_size)):
            print(f"\n[DEBUG] Processing batch {batch_idx + 1}, size: {len(batch)}")
            
            def _batch_tx(tx):
                nonlocal failed, ingested
                batch_created = 0
                
                for _, edge in enumerate(batch):
                    subj_key = edge["subject_key"]
                    obj_key = edge["object_key"]
                    predicate = _cypher_safe(edge["predicate"])
                    
                    check_subj = "MATCH (a {_node_key: $key}) RETURN count(a) as cnt"
                    check_obj = "MATCH (b {_node_key: $key}) RETURN count(b) as cnt"
                    
                    subj_exists = tx.run(check_subj, key=subj_key).single()["cnt"]
                    obj_exists = tx.run(check_obj, key=obj_key).single()["cnt"]
                    
                    if subj_exists == 0:
                        failed += 1
                        if failed <= 10: 
                            print(f"\n  Subject node NOT FOUND:")
                            print(f"    Key: {subj_key[:100]}...")
                        continue
                    
                    if obj_exists == 0:
                        failed += 1
                        if failed <= 10:
                            print(f"\n  Object node NOT FOUND:")
                            print(f"    Key: {obj_key[:100]}...")
                        continue
                    
                    cypher = (
                        "MATCH (a {_node_key: $subj_key}), (b {_node_key: $obj_key}) "
                        f"MERGE (a)-[r:`{predicate}`]->(b) "
                        "RETURN id(r) as rel_id"
                    )
                    
                    try:
                        result = tx.run(cypher, subj_key=subj_key, obj_key=obj_key).single()
                        if result:
                            batch_created += 1
                            ingested += 1
                            if batch_created <= 3:
                                print(f"  Created: {predicate} (rel_id: {result['rel_id']})")
                        else:
                            failed += 1
                            if failed <= 10:
                                print(f"\n  No result returned for edge")
                    except Exception as e:
                        failed += 1
                        if failed <= 10:
                            print(f"\n  Exception: {e}")
                            print(f"    Predicate: {predicate}")
                
                return batch_created
            
            batch_created = session.execute_write(_batch_tx)
            print(f"  Batch result: {batch_created} created, {failed} failed so far")
            print(f"  Total progress: {ingested}/{total} edges", end="\r")
    
    print(f"\n\nCreated {ingested} edges")
    if failed > 0:
        print(f"Failed to create {failed} edges")
        if failed > 10:
            print(f"  (only first 10 failures shown above)")
    return ingested

def print_graph_stats(driver: Driver):
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    
    with driver.session() as session:
        result = session.run(
            "MATCH (n) "
            "RETURN labels(n)[0] as label, count(n) as count "
            "ORDER BY count DESC"
        )
        print("\nNodes by type:")
        for record in result:
            print(f"  {record['label']}: {record['count']}")
        
        result = session.run(
            "MATCH ()-[r]->() "
            "RETURN type(r) as rel_type, count(r) as count "
            "ORDER BY count DESC"
        )
        print("\nRelationships by type:")
        for record in result:
            print(f"  {record['rel_type']}: {record['count']}")
        
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        node_count = result.single()["node_count"]
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) as edge_count")
        edge_count = result.single()["edge_count"]
        
        print(f"\nTotal: {node_count} nodes, {edge_count} relationships")
    
    print("="*60)

parser = argparse.ArgumentParser(description="Load edge-list knowledge graph into Neo4j with schema validation")
parser.add_argument("--graph", type=Path, help="JSON file with edge list")
parser.add_argument("--schema", type=Path, help="Schema JSON")
parser.add_argument("--uri", default="bolt://localhost:7687")
parser.add_argument("--user", default="neo4j")
parser.add_argument("--pwd", default="emeraldmind")
parser.add_argument("--clear", action="store_true",
                    help="Clear database before loading")
parser.add_argument("--batch-size", type=int, default=5000,
                    help="Batch size for node/edge creation (default: 5000)")
parser.add_argument("--skip-invalid", action="store_true",
                    help="Skip invalid triples and continue loading")
args = parser.parse_args()

def main():
    print(f"Loading graph from: {args.graph}")
    print(f"Using schema: {args.schema}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        schema = load_schema(args.schema)
        print(f"Loaded schema with {len(schema['nodes'])} node types and {len(schema['edges'])} edge types")
    except Exception as e:
        print(f"Failed to load schema: {e}")
        return
    
    print("\nLoading graph...")
    try:
        edge_list = json.loads(args.graph.read_text(encoding="utf-8"))
        print(f"Loaded {len(edge_list)} edges from file")
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return
    
    print("\nValidating and extracting nodes and edges...")
    nodes, edges, errors = validate_and_extract(edge_list, schema)
    
    print(f"\nValidation results:")
    print(f"  Total edges in file: {len(edge_list)}")
    print(f"  Valid edges: {len(edges)}")
    print(f"  Invalid edges: {len(errors)}")
    print(f"  Unique nodes: {len(nodes)}")
    
    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for error in errors[:10]:
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        
        if not args.skip_invalid and errors:
            print("\nValidation failed. Use --skip-invalid to load valid triples only.")
            return
    
    if not nodes:
        print("\nNo valid nodes to ingest.")
        return
    
    print(f"\nConnecting to Neo4j at {args.uri}...")
    try:
        driver = GraphDatabase.driver(args.uri, auth=(args.user, args.pwd))
        driver.verify_connectivity()
        print("Connected to Neo4j")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return
    
    try:
        if args.clear:
            clear_database(driver)
        
        setup_indexes(driver, schema)
        print(f"\nIngesting {len(nodes)} nodes...")
        ingest_nodes_batch_unwind(nodes, driver, args.batch_size)
        print(f"\nIngesting {len(edges)} edges...")
        ingest_edges_batch_unwind(edges, driver, args.batch_size)
        print_graph_stats(driver)
        print("\nGraph successfully loaded into Neo4j")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()