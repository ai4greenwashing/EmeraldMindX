#!/usr/bin/env python3

import json
import pathlib
from google import genai
from dotenv import load_dotenv
import re
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def create_clients():
    clients = []
    for i in range(1, 8):
        api_key = os.getenv(f"GEMINI_API_KEY_{i}")
        if api_key:
            clients.append(genai.Client(api_key=api_key))
            logger.info(f"Initialized client {i}")
        else:
            logger.warning(f"GEMINI_API_KEY_{i} not found")
    
    if not clients:
        raise ValueError("No Gemini API keys found in environment")
    
    return clients


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
    
    logger.info(f"Loaded {len(edge_directions)} edge direction rules")
    return entity_classes, edge_labels, edge_directions


def load_triples_from_file(file_path: pathlib.Path) -> Tuple[List[Dict], str]:
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        
        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            nodes = data["nodes"]
            edges = data["edges"]
            triples = []
            
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                subj_idx = edge.get("subject")
                obj_idx = edge.get("object")
                pred = edge.get("predicate")
                
                if subj_idx is None or obj_idx is None or not pred:
                    continue
                
                if not (0 <= subj_idx < len(nodes)) or not (0 <= obj_idx < len(nodes)):
                    logger.warning(f"Invalid node index in {file_path.name}")
                    continue
                
                triple = {
                    "subject": {
                        "class": nodes[subj_idx].get("class"),
                        "properties": nodes[subj_idx].get("properties", {})
                    },
                    "predicate": pred,
                    "object": {
                        "class": nodes[obj_idx].get("class"),
                        "properties": nodes[obj_idx].get("properties", {})
                    }
                }
                
                if "temporal_metadata" in edge:
                    triple["temporal_metadata"] = edge["temporal_metadata"]
                
                triples.append(triple)
            
            return triples, "graph"
        
        elif isinstance(data, list):
            valid_triples = []
            for i, triple in enumerate(data):
                if not isinstance(triple, dict):
                    logger.warning(f"{file_path.name}: Skipping non-dict triple at index {i}")
                    continue
                if not triple.get("subject") or not triple.get("object") or not triple.get("predicate"):
                    logger.warning(f"{file_path.name}: Skipping triple with missing fields at index {i}")
                    continue
                valid_triples.append(triple)
            
            if len(valid_triples) < len(data):
                logger.info(f"{file_path.name}: Filtered {len(data) - len(valid_triples)} malformed triples")
            
            return valid_triples, "triple_array"
        else:
            logger.warning(f"{file_path.name}: Unexpected format")
            return [], "unknown"
            
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return [], "error"


def fix_direction(triple: Dict, edge_directions: Dict[str, List[Tuple[str, str]]]) -> Tuple[Dict, bool]:
    pred = triple.get("predicate")
    subj = triple.get("subject") or {}
    obj = triple.get("object") or {}
    
    subj_class = subj.get("class") if isinstance(subj, dict) else None
    obj_class = obj.get("class") if isinstance(obj, dict) else None
    
    if not pred or not subj_class or not obj_class:
        return triple, False
    
    if pred not in edge_directions:
        return triple, False
    
    valid_directions = edge_directions[pred]
    
    current_valid = any(
        (from_cls == subj_class and to_cls == obj_class)
        for from_cls, to_cls in valid_directions
    )
    
    if current_valid:
        return triple, False
    
    reversed_valid = any(
        (from_cls == obj_class and to_cls == subj_class)
        for from_cls, to_cls in valid_directions
    )
    
    if reversed_valid:
        # swap
        fixed_triple = {
            "subject": obj,
            "predicate": pred,
            "object": subj
        }
        if "temporal_metadata" in triple:
            fixed_triple["temporal_metadata"] = triple["temporal_metadata"]
        return fixed_triple, True
    
    return triple, False


def validate_triple(triple: Dict, entity_classes: Set[str], edge_labels: Set[str], 
                     edge_directions: Dict[str, List[Tuple[str, str]]]) -> Tuple[bool, List[str]]:
    errors = []
    
    # check structure
    if not isinstance(triple, dict):
        return False, ["Not a dict"]
    
    if not {"subject", "predicate", "object"}.issubset(triple.keys()):
        return False, ["Missing required keys (subject/predicate/object)"]
    
    subj = triple.get("subject")
    if subj is None:
        errors.append("Subject is None")
    elif not isinstance(subj, dict):
        errors.append("Subject not a dict")
    elif "class" not in subj or "properties" not in subj:
        errors.append("Subject missing class or properties")
    elif subj["class"] not in entity_classes:
        errors.append(f"Invalid subject class: {subj.get('class')}")
    elif not isinstance(subj["properties"], dict):
        errors.append("Subject properties not a dict")
    else:
        props = subj["properties"]
        if "valid_from" not in props:
            errors.append("Subject missing valid_from")
        if "valid_to" not in props:
            errors.append("Subject missing valid_to")
        if "is_current" not in props:
            errors.append("Subject missing is_current")
    
    obj = triple.get("object")
    if obj is None:
        errors.append("Object is None")
    elif not isinstance(obj, dict):
        errors.append("Object not a dict")
    elif "class" not in obj or "properties" not in obj:
        errors.append("Object missing class or properties")
    elif obj["class"] not in entity_classes:
        errors.append(f"Invalid object class: {obj.get('class')}")
    elif not isinstance(obj["properties"], dict):
        errors.append("Object properties not a dict")
    else:
        props = obj["properties"]
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
        # Check edge direction
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


def process_file_offline(file_path: pathlib.Path, entity_classes: Set[str], 
                        edge_labels: Set[str], edge_directions: Dict) -> Tuple[List[Dict], List[Dict], Dict[str, int], str]:
    stats = {
        "total": 0,
        "direction_fixed": 0,
        "valid": 0,
        "invalid": 0
    }
    
    # Load triples
    triples, format_type = load_triples_from_file(file_path)
    stats["total"] = len(triples)
    
    if not triples:
        return [], [], stats, format_type
    
    fixed_triples = []
    for triple in triples:
        fixed, was_fixed = fix_direction(triple, edge_directions)
        if was_fixed:
            stats["direction_fixed"] += 1
        fixed_triples.append(fixed)
    
    valid = []
    invalid = []
    
    for triple in fixed_triples:
        is_valid, errors = validate_triple(triple, entity_classes, edge_labels, edge_directions)
        if is_valid:
            valid.append(triple)
            stats["valid"] += 1
        else:
            triple["_validation_errors"] = errors
            triple["_source_file"] = str(file_path)
            invalid.append(triple)
            stats["invalid"] += 1
    
    return valid, invalid, stats, format_type


BATCH_FIX_PROMPT = (
    "You are fixing invalid ESG knowledge graph triples to match a schema.\n\n"
    "## VALIDATION RULES\n"
    "1. **Fix typos/synonyms**: Correct class names and predicates to match schema exactly\n"
    "2. **Add missing temporal properties**: Ensure all nodes have valid_from, valid_to, is_current\n"
    "3. **Add missing edge metadata**: Ensure all edges have temporal_metadata\n"
    "4. **Schema compliance**:\n"
    "   - predicate must be in edge labels\n"
    "   - subject.class & object.class must be in entity classes\n"
    "5. **Discard unfixable**: If triple cannot be corrected, omit it from output\n\n"
    "## TEMPORAL PROPERTIES (REQUIRED)\n"
    "All nodes MUST have:\n"
    "• valid_from: When information became valid (YYYY or YYYY-MM-DD)\n"
    "• valid_to: When superseded (null if current)\n"
    "• is_current: Boolean\n\n"
    "All edges MUST have temporal_metadata:\n"
    "• valid_from, valid_to, recorded_at\n\n"
    "## COMMON FIXES\n"
    "• Missing temporal properties: Use context year as valid_from\n"
    "• Typo in class/predicate: Match to closest schema term\n"
    "• Missing temporal_metadata: Create with reasonable defaults\n\n"
    "SCHEMA:\n"
    "{schema_json}\n\n"
    "OUTPUT FORMAT:\n"
    "Return JSON array of valid triples in the SAME ORDER as input.\n"
    "For unfixable triples, return null in that position.\n\n"
    "Output ONLY valid JSON - no markdown, no prose.\n\n"
    "INVALID tripleS TO FIX:\n"
)


def extract_json_from_response(response_text: str) -> list:
    text = response_text.strip()
    
    # Remove markdown
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    
    # Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Remove comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Find JSON array boundaries
    start = text.find('[')
    end = text.rfind(']') + 1
    
    if start != -1 and end > start:
        text = text[start:end]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}")
    
    return []


def fix_batch_with_llm(batch: List[Dict], schema: dict, client: genai.Client) -> List[Dict]:
    if not batch:
        return []
    
    clean_batch = []
    for triple in batch:
        clean_triple = {k: v for k, v in triple.items() if not k.startswith("_")}
        clean_batch.append(clean_triple)
    
    prompt = BATCH_FIX_PROMPT.format(
        schema_json=json.dumps(schema, indent=2, ensure_ascii=False)
    ) + json.dumps(clean_batch, indent=2, ensure_ascii=False)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json"
            }
        )
        
        fixed = extract_json_from_response(response.text)
        
        if isinstance(fixed, list):
            return [t for t in fixed if t is not None]
        
        return []
        
    except Exception as e:
        logger.error(f"LLM fix failed: {e}")
        return []


def process_all_files(input_dir: pathlib.Path, schema: dict, 
                     entity_classes: Set[str], edge_labels: Set[str],
                     edge_directions: Dict, clients: List,
                     batch_size: int = 50) -> None:
    """
    Process all files:
    1. Load and fix directions offline
    2. Batch invalid triples by error type
    3. Fix batches with LLM
    4. Save results
    """
    graph_files = list(input_dir.rglob("page*.json"))
    
    normal_files = [
        f for f in graph_files 
        if not any(suffix in f.stem for suffix in ["_validated", "_bugged", "_fixed", "_unfixable"])
    ]
    
    bugged_files = [
        f for f in graph_files 
        if "_bugged" in f.stem and not any(suffix in f.stem for suffix in ["_validated", "_fixed", "_unfixable"])
    ]
    
    all_files = normal_files + bugged_files
    
    if not all_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(normal_files)} normal files and {len(bugged_files)} bugged files")
    
    logger.info("═══ Phase 1: Offline Processing ═══")
    all_valid = []
    all_invalid = []
    total_stats = defaultdict(int)
    format_counts = defaultdict(int)
    
    for file_path in all_files:
        valid, invalid, stats, format_type = process_file_offline(file_path, entity_classes, edge_labels, edge_directions)
        
        all_valid.extend(valid)
        all_invalid.extend(invalid)
        format_counts[format_type] += 1
        
        for key, value in stats.items():
            total_stats[key] += value
        
        if stats["direction_fixed"] > 0:
            logger.info(f"  {file_path.name}: Fixed {stats['direction_fixed']} directions offline")
    
    logger.info(
        f"\nOffline Results:\n"
        f"  Total files: {len(all_files)} (graph: {format_counts['graph']}, triple_array: {format_counts['triple_array']})\n"
        f"  Total triples: {total_stats['total']}\n"
        f"  Direction fixed: {total_stats['direction_fixed']}\n"
        f"  Valid: {total_stats['valid']}\n"
        f"  Invalid (need LLM): {total_stats['invalid']}"
    )
    
    fixed_triples = []
    if all_invalid:
        logger.info(f"\n═══ Phase 2: LLM Batch Fixing ({len(all_invalid)} triples) ═══")
        
        client_idx = 0
        
        for i in range(0, len(all_invalid), batch_size):
            batch = all_invalid[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_invalid)-1)//batch_size + 1} ({len(batch)} triples)")
            
            client = clients[client_idx % len(clients)]
            client_idx += 1
            
            fixed_batch = fix_batch_with_llm(batch, schema, client)
            
            for triple in fixed_batch:
                is_valid, _ = validate_triple(triple, entity_classes, edge_labels, edge_directions)
                if is_valid:
                    fixed_triples.append(triple)
            
            logger.info(f"  Batch result: {len(fixed_batch)} returned, {len([t for t in fixed_batch if t])} validated")
            
            time.sleep(1)
        
        logger.info(f"\nLLM Fixed: {len(fixed_triples)}/{len(all_invalid)} triples")
        all_valid.extend(fixed_triples)
    
    logger.info(f"\n═══ Phase 3: Saving Results ═══")
    
    output_dir = input_dir
    output_dir.mkdir(exist_ok=True)
    
    # Save all valid triples
    output_file = output_dir / "all_validated_triples.json"
    output_file.write_text(
        json.dumps(all_valid, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"Saved {len(all_valid)} valid triples to {output_file}")
    
    # Save unfixable triples
    unfixable_count = total_stats['invalid'] - len(fixed_triples) if all_invalid else 0
    if unfixable_count > 0:
        unfixable = [t for t in all_invalid if t not in fixed_triples]
        unfixable_file = output_dir / "unfixable_triples.json"
        unfixable_file.write_text(
            json.dumps(unfixable, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Saved {unfixable_count} unfixable triples to {unfixable_file}")
    
    logger.info(
        f"\n═══ Final Summary ═══\n"
        f"Total input files: {len(all_files)} (normal: {len(normal_files)}, bugged: {len(bugged_files)})\n"
        f"Total input triples: {total_stats['total']}\n"
        f"Direction fixed offline: {total_stats['direction_fixed']}\n"
        f"Initially valid: {total_stats['valid']}\n"
        f"Fixed by LLM: {len(fixed_triples)}\n"
        f"Final valid triples: {len(all_valid)}\n"
        f"Unfixable: {unfixable_count}\n"
        f"Success rate: {len(all_valid)/total_stats['total']*100:.1f}%" if total_stats['total'] > 0 else "Success rate: N/A"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fix ESG graph triples with offline direction fixing and batched LLM calls"
    )
    parser.add_argument(
        "--input_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing graph JSON files (both normal and _bugged)"
    )
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        required=True,
        help="Path to schema JSON file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Number of triples per LLM batch (default: 25)"
    )
    args = parser.parse_args()
    
    try:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        entity_classes, edge_labels, edge_directions = load_schema_sets(schema)
        logger.info(f"Loaded schema: {len(entity_classes)} entities, {len(edge_labels)} edges")
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        return
    
    clients = create_clients()
    process_all_files(
        args.input_dir, 
        schema, 
        entity_classes, 
        edge_labels, 
        edge_directions,
        clients,
        args.batch_size
    )

if __name__ == "__main__":
    main()