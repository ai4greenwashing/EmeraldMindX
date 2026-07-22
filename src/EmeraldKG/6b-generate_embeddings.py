from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return Path(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[2]
    

PROJECT_ROOT = get_git_root()
DATASETS_DIR = PROJECT_ROOT / "datasets"
SCHEMA_DIR = PROJECT_ROOT / "schemas"

def node_to_text(node: Dict[str, Any]) -> str:
    class_name = node.get('class', 'Unknown')
    properties = node.get('properties', {})
    text_parts = [f"{class_name}:"]
    
    for key, value in properties.items():
        if value is None or key in ['valid_from', 'valid_to', 'is_current']:
            continue
        text_parts.append(f"{key}: {value}")
    
    return " | ".join(text_parts)


def add_embeddings_to_claim_file(
    claim_file_path: Path,
    encoder: SentenceTransformer,
    overwrite: bool = False
) -> None:
    with open(claim_file_path, 'r') as f:
        data = json.load(f)
    
    if 'nodes' not in data:
        print(f"Warning: No 'nodes' key in {claim_file_path.name}, skipping...")
        return
    
    nodes = data['nodes']
    nodes_updated = 0
    
    for node in nodes:
        if not overwrite and 'embedding' in node:
            continue
        node_text = node_to_text(node)
        embedding = encoder.encode(node_text, normalize_embeddings=True)
        node['embedding'] = embedding.tolist()
        nodes_updated += 1
    with open(claim_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return nodes_updated


def generate_node_embeddings(
    claims_dir: str = "claims",
    model_name: str = "multi-qa-MiniLM-L6-cos-v1",
    overwrite: bool = False
) -> None:
    claims_path = Path(claims_dir)
    
    if not claims_path.exists():
        print(f"Error: Directory {claims_path} does not exist!")
        return
    print(f"Loading model: {model_name}...")
    encoder = SentenceTransformer(model_name)
    claim_files = sorted(claims_path.glob("claim_*.json"))
    
    if not claim_files:
        print(f"Error: No claim_*.json files found in {claims_path}")
        return
    
    print(f"\nFound {len(claim_files)} claim files")
    print(f"Overwrite existing embeddings: {overwrite}\n")
    
    total_nodes = 0
    total_updated = 0
    
    for claim_file in claim_files:
        print(f"Processing {claim_file.name}...", end=" ")
        
        try:
            nodes_updated = add_embeddings_to_claim_file(claim_file, encoder, overwrite)
            total_updated += nodes_updated
            with open(claim_file, 'r') as f:
                data = json.load(f)
                total_nodes += len(data.get('nodes', []))
            
            print(f"({nodes_updated} nodes updated)")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total claim files processed: {len(claim_files)}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Nodes with embeddings added/updated: {total_updated}")
    print(f"{'='*60}")


def verify_node_embeddings(claims_dir: str = "claims", sample_id: int = 0) -> None:
    claims_path = Path(claims_dir)
    sample_file = claims_path / f"claim_{sample_id}.json"
    
    if not sample_file.exists():
        print(f"Error: Sample file {sample_file} not found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Verification - Inspecting {sample_file.name}")
    print(f"{'='*60}\n")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    print(f"Total nodes: {len(nodes)}")
    
    class_counts = {}
    nodes_with_embeddings = 0
    
    for node in nodes:
        node_class = node.get('class', 'Unknown')
        class_counts[node_class] = class_counts.get(node_class, 0) + 1
        
        if 'embedding' in node:
            nodes_with_embeddings += 1
    
    print(f"Nodes with embeddings: {nodes_with_embeddings}/{len(nodes)}")
    print(f"\nNodes by class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  - {class_name}: {count}")
    print(f"\nSample node:")
    for node in nodes:
        if 'embedding' in node:
            print(f"  Class: {node.get('class')}")
            print(f"  Properties: {node.get('properties')}")
            print(f"  Embedding dimension: {len(node['embedding'])}")
            print(f"  First 10 embedding values: {node['embedding'][:10]}")
            break
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings for nodes in claim_{i}.json files"
    )
    parser.add_argument(
        "--claims_dir",
        type=str,
        default=f"{DATASETS_DIR}/claims",
        help="Directory containing claim_{i}.json files (default: claims)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="SentenceTransformer model name (default: multi-qa-MiniLM-L6-cos-v1)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings (default: False)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing embeddings without generating (default: False)"
    )
    parser.add_argument(
        "--sample_id",
        type=int,
        default=0,
        help="Sample claim ID to inspect during verification (default: 0)"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_node_embeddings(args.claims_dir, args.sample_id)
    else:
        generate_node_embeddings(
            claims_dir=args.claims_dir,
            model_name=args.model,
            overwrite=args.overwrite
        )
        verify_node_embeddings(args.claims_dir, args.sample_id)
