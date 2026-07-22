# EmeraldMind: A Knowledge Graph–Augmented Framework for Greenwashing Detection


---

## Overview

This repository includes:

**Key Features:**

- EmeraldGraph, a domain-specific Knowledge Graph, its schema and the full pipeline for its construction
- EmeraldDB, a document store extracted from ESG reports for text-based retrieval
- Claim grounding, retrieval, and classification modules
- Scripts for reproducible end-to-end ESG knowledge extraction

This README provides instructions for installation, environment setup, pipeline execution, and quick-start options.

---

## Citation

If you use EmeraldMind, EmeraldGraph, or EmeraldData in your research, please cite our paper as follows

```
@inproceedings{kaoukis2026emeraldmind,
  author = {Kaoukis, Georgios and Koufopoulos, Ioannis-Aris and Psaroudaki, Eleni and Karidi, Danae Pla Karidi and Pitoura, Evaggelia and Papastefanatos, George and Tsaparas, Panayiotis},
  title = {EmeraldMind: A Knowledge Graph-Augmented Framework for Greenwashing Detection},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year = {2026},
  location = {Dubai, United Arab Emirates},
  publisher = {ACM},
  address = {New York, NY, USA},
  pages = {11}
}
```
## Final directory structure

```
emeraldmind/
├── claim_gen_scripts/       # Scripts for generating the Emerald Data
├── schemas/                 # Schemas defining relationship and classification logic
├── data/                    # Compressed ESG report data (images, text, KPIs)
├── datasets/                # Processed datasets (Green Claims, Emerald Data)
├── figures/                 # Assets
├── llm_judge_scripts/       # Evaluation scripts using LLMs (ranking, ILORA)
├── results/                 # Experiment results
├── src/                     # Source code for the core application
│   ├── EmeraldDB/           # Vector Database and RAG implementation
│   ├── EmeraldKG/           # Knowledge Graph construction pipeline and KG-RAG
│   │   ├── 0-get_all_image_descriptions.py
│   │   ├── ...
│   │   └── 7-classify.py
│   ├── utils/               # Utility scripts, prompts, and hybrid LLM logic
│   ├── baseline.py          # LLM classification, without RAG
│   └── hybrid_llm.py        # Classification using EmeraldDB and EmeraldKG combined
├── .env.example             # Template for environment variables
├── graph_latest.json        # Latest knowledge graph export
├── kpi_definitions.json     # KPI definitions configuration
├── requirements.txt         # Python dependencies
└── result_analysis.ipynb    # Jupyter notebook for analyzing results
```

---

## Installation

```bash
# Download the code from the anonymous repo
cd emeraldmind
pip install -r requirements.txt

# Setup Neo4j
docker run -d --name neo4j-emerald \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/emeraldmind \
  neo4j

# Configure environment
cp .env.example .env
# Edit .env

# Extract pre-processed reports
cd data
ls *.gz |xargs -n1 tar -xzf
cd ..
mkdir reports
cp -r data/*/* reports/
```

---

## Pipeline Execution

### EmeraldDB

EmeraldDB handles the text-based retrieval (RAG) component using ChromaDB. It chunks the ESG reports and generates vector embeddings.

#### Step 1: Ingest Reports into ChromaDB

Ensure your text reports are located in reports/ (extracted in the installation step). This script chunks the text and creates the persistent database in `chroma_db/`.

```bash
python src/EmeraldDB/vectordb.py
```

#### Step 2: Run Classification

```bash
python src/EmeraldDB/em_rag.py
```

### EmeraldKG

**Quick Start:** The repository includes a pre-built knowledge graph (`graph_latest.json`). You can **skip directly to Step 5** to load it into Neo4j, then proceed with Steps 6-7 for classification.

To rebuild the graph from scratch, follow Steps 0-4 below.

[View the EmeraldGraph schema.](./figures/schema_neo4.pdf)

---

#### Step 0 (Optional): Extract Image Descriptions

Only needed if you have raw PDFs without pre-extracted data.

```bash
python src/EmeraldKG/0-get_all_image_descriptions.py
```

Scans `reports/*.pdf`, extracts charts/figures using Gemini Vision, creates `{company}_{year}_images/` folders.

---

#### Step 1: Extract KPI Observations

```bash
python src/EmeraldKG/1-kpi-extraction.py \
  --reports-dir path/to/reports \
  --kpi_defs path/to/kpi_definitions.json
```

Reads `reports/{company}_{year}_text/` and `reports/{company}_{year}_images/`, extracts KPI values, saves to `reports/{company}_{year}_kpi/`.

---

#### Step 2: Extract Entity-Relation Triples

```bash
python src/EmeraldKG/2-extract-triple.py \
  --input_dir reports \
  --schema path/to/schema.json \
  --output path/to/graphs_folder
```

Uses LLM to extract (entity, relation, entity) triples from text/KPI/image data.

---

#### Step 3: Validate Triples

```bash
python src/EmeraldKG/3-fix-invalid-triple.py \
  --input_dir path/to/graphs_folder \
  --schema path/to/schema.json \
```

Validates triples against schema, fixes type mismatches, removes duplicates. Stores all valid triples in `graphs_folder/all_validated_triples.json` and invalid in `graphs_folder/unfixable_triples.json`.

---

#### Step 4: Entity Resolution

```bash
python src/EmeraldKG/4-entity_resolution.py \
  --input_file path/to/graphs_folder/all_validated_triples.json \
  --output_file path/to/graph_latest.json \
  --schema path/to/schemas/schema.json \
  --similarity_threshold 0.8
```

Finds and merges duplicate entities using embedding similarity + LLM validation.

---

#### Step 5: Load to Neo4j

**Start here if using the pre-built graph.**

```bash
python src/EmeraldKG/5-load_edgelist_graph.py \
  --graph path/to/graph_latest.json \
  -- schema path/to/graph_schema.json \
  --clear
```

Loads graph into Neo4j. Use `--clear` to remove existing data first.

Verify at `http://localhost:7474`:

```cypher
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);
```

---

#### Step 6a: Parse Claims to Nodes

```bash
python src/EmeraldKG/6a-parse_claims_to_nodes.py \
  -i path/to/datasets/emerald_data.csv \
  -o claims_mixed \
  --claim-col claim \
  --company-col company
```

```bash
python src/EmeraldKG/6a-parse_claims_to_nodes.py \
  -i path/to/datasets/green_claims.csv \
  -o claims_small \
  --claim-col claim \
  --company-col company
```

Grounds claims to graph entities, creates `claim_{i}.json` files.

**Required:** `GEMINI_API_KEY_1` through `GEMINI_API_KEY_8` in `.env` for parallel processing.

---

#### Step 6b: Generate Embeddings

```bash
python src/EmeraldKG/6b-generate_embeddings.py \
  --claims_dir path/to/claims_emerald_data \
  --model multi-qa-MiniLM-L6-cos-v1
```

```bash
python src/EmeraldKG/6b-generate_embeddings.py \
  --claims_dir path/to/claims_green_claims \
  --model multi-qa-MiniLM-L6-cos-v1
```

Adds embeddings to nodes in `claim_{i}.json` files.

Options:

- `--overwrite`: Regenerate existing embeddings
- `--verify`: Check embeddings without generating

---

#### Step 7: Run Classification

```bash
python src/EmeraldDB/7-classify.py \
  --dataset path/to/claims_emerald_data.csv \
  --claims_dir  path/to/claims_emerald_data \
  --output  path/to/results/emerald_data_results.json
```

```bash
python src/EmeraldDB/7-classify.py \
  --dataset  path/todatasets/small_dataset.csv \
  --claims_dir  path/to/claims_green_claims \
  --output  path/toresults/green_claim_results.json
```

Runs EmeraldMind pipeline: claim grounding → retrieval → classification.

---

## Quick Start Options

### Option A: Use Pre-built Graph (Recommended)

```bash
cd src/EmeraldDB
# Start from Step 5
python 5-load_edgelist_graph.py --graph graph_latest.json --clear
python 6a-parse_claims_to_nodes.py -i datasets/mixed_dataset.csv -o claims_mixed
python 6a-parse_claims_to_nodes.py -i datasets/small_dataset.csv -o claims_small
python 6b-generate_embeddings.py --claims_dir claims_mixed
python 6b-generate_embeddings.py --claims_dir claims_small
python 7-classify.py --dataset datasets/mixed_dataset.csv --claims_dir claims_mixed
python 7-classify.py --dataset datasets/small_dataset.csv --claims_dir claims_small
```

### Option B: Build Graph from Scratch

```bash
cd src/EmeraldDB
# Run all steps (with pre-extracted data)
python 1-kpi-extraction.py
python 2-extract-triple.py
python 3-fix-invalid-triple.py
python 3-merge-graphs.py
python 4-entity_resolution.py
python 5-load_edgelist_graph.py --clear
python 6a-parse_claims_to_nodes.py -i datasets/mixed_dataset.csv -o claims_mixed
python 6a-parse_claims_to_nodes.py -i datasets/small_dataset.csv -o claims_small
python 6b-generate_embeddings.py --claims_dir claims_mixed
python 6b-generate_embeddings.py --claims_dir claims_small
python 7-classify.py --dataset datasets/mixed_dataset.csv --claims_dir claims_mixed
python 7-classify.py --dataset datasets/small_dataset.csv --claims_dir claims_small
```

---

### Generating Emerald Data

To generate the Emerald Data dataset Use the scripts under `claim_gen_scripts/` . Run the claim generation scripts, then merge their outputs. The claim generation scripts produce per-claim JSON/CSV artifacts — a simple merge/concatenate step (e.g., jq, pandas.concat, or CSV cat/awk) aggregates these into the final dataset used by downstream steps.

### Evaluation & Ranking

**Reproduce ILoRA evaluation:**

- See `llm_judge_scripts/ilora_eval.py` for the reproducible ILoRA evaluation pipeline. That script contains the configuration and invocation sequence needed to run the ILoRA-style judged evaluation used in our experiments.

**Rank pipelines and create hybrid combinations::**

- Use `llm_judge_scripts/pipeline_ranking.py` to rank retrieval/classification pipelines and to help construct hybrid ensembles. That script implements pairwise comparisons and ranking logic over pipeline outputs.

**Run classification: non-RAG and hybrid modes**

- Non-RAG (baseline LLM classification): python `src/baseline.py` (performs classification using the LLM without RAG).
- Hybrid (EmeraldDB + EmeraldKG combined): python `src/hybrid_llm.py` (Utilizes results from pipeline_ranking.py to determine the best retrieval method).

## Environment Variables

Required in `.env`:

```bash
# Gemini API (8 keys for parallel processing in Step 6a)
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3
GEMINI_API_KEY_4=your_key_4
GEMINI_API_KEY_5=your_key_5
GEMINI_API_KEY_6=your_key_6
GEMINI_API_KEY_7=your_key_7
GEMINI_API_KEY_8=your_key_8
HF_TOKEN=your_hf_token
```

