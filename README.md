# EmeraldMindX

Code, data and per-claim outputs for *EmeraldMindX: A Hybrid Framework for Greenwashing Detection* (IEEE TKDE, under review).

## Repository structure

| Path | Contents |
|---|---|
| `datasets/` | The three evaluation datasets (`green_claims.csv`, `emerald_data.csv`, `emerald_promise.csv`) |
| `datasets/data/` | Evidence of the GreenClaims reports: page-level text (`text.tar.gz`), extracted KPIs (`kpis.tar.gz`) and image descriptions (`image_desc.tar.gz`) |
| `schemas/`, `kpi_definitions.json` | Knowledge-graph schema and KPI definitions used to build EmeraldGraphX |
| `graph_latest.json` | EmeraldGraphX as an edge list |
| `src/EmeraldKG/` | Construction of EmeraldGraphX (steps `0`–`6b`), classification with the EmeraldMind graph retrieval (`7-classify.py`) and EMX-KGRAG with guided beam search (`emx_kgrag.py`) |
| `src/EmeraldDB/` | Construction of EmeraldDBX (`vectordb.py`) and EMX-RAG (`emx_rag.py`) |
| `src/baseline.py` | Baseline (LLM without retrieval) |
| `src/voting_scripts/` | EMX-VOTING (`emx-voting.py`) and EMX-SVOTING (`emx-svoting.py`) |
| `llm_judge_scripts/` | ILORA scoring of the justifications (`pipeline_ranking.py`) and radar charts (`ilora_eval.py`) |
| `claim_gen_scripts/` | Generation of the EmeraldData and EmeraldPromise claims |
| `tests/` | Unit tests of the guided beam search (`pytest tests/`, no database or GPU needed) |
| `results/` | Per-claim outputs (see below) |

## Datasets

| Dataset | File | Claims (greenwashing / not) | Source | License |
|---|---|---|---|---|
| GreenClaims | `datasets/green_claims.csv` | 51 (24 / 27) | Subset of the GreenClaims dataset (Fornasiero, 2024, https://github.com/DizzyPanda1/GreenwashingDetectionDataset) with publicly available ESG reports | MIT |
| EmeraldData | `datasets/emerald_data.csv` | 620 (225 / 395) | Semi-synthetic claims generated from ESG news articles (`claim_gen_scripts/`), introduced in EmeraldMind (Kaoukis et al., WWW 2026, https://github.com/ai4greenwashing/EmeraldMind) | Apache 2.0, as released with EmeraldMind |
| EmeraldPromise | `datasets/emerald_promise.csv` | 558 (258 / 300) | Derived from the English part of ML-Promise (Seki et al., EMNLP 2025, http://MLpromise.nlpfin.com/) | CC BY-NC-SA 4.0, as ML-Promise |

The evidence archives in `datasets/data/` cover the 37 ESG reports of the GreenClaims companies. The reports remain the property of the respective companies and are publicly available from their websites. The reports of the EmeraldPromise companies are not redistributed. Their links are in the `source_url` column of `emerald_promise.csv`.

## Models

| Use | Model |
|---|---|
| Baseline, EMX-RAG and EMX-KGRAG generation | `google/gemma-3-27b-it` |
| Knowledge-graph construction (steps `1`, `2`, `3`, `6a`) | `gemini-2.5-flash` (Google API) |
| Image descriptions for EmeraldDBX (`vectordb.py`) | `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` |
| Entity resolution (step `4`) | `nomic-embed-text` (embeddings) and `gemma3:latest` (match verification), both run locally through Ollama |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (EmeraldDBX), `multi-qa-MiniLM-L6-cos-v1` (claim nodes), `sentence-transformers/all-MiniLM-L6-v2` (KPI extraction) |
| Voting judges | `prometheus-eval/prometheus-7b-v2.0`, `Qwen/Qwen2.5-7B-Instruct`, `microsoft/Phi-4-mini-instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `google/gemma-3-4b-it`, `deepseek-ai/deepseek-llm-7b-chat`, `ibm-granite/granite-4.1-8b`; single large judge `google/gemma-3-27b-it` |
| ILORA evaluator | `prometheus-eval/prometheus-13b-v1.0` |

Open models are loaded from Hugging Face. Their licenses are given on the respective model cards. The panel compositions are defined in `src/voting_scripts/common_utils.py`. The Gemini API does not allow pinning a fixed model snapshot, and the Ollama tag `gemma3:latest` may point to a newer model version, so the graph-construction steps may not be exactly reproducible.

## Decoding settings

Classification, judging and ILORA evaluation use temperature 0 or close to it (EMX-RAG 0.01, baseline 1e-5). No sampling seed is set.

## Running the pipelines

1. Install the requirements (`pip install -r requirements.txt`) and fill in `.env` from `.env.example` (Gemini API keys, Hugging Face token). EmeraldGraphX requires a Neo4j instance with the Graph Data Science plugin.
2. Build the evidence stores: `src/EmeraldKG/0-…` to `6b-…` in order for EmeraldGraphX, and `src/EmeraldDB/vectordb.py` for EmeraldDBX.
3. Classify the claims with `src/baseline.py`, `src/EmeraldDB/emx_rag.py` and `src/EmeraldKG/emx_kgrag.py`.
4. Select between EMX-RAG and EMX-KGRAG with `src/voting_scripts/emx-voting.py` or `emx-svoting.py` (`--panel-size 1|3|5|7`).
5. Score the justifications with `llm_judge_scripts/pipeline_ranking.py` and plot them with `llm_judge_scripts/ilora_eval.py`.

Scripts with command-line options list them with `--help`. The others are configured at the top of the file.

## Per-claim outputs

| Path | Contents |
|---|---|
| `results/emx_pipelines/` | Baseline, EMX-RAG and EMX-KGRAG: label, justification and retrieved evidence for every claim |
| `results/base_predictions.csv` | Gold label and the EMX-RAG and EMX-KGRAG labels for every claim |
| `results/voting_results/` | EMX-VOTING and EMX-SVOTING for panels of 1, 3, 5 and 7 judges: votes or summed confidences, selected pipeline and final label per claim. The file `abstention_summary.csv` gives overall accuracy and abstentions |
| `results/pip_ranking/` | ILORA scores of every justification (Fig. 8) |

These files contain the per-claim outputs behind the EmeraldMindX results of the paper.

## License

The code is released under the Apache License 2.0 (`LICENSE`). The datasets are released under the licenses listed above. `emerald_promise.csv` contains annotations from ML-Promise (`original_text`, `page_number`, `promise_status`, `verification_timeline`, `evidence_status`, `evidence_quality`, `source_url`) and is therefore released under CC BY-NC-SA 4.0, for non-commercial use only. Please cite ML-Promise (Seki et al., EMNLP 2025) when using it.
