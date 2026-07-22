import pandas as pd
import glob
import os

# === Gather CSV files ===
ranking_files = glob.glob("your_pipeline_ranking_files.csv")
ranking_files = [f for f in ranking_files if "fixed" in f]

os.makedirs("results_hybrid_ranking", exist_ok=True)


def normalize_claim(claim: str) -> str:
    return str(claim).lower().strip()


file_mapping = {
    "ranking_emerald_few_shot_fixed": {
        "mode": "few_shot",
        "files": {
            "GraphRAG": "../results/emerald_few_one_hop_3_week.csv",
            "RAG": "../results/emerald_few_rag_week.csv",
            "Baseline": "../results/emerald_few_baseline_week.csv",
        },
    },
    "ranking_emerald_zero_shot_fixed": {
        "mode": "zero_shot",
        "files": {
            "GraphRAG": "../results/emerald_zero_one_hop_3_week.csv",
            "RAG": "../results/emerald_zero_rag_week.csv",
            "Baseline": "../results/emerald_zero_baseline_week.csv",
        },
    },
    "ranking_green_few_shot_fixed": {
        "mode": "few_shot",
        "files": {
            "GraphRAG": "../results/green_few_one_hop_3_week.csv",
            "RAG": "../results/green_few_rag_week.csv",
            "Baseline": "../results/green_few_baseline_week.csv",
        },
    },
    "ranking_green_zero_shot_fixed": {
        "mode": "zero_shot",
        "files": {
            "GraphRAG": "../results/green_zero_one_hop_3_week.csv",
            "RAG": "../results/green_zero_rag_week.csv",
            "Baseline": "../results/green_zero_baseline_week.csv",
        },
    },
}


for ranking_file in ranking_files:
    ranking_basename = os.path.basename(ranking_file).replace(".csv", "")

    if ranking_basename not in file_mapping:
        print(f"Skipping {ranking_file}, not in file_mapping")
        continue

    print(f"\nProcessing: {ranking_basename}")

    mode = file_mapping[ranking_basename]["mode"]
    pipeline_files = file_mapping[ranking_basename]["files"]

    ranking_df = pd.read_csv(ranking_file)

    results = []

    for idx, row in ranking_df.iterrows():
        claim = row["claim"]
        claim_norm = normalize_claim(claim)

        # Determine which pipeline to use
        selected_pipeline = row["rank_1st"]

        # If rank_1st is Baseline, use rank_2nd
        if selected_pipeline in ["Baseline", "baseline"]:
            selected_pipeline = row["rank_2nd"]
            print(f"rank_1st was Baseline, using rank_2nd: {selected_pipeline}")

        if selected_pipeline not in ["RAG", "GraphRAG"]:
            print(f"Unknown pipeline: {selected_pipeline}, skipping claim")
            raise Exception(f"Unknown pipeline selected: {selected_pipeline}")

        try:
            pipeline_file = pipeline_files[selected_pipeline]
            df_pipeline = pd.read_csv(pipeline_file)

            df_pipeline["claim_norm"] = df_pipeline["claim"].apply(normalize_claim)

            matched_rows = df_pipeline[df_pipeline["claim_norm"] == claim_norm]

            if matched_rows.empty:
                raise Exception(f"Claim not found in {pipeline_file}")

            matched_row = matched_rows.iloc[0]

            predicted_label = matched_row.get("predicted_label", None)
            predicted_justification = matched_row.get("predicted_justification", None)
            predicted_type = matched_row.get("predicted_type", None)
            ground_truth = matched_row.get("label", None)

            results.append(
                {
                    "claim": claim,
                    "label": ground_truth,
                    "selected_pipeline": selected_pipeline,
                    "predicted_label": predicted_label,
                    "predicted_justification": predicted_justification,
                    "predicted_type": predicted_type,
                }
            )

            print(
                f"Claim {idx+1}/{len(ranking_df)} - Pipeline: {selected_pipeline}, Label: {predicted_label}"
            )

        except Exception as e:
            print(f"Error processing claim: {e}")
            continue

    # Save results to CSV
    if results:
        result_df = pd.DataFrame(results)
        output_path = os.path.join(
            "results_hybrid_ranking", f"hybrid_{ranking_basename}.csv"
        )
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(results)} results to {output_path}")
    else:
        print(f"\nNo results to save for {ranking_basename}")
