import os
import pandas as pd
import google.generativeai as genai
import tiktoken
import time
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Load Gemini API keys from environment variables
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]


# Function to approximate token count using tiktoken
def approximate_token_count(text: str):
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Token count approximation failed: {e}")
        return 0


current_key_index = 0


def call_llm(full_prompt):
    global current_key_index

    token_count = approximate_token_count(full_prompt)
    model_name = "gemini-2.5-flash" if token_count < 250000 else "gemini-2.0-flash"
    total_keys = len(API_KEYS)
    attempts = 0

    while attempts < total_keys:
        key = API_KEYS[current_key_index]
        if not key:
            print(f"[Key #{current_key_index + 1}] API key is missing or empty.")
        else:
            try:
                time.sleep(2)  # Respect rate limits
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)

                response = model.generate_content(
                    full_prompt, generation_config={"temperature": 0.0}
                )
                if response and hasattr(response, "text"):
                    print(f"[Key #{current_key_index + 1}] Success")
                    return response.text.strip(), model_name
            except Exception as e:
                print(f"[Key #{current_key_index + 1}] LLM call failed: {e}")

        current_key_index = (current_key_index + 1) % total_keys
        attempts += 1

    raise ValueError("All API keys are exhausted")


base_prompt = """
You are a fact-focused assistant tasked with generating KPI-related ESG striclty numerical claims from a news article.
 
**Instructions:**
 
1. You will be given a **news article text**. Only use information explicitly stated or clearly implied in the article. **Do not hallucinate or invent facts.**
 
2. Generate striclty numerical **synthetic claims** that are either:
   - **greenwashing** (claims that exaggerate or misrepresent ESG performance)  
   - **not_greenwashing** (claims that are accurate and supported by the article)
 
3. Each claim must:
   - Be **related to a KPI from the provided ESG definitions** (see below)  
   - Include the following fields:
     - `claim`: a short statement about ESG performance derived from the article.
     - `company`: the company that the claim refers to
     - `label`: `"greenwashing"` or `"not_greenwashing"` based on whether the claim is misleading or accurate.
     - `justification`: a concise explanation of why the claim is labeled greenwashing or not_greenwashing, strictly based on evidence from the article.
4. Only generate striclty numerical claims that **can be supported or contradicted by the article text**. Skip any KPIs that are not mentioned or cannot be linked to evidence in the article.
 
**ESG KPIs (JSON format):**
 
[
  {
    "id": "ESG 1‑1",
    "name": "Energy consumption, total",
    "definition": "Total amount of energy consumed by the organisation during the reporting period (Scope 1 + Scope 2).",
    "sector": ["Industrial Transportation","Banks","Nonlife Insurance","Automobiles","Electricity Utilities","None of the listed"]
  },
  {
    "id": "ESG 1‑2",
    "name": "Energy consumption intensity",
    "definition": "Energy consumed per unit of activity (options: per unit of revenue, per employee, per unit of production volume).",
    "sector": ["Industrial Transportation","Banks","Nonlife Insurance","Automobiles","Electricity Utilities","None of the listed"]
  },
  {
    "id": "ESG 2‑1",
    "name": "GHG emissions, total",
    "definition": "Total Scope 1 + Scope 2 greenhouse‑gas emissions during the reporting period.",
    "sector": ["Industrial Transportation","Banks","Nonlife Insurance","Automobiles","Electricity Utilities","None of the listed"]
  },
  {
    "id": "ESG 2‑2",
    "name": "GHG emissions intensity",
    "definition": "GHG emissions per unit of activity (options: per unit of revenue, per employee, per unit of production volume).",
    "sector": ["Industrial Transportation","Banks","Nonlife Insurance","Automobiles","Electricity Utilities","None of the listed"]
  },
  {
    "id": "ESG 10‑1",
    "name": "% of energy from renewable sources",
    "definition": "Share of total energy consumed that originates from renewable energy sources.",
    "sector": ["Industrial Transportation","Banks","Nonlife Insurance","Automobiles","Electricity Utilities"]
  },
  {
    "id": "ESG 10‑2",
    "name": "% of energy from combined heat and power (CHP)",
    "definition": "Share of total energy consumed that is generated through combined heat and power plants.",
    "sector": ["Industrial Transportation","Banks","Nonlife Insurance","Automobiles","Electricity Utilities"]
  },
  {
    "id": "ESG 10‑3",
    "name": "Investments in renewable energy generation as % of total investments",
    "definition": "Share of total investments that are directed to renewable‑energy generation projects.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 11",
    "name": "NOx / SOx emissions (total)",
    "definition": "Total nitrogen‑oxide (NOx) and sulphur‑oxide (SOx) emissions from operations.",
    "sector": ["Industrial Transportation","Electricity Utilities","Automobiles"]
  },
  {
    "id": "ESG 11‑1",
    "name": "NOx / SOx emissions – electricity utilities total",
    "definition": "Total NOx and SOx emissions from electricity‑generation activities.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 11‑2",
    "name": "NOx / SOx emissions – coal‑fired generation portfolio",
    "definition": "NOx and SOx emissions attributable to coal‑fired generation within the portfolio.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 11‑3",
    "name": "NOx / SOx emissions per kWh produced",
    "definition": "NOx and SOx emissions normalised by electricity output.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 11‑4",
    "name": "NOx / SOx emissions by passenger‑kilometre",
    "definition": "NOx and SOx emissions per passenger‑kilometre transported.",
    "sector": ["Industrial Transportation"]
  },
  {
    "id": "ESG 11‑5",
    "name": "NOx / SOx emissions by passenger‑mile",
    "definition": "NOx and SOx emissions per passenger‑mile transported.",
    "sector": ["Industrial Transportation"]
  },
  {
    "id": "ESG 11‑6",
    "name": "NOx / SOx emissions by cargo‑kilometre",
    "definition": "NOx and SOx emissions per cargo‑kilometre transported.",
    "sector": ["Industrial Transportation"]
  },
  {
    "id": "ESG 11‑7",
    "name": "NOx / SOx emissions by cargo‑mile",
    "definition": "NOx and SOx emissions per cargo‑mile transported.",
    "sector": ["Industrial Transportation"]
  },
  {
    "id": "ESG 11‑8",
    "name": "NOx / SOx emissions – total production sites",
    "definition": "Total NOx and SOx emissions from all automobile production sites.",
    "sector": ["Automobiles"]
  },
  {
    "id": "ESG 12",
    "name": "Waste generated (total)",
    "definition": "Total waste generated during the reporting period.",
    "sector": ["Industrial Transportation","Automobiles","Banks"]
  },
  {
    "id": "ESG 12‑1",
    "name": "Waste per unit produced",
    "definition": "Waste generated per unit of product or output.",
    "sector": ["Industrial Transportation","Automobiles","Banks"]
  },
  {
    "id": "ESG 12‑2",
    "name": "% of waste recycled",
    "definition": "Proportion of total generated waste that is recycled.",
    "sector": ["Automobiles","Banks"]
  },
  {
    "id": "ESG 13",
    "name": "Environmental compatibility (overall)",
    "definition": "Composite indicator covering multiple environmental‑compatibility aspects relevant to the sector.",
    "sector": ["Automobiles","Electricity Utilities"]
  },
  {
    "id": "ESG 13‑1",
    "name": "Average fuel consumption of sold car fleet",
    "definition": "Fleet‑weighted average fuel consumption for all vehicles sold in the reporting year.",
    "sector": ["Automobiles"]
  },
  {
    "id": "ESG 13‑2",
    "name": "Percentage of ISO 14001‑certified sites",
    "definition": "Share of production or corporate sites certified to ISO 14001 environmental‑management standard.",
    "sector": ["Automobiles"]
  },
  {
    "id": "ESG 13‑3",
    "name": "% of renewable energy produced",
    "definition": "Share of total electricity produced that originates from renewable sources.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 13‑4",
    "name": "Total renewable energy produced – biomass",
    "definition": "Total electricity generated from biomass‑based renewable sources.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 13‑5",
    "name": "Total renewable energy produced – wind",
    "definition": "Total electricity generated from wind‑based renewable sources.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 13‑6",
    "name": "Total renewable energy produced – hydro",
    "definition": "Total electricity generated from hydro‑based renewable sources.",
    "sector": ["Electricity Utilities"]
  },
  {
    "id": "ESG 13‑7",
    "name": "Percentage of revenues from eco‑labelled products",
    "definition": "Share of total revenues generated from products or services that carry recognised eco‑labels.",
    "sector": ["Electricity Utilities"]
  }
]
 
**Output format (JSON array):**
 
[
  {
    "claim": "text of the claim",
    "label": "greenwashing or not_greenwashing",
    "justification": "why the claim is greenwashing or not_greenwashing based on the article"
  }
]
 
**Important:** Only generate striclty numerical claims that are **directly supported or contradicted by the article text**.

**News Article:**

{article_text}
"""


# Function to parse LLM response and extract claims
def parse_llm_response(response_text: str, article_id: str = None) -> List[Dict]:

    try:
        # Try to find JSON array in the response
        start_idx = response_text.find("[")
        end_idx = response_text.rfind("]") + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            claims = json.loads(json_str)
            return claims
        else:
            print(f"No JSON array found in response for article {article_id}")
            print(f"Response preview: {response_text[:300]}...")
            return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for article {article_id}: {e}")
        print(f"Response preview: {response_text[:500]}...")

        # Save problematic response to file for debugging
        if article_id:
            error_file = f"claims_data/parse_error_{article_id}.txt"
            try:
                with open(error_file, "w") as f:
                    f.write(f"Article ID: {article_id}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Full Response:\n")
                    f.write(f"{'='*60}\n")
                    f.write(response_text)
                print(f"Saved problematic response to {error_file}")
            except Exception as save_error:
                print(f"Could not save error file: {save_error}")

        return []


# Function to save checkpoint
def save_checkpoint(
    claims: List[Dict],
    processed_ids: set,
    checkpoint_file: str,
    claims_file: str,
    llm_responses: List[Dict] = None,
):

    os.makedirs("claims_data", exist_ok=True)

    # Save claims to CSV
    if claims:
        claims_df = pd.DataFrame(claims)
        column_order = [
            "article_id",
            "claim",
            "company",
            "label",
            "justification",
            "model_used",
        ]
        claims_df = claims_df[column_order]
        claims_path = os.path.join("claims_data", claims_file)
        claims_df.to_csv(claims_path, index=False)
        print(f"Saved {len(claims)} claims to {claims_path}")

    # Save processed IDs to checkpoint
    checkpoint_path = os.path.join("claims_data", checkpoint_file)
    checkpoint_data = {
        "processed_ids": list(processed_ids),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Saved checkpoint with {len(processed_ids)} processed articles")

    # Save all LLM responses
    if llm_responses:
        responses_file = claims_file.replace(".csv", "_llm_responses.json")
        responses_path = os.path.join("claims_data", responses_file)
        with open(responses_path, "w") as f:
            json.dump(llm_responses, f, indent=2)
        print(f"Saved {len(llm_responses)} LLM responses to {responses_path}")


def load_checkpoint(checkpoint_file: str, claims_file: str):
    checkpoint_path = os.path.join("claims_data", checkpoint_file)
    claims_path = os.path.join("claims_data", claims_file)

    processed_ids = set()
    all_claims = []

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                processed_ids = set(checkpoint_data.get("processed_ids", []))
                timestamp = checkpoint_data.get("timestamp", "unknown")
                print(f"Loaded checkpoint from {timestamp}")
                print(f"Found {len(processed_ids)} previously processed articles")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    # Load existing claims
    if os.path.exists(claims_path):
        try:
            claims_df = pd.read_csv(claims_path)
            all_claims = claims_df.to_dict("records")
            print(f"Loaded {len(all_claims)} existing claims")
        except Exception as e:
            print(f"Warning: Could not load existing claims: {e}")

    return processed_ids, all_claims


# Main function to generate claims dataset
def generate_claims_dataset(
    input_csv: str, output_csv: str, checkpoint_interval: int = 5
):

    # Setup checkpoint file names
    checkpoint_file = output_csv.replace(".csv", "_checkpoint.json")

    # Load previous progress
    print("Checking for previous progress...")
    processed_ids, all_claims = load_checkpoint(checkpoint_file, output_csv)

    # Read input CSV
    print(f"\nReading input file: {input_csv}")
    input_path = os.path.join("claims_data", input_csv)
    df = pd.read_csv(input_path)

    # Verify required columns exist
    if "content" not in df.columns or "id" not in df.columns:
        raise ValueError("Input CSV must contain 'content' and 'id' columns")

    # Filter out already processed articles
    df_remaining = df[~df["id"].isin(processed_ids)]

    print(f"\nTotal articles: {len(df)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to process: {len(df_remaining)}")

    if len(df_remaining) == 0:
        print("\nAll articles already processed!")
        return pd.DataFrame(all_claims)

    # Process each remaining article
    articles_since_checkpoint = 0
    failed_articles = []  # Track articles that failed to parse

    for idx, row in df_remaining.iterrows():
        article_id = row["id"]
        article_content = row["content"]

        # Calculate actual progress
        current_num = len(processed_ids) + articles_since_checkpoint + 1
        total_articles = len(df)

        print(f"\n{'='*60}")
        print(f"Processing article {current_num}/{total_articles} (ID: {article_id})")
        print(f"{'='*60}")

        # Create full prompt with article content
        full_prompt = base_prompt + f"Article content: {article_content}"

        try:
            # Call LLM
            response_text, model_used = call_llm(full_prompt)
            print(f"Model used: {model_used}")

            # Parse response
            claims = parse_llm_response(response_text, article_id)

            if len(claims) == 0:
                print(f"No claims generated for article {article_id}")
                failed_articles.append(
                    {
                        "article_id": article_id,
                        "reason": "JSON parsing failed or no claims returned",
                    }
                )
            else:
                print(f"Generated {len(claims)} claims")

            # Add article_id to each claim
            for claim in claims:
                claim["article_id"] = article_id
                claim["model_used"] = model_used
                all_claims.append(claim)

            # Mark as processed (even if no claims were generated, to avoid re-processing)
            processed_ids.add(article_id)
            articles_since_checkpoint += 1

            # Save checkpoint at intervals
            if articles_since_checkpoint >= checkpoint_interval:
                print(f"\nSaving checkpoint...")
                save_checkpoint(all_claims, processed_ids, checkpoint_file, output_csv)
                articles_since_checkpoint = 0

        except ValueError as e:
            if "All API keys are exhausted" in str(e):
                print(f"\nAPI keys exhausted!")
                print(f"Saving progress before stopping...")
                save_checkpoint(all_claims, processed_ids, checkpoint_file, output_csv)

                # Save failed articles report
                if failed_articles:
                    failed_report_path = os.path.join(
                        "claims_data", "failed_articles.json"
                    )
                    with open(failed_report_path, "w") as f:
                        json.dump(failed_articles, f, indent=2)
                    print(f"Saved failed articles report to {failed_report_path}")

                print(f"\n{'='*60}")
                print(f"Progress saved! You can resume later.")
                print(f"Processed: {len(processed_ids)}/{len(df)} articles")
                print(f"Generated: {len(all_claims)} claims so far")
                print(f"Failed to parse: {len(failed_articles)} articles")
                print(f"{'='*60}")
                raise
        except Exception as e:
            print(f"Unexpected error processing article {article_id}: {e}")
            failed_articles.append(
                {"article_id": article_id, "reason": f"Unexpected error: {str(e)}"}
            )
            # Mark as processed to avoid infinite retry loop
            processed_ids.add(article_id)
            continue

    # Final save
    print(f"\nSaving final results...")
    save_checkpoint(all_claims, processed_ids, checkpoint_file, output_csv)

    # Save failed articles report
    if failed_articles:
        failed_report_path = os.path.join("claims_data", "failed_articles.json")
        with open(failed_report_path, "w") as f:
            json.dump(failed_articles, f, indent=2)
        print(f"Saved failed articles report: {len(failed_articles)} articles failed")
        print(f"Report location: {failed_report_path}")

    # Create final DataFrame
    claims_df = pd.DataFrame(all_claims)

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"Total articles processed: {len(processed_ids)}")
    print(f"Total claims generated: {len(claims_df)}")
    print(f"Articles with no claims: {len(failed_articles)}")
    print(f"Output saved to: claims_data/{output_csv}")
    print(f"{'='*60}")

    # Print summary statistics
    if len(claims_df) > 0:
        print("\nLabel distribution:")
        print(claims_df["label"].value_counts())

    return claims_df


if __name__ == "__main__":

    input_file = "articles.csv"  # Input CSV file with news articles
    output_file = "new_claims.csv"  # Output CSV file for claims

    try:
        claims_df = generate_claims_dataset(
            input_file, output_file, checkpoint_interval=5  # Save every 5 articles
        )
    except ValueError as e:
        if "All API keys are exhausted" in str(e):
            print("\nScript stopped due to API key exhaustion")
            print("Progress has been saved")
            print("Simply run the script again to resume from where you stopped")
        else:
            raise
