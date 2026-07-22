import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import subprocess
load_dotenv()

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

RATE_LIMIT_DELAY = 10 
SCHEMA_PATH = SCHEMA_DIR / "schema.json"
MAX_WORKERS = 8
SCHEMA_STR = None 

clients = [
    genai.Client(api_key=os.getenv(f"GEMINI_API_KEY_{i}"))
    for i in range(1, 9)
]

CFG_JSON = types.GenerateContentConfig(
    temperature=0.0,
    response_mime_type="application/json",
    system_instruction="Return *only* valid JSON – no prose.",
)


def build_prompt(company: str, claim: str) -> str:
    return (f"""You are an ESG knowledge-graph assistant.

        Given a company and a green claim, extract all possible nodes and relationships that strictly adhere to the following schema.

        Schema:
        {SCHEMA_STR}

        Input:
        company : {company}
        claim   : "{claim}"

        Do not include any explanatory text or markdown – only the JSON.
        
        Instructions:
        - Only use the keys and data types defined in the schema.
        - Do not include prose, explanation, or formatting outside the JSON.
        - Return valid JSON. If there are multiple nodes, return a JSON array.

        #### POSITIVE EXAMPLE ####

        nodes:
        [
            {{
                "class": "Organization",
                "properties": {{
                    "name": "Apple",
                    "industry": "electronics",
                    "valid_from": "2020-01-01",
                    "valid_to": null,
                    "is_current": true
                }}
            }},
            {{
                "class": "Product",
                "properties": {{
                    "name": "iPhone 12",
                    "description": "smartphone with improved energy efficiency",
                    "valid_from": "2020-10-23",
                    "valid_to": null,
                    "is_current": true
                }}
            }}
        ]

        edges:
        [
            {{
                "label": "producedBy",
                "source": 1,
                "target": 0,
                "temporal_properties": {{
                    "valid_from": "2020-10-23",
                    "valid_to": null,
                    "recorded_at": "2020-10-23"
                }}
            }}
        ]
        Now generate the JSON for the input above.
        """
    )


def extract_llm_json(response) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except Exception: 
        return str(response)



def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def process_single_row(row_data: Dict, output_dir: Path, client: genai.Client) -> Dict[str, Any]:
    idx = row_data['idx']
    record_id = row_data['unique_id']
    company = row_data['company']
    claim = row_data['claim']
    
    out_file = output_dir / f"claim_{record_id}.json"
    
    if out_file.exists():
        return {
            'status': 'skipped',
            'idx': idx,
            'record_id': record_id,
            'message': 'Already processed'
        }
    
    prompt = build_prompt(company, claim)
    time.sleep(RATE_LIMIT_DELAY)
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=CFG_JSON,
        )
    except Exception as err:
        return {
            'status': 'error',
            'idx': idx,
            'record_id': record_id,
            'message': f"LLM call failed: {err}"
        }

    json_text = extract_llm_json(resp)
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        return {
            'status': 'error',
            'idx': idx,
            'record_id': record_id,
            'message': f"Invalid JSON: {e}"
        }

    out_file.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    time.sleep(RATE_LIMIT_DELAY)
    return {
        'status': 'success',
        'idx': idx,
        'record_id': record_id,
        'file': rel_path(out_file)
    }


def process_dataset(df: pd.DataFrame, output_dir: Path, dataset_name: str, 
                    claim_col: str, company_col: str, year_col: str = None) -> None:
    df = df.reset_index(drop=True)
    df['unique_id'] = range(len(df))
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} - Total rows: {len(df)}")
    print(f"{'='*60}")
    rows_to_process = []
    for idx, row in df.iterrows():
        company = str(row[company_col]).strip()
        claim = str(row[claim_col]).strip()
        record_id = row['unique_id']
        rows_to_process.append({
            'idx': idx,
            'unique_id': record_id,
            'company': company,
            'claim': claim,
        })
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        client_idx = 0
        
        future_to_row = {}
        for row_data in rows_to_process:
            client = clients[client_idx % len(clients)]
            client_idx += 1
            future = executor.submit(process_single_row, row_data, output_dir, client)
            future_to_row[future] = row_data
        
        for future in as_completed(future_to_row):
            result = future.result()
            
            if result['status'] == 'success':
                processed_count += 1
                print(f"Saved {result['file']}")
            elif result['status'] == 'skipped':
                skipped_count += 1
                print(f"Skipping row {result['idx']} (unique_id: {result['record_id']}) - {result['message']}")
            elif result['status'] == 'error':
                error_count += 1
                print(f"Error row {result['idx']} (id: {result['record_id']}): {result['message']}")
    
    print(f"\n{dataset_name} Summary:")
    print(f"  - Processed: {processed_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Errors: {error_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract knowledge graph entities from ESG claims using Gemini API',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output directory for JSON files'
    )
    
    parser.add_argument(
        '-s', '--schema',
        type=Path,
        default=SCHEMA_PATH,
        help='Path to graph schema JSON file'
    )
    
    parser.add_argument(
        '--claim-col',
        type=str,
        default='claim',
        help='Name of the claim column in CSV'
    )
    
    parser.add_argument(
        '--company-col',
        type=str,
        default='company',
        help='Name of the company column in CSV'
    )
    
    parser.add_argument(
        '--year-col',
        type=str,
        default=None,
        help='Name of the year column in CSV (optional)'
    )
    
    return parser.parse_args()


def main() -> None:
    global SCHEMA_STR
    
    args = parse_args()
    try:
        SCHEMA_STR = json.dumps(json.loads(args.schema.read_text()), separators=(",", ":"))
    except FileNotFoundError:
        print(f"Schema file not found at {args.schema}")
        return
    except Exception as e:
        print(f"Error loading schema: {e}")
        return
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    missing_keys = [i for i in range(1, 9) if not os.getenv(f"GEMINI_API_KEY_{i}")]
    if missing_keys:
        print(f"Warning: Missing API keys for clients: {missing_keys}")
        print("Please ensure all GEMINI_API_KEY_1 through GEMINI_API_KEY_8 are set")
    
    print("\n" + "="*60)
    print(f"Processing: {args.input}")
    print("="*60)
    
    try:
        df = pd.read_csv(args.input)
        df.columns = df.columns.str.lower()
        
        claim_col = args.claim_col.lower()
        company_col = args.company_col.lower()
        year_col = args.year_col.lower() if args.year_col else None
        
        required_cols = [claim_col, company_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns: {', '.join(missing_cols)}")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        cols_to_keep = required_cols.copy()
        if year_col and year_col in df.columns:
            cols_to_keep.append(year_col)
        
        df = df[cols_to_keep]
        
        process_dataset(
            df=df,
            output_dir=args.output,
            dataset_name=args.input.name,
            claim_col=claim_col,
            company_col=company_col,
            year_col=year_col
        )
        
        print(f"\nProcessing complete. Output in {rel_path(args.output)}")
    
    except FileNotFoundError:
        print(f"Input file not found at {args.input}")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()