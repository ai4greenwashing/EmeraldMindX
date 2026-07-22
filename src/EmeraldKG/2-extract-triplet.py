#!/usr/bin/env python3

import argparse
import json
import logging
import os
import pathlib
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from threading import Lock
from google import genai
from google.genai import types
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
DEBUG_DIR = "debug_outputs_per_page"

# Rate limiter class to track API calls per client
class RateLimiter:
    """Rate limiter that allows max 10 requests per minute per client"""
    def __init__(self, max_calls_per_minute: int = 10):
        self.max_calls = max_calls_per_minute
        self.call_times: Dict[int, deque] = {}
        self.locks: Dict[int, Lock] = {}
    
    def wait_if_needed(self, client_idx: int) -> None:
        """Wait if necessary to respect rate limit"""
        if client_idx not in self.call_times:
            self.call_times[client_idx] = deque()
            self.locks[client_idx] = Lock()
        
        with self.locks[client_idx]:
            now = time.time()
            calls = self.call_times[client_idx]
            
            # Remove calls older than 60 seconds
            while calls and now - calls[0] >= 60:
                calls.popleft()
            
            # If at limit, wait until oldest call is 60 seconds old
            if len(calls) >= self.max_calls:
                wait_time = 60 - (now - calls[0]) + 0.1  # Add small buffer
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s for client {client_idx}")
                    time.sleep(wait_time)
                    # Remove old calls after waiting
                    now = time.time()
                    while calls and now - calls[0] >= 60:
                        calls.popleft()
            
            # Record this call
            calls.append(time.time())


def ensure_debug_dir() -> None:
    pathlib.Path(DEBUG_DIR).mkdir(exist_ok=True)


def parse_company_year_from_filename(filename: str) -> Tuple[str, str]:
    """Extract company name and year from filename"""
    year_match = re.search(r'(\d{4})$', filename)
    if year_match:
        year = year_match.group(1)
        company = filename[:year_match.start()].rstrip('_-')
        return company, year
    
    parts = filename.split('_')
    if len(parts) >= 2:
        for i in range(len(parts) - 1, -1, -1):
            if re.match(r'^\d{4}$', parts[i]):
                return '_'.join(parts[:i]), parts[i]
    
    logger.warning(f"Could not parse company and year from: {filename}")
    return filename, "unknown"


def load_pages_from_text_dir(pdf_path: pathlib.Path, input_dir: pathlib.Path) -> List[Dict[str, str]]:
    """Load pages from text files"""
    company, year = parse_company_year_from_filename(pdf_path.stem)
    text_dir = input_dir / f"{company}_{year}_text"
    
    if not text_dir.exists():
        logger.error(f"Text directory not found: {text_dir}")
        return []
    
    pages = []
    txt_files = sorted(text_dir.glob(f"{company}_{year}_page_*.txt"))
    
    for txt_file in txt_files:
        match = re.search(r'page_(\d+)\.txt$', txt_file.name)
        if not match:
            logger.warning(f"Cannot parse page number from: {txt_file.name}")
            continue
        
        page_num = int(match.group(1)) - 1
        
        try:
            text = txt_file.read_text(encoding='utf-8')
            pages.append({"page": page_num, "text": text})
        except Exception as e:
            logger.error(f"Error reading {txt_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(pages)} pages from {text_dir}")
    return pages


def load_kpis_by_page(pdf_path: pathlib.Path, input_dir: pathlib.Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load KPIs for each page from the directory structure"""
    pdf_stem = pdf_path.stem
    kpi_dir = input_dir / f"{pdf_stem}_kpis"
    
    if not kpi_dir.exists():
        logger.warning(f"KPI directory not found: {kpi_dir}")
        return {}
    
    page_map: Dict[int, List[Dict[str, Any]]] = {}
    kpi_files = list(kpi_dir.glob("page_*_kpis.json"))
    
    for kpi_file in kpi_files:
        try:
            page_match = re.search(r'page_(\d+)_kpis\.json$', kpi_file.name)
            if not page_match:
                logger.warning(f"Cannot parse page number from: {kpi_file.name}")
                continue
                
            page_num = int(page_match.group(1)) - 1
            kpi_data = json.loads(kpi_file.read_text(encoding="utf-8"))
            
            if page_num not in page_map:
                page_map[page_num] = []
            
            if isinstance(kpi_data, list):
                page_map[page_num].extend(kpi_data)
            elif isinstance(kpi_data, dict):
                page_map[page_num].append(kpi_data)
                
        except Exception as e:
            logger.error(f"Error loading KPI file {kpi_file}: {e}")
            continue
    
    logger.info(f"Loaded KPIs for {len(page_map)} pages from {kpi_dir}")
    return page_map


def get_identity_keys(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract identity keys from schema for each node class"""
    identity_map = {}
    for node in schema.get("nodes", []):
        class_name = node.get("class")
        identity_keys = node.get("identity_keys", ["name"])
        identity_map[class_name] = identity_keys
    return identity_map


def get_stable_entity_id(entity: Dict[str, Any], identity_keys_map: Dict[str, List[str]]) -> str:
    """Generate a stable identifier for an entity based on its identity keys"""
    entity_class = entity.get("class", "Unknown")
    props = entity.get("properties", {})
    
    identity_keys = identity_keys_map.get(entity_class, ["name"])
    
    id_parts = [entity_class]
    for key in identity_keys:
        value = props.get(key, "")
        if isinstance(value, str):
            value = value.strip().lower()
        id_parts.append(str(value))
    
    return "|".join(id_parts)


def schema_sets(schema: Dict[str, Any]) -> Tuple[set[str], set[str]]:
    """Return (entity_classes, edge_labels)"""
    classes = {n["class"] for n in schema["nodes"]}
    edges = {e["label"] for e in schema["edges"]}
    return classes, edges


CFG_JSON = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    system_instruction="Return *only* valid JSON – no prose.",
)

TEMPORAL_GRAPH_PROMPT_TEMPLATE = (
    "You are an ESG temporal knowledge-graph extractor.\n\n"
    "## INPUTS\n"
    "• KNOWLEDGE GRAPH SCHEMA: list of entity classes, edge labels, and temporal properties (JSON).\n"
    "• documents: plain text from one ESG-related PDF page.\n"
    "• KPI records: optional JSON list for that page.\n\n"
    "## Task\n"
    "Extract **temporal** relations explicitly stated in the text.\n"
    "This is a TEMPORAL knowledge graph - you MUST include temporal properties for all nodes and edges.\n"
    "Obey the ontology below.\n\n"
    "──────────────────\n"
    "## KNOWLEDGE GRAPH SCHEMA\n"
    "──────────────────\n"
    "{schema_json}\n\n"
    "──────────────────\n"
    "## TEMPORAL EXTRACTION RULES\n"
    "──────────────────\n"
    "ALL nodes and edges MUST include temporal information:\n\n"
    "**For ALL Nodes:**\n"
    "• valid_from: The date when this information became valid (ISO format YYYY-MM-DD or YYYY)\n"
    "• valid_to: The date when this information was superseded (ISO format or null if current)\n"
    "• is_current: Boolean indicating if this is the current/latest version (true/false)\n\n"
    "**For ALL Edges (relationships):**\n"
    "Include these as additional properties in the temporal_metadata object:\n"
    "• valid_from: When this relationship started\n"
    "• valid_to: When this relationship ended (null if still active)\n"
    "• recorded_at: When this relationship was recorded/reported\n\n"
    "**Temporal Inference Rules:**\n"
    "1. If the text mentions a specific year (e.g., '2023 emissions'), set valid_from to that year\n"
    "2. If reporting year is {year}, and no end date is mentioned, set valid_to to null and is_current to true\n"
    "3. For historical data, set is_current to false\n"
    "4. For KPI observations, use the 'year' field as valid_from\n"
    "5. If no temporal info is explicit, infer from context (reporting year, document date, etc.)\n"
    "6. For organizational facts (like industry), if stated in a {year} report without historical context, use {year} as valid_from\n"
    "7. For time-bound observations (emissions, waste, KPIs), each year/period is a separate node version\n"
    "8. For entities (organizations, facilities), only create new versions when properties actually change\n\n"
    "**Entity Versioning:**\n"
    "• Observations (KPIObservation, Emission, Waste) are inherently time-bound - each is a unique node\n"
    "• Entities (Organization, Facility, Person) should be versioned only when their properties change)\n"
    "• Use 'supersedes' edges to link entity versions (newer version supersedes older version)\n"
    "• The newest version of an entity should have is_current=true, older versions is_current=false\n\n"
    "──────────────────\n"
    "## STRICT EXTRACTION RULES\n"
    "──────────────────\n"
    "Return a single JSON *array* of objects with keys:\n"
    "    subject  | predicate | object | temporal_metadata\n"
    "where:\n"
    "• predicate ∈ edge labels from schema.\n"
    "• subject.class & object.class ∈ entity classes from schema.\n"
    "• properties ⊆ declared keys for that class (INCLUDING valid_from, valid_to, is_current).\n"
    "• temporal_metadata contains edge temporal properties (valid_from, valid_to, recorded_at)\n"
    "Do not add extra keys, comments, or prose.\n\n"
    "-----------------\n"
    "POSITIVE EXAMPLE (valid temporal extraction)\n"
    "-----------------\n"
    "[{{\n"
    '  "subject": {{"class": "Organization", "properties": {{"name": "Acme Corp", "industry": "Textiles", '
    '"valid_from": "2020-01-01", "valid_to": null, "is_current": true}}}},\n'
    '  "predicate": "reportsKPI",\n'
    '  "object": {{"class": "KPIObservation", "properties": {{"kpi_type": "ESG-1-1", "title": "Total energy consumed", '
    '"value": 42.7, "unit": "MWh", "kind": "achieved", "direction": "reduction", "year": 2023, "target_year": null, '
    '"baseline_year": 2020, "source_id": "acme_2023.pdf_1_2", "company": "acme", '
    '"valid_from": "2023-01-01", "valid_to": "2023-12-31", "is_current": false}}}},\n'
    '  "temporal_metadata": {{"valid_from": "2023-01-01", "valid_to": null, "recorded_at": "{year}-01-01"}}\n'
    "}}]\n\n"
    "-----------------\n"
    "BEGIN EXTRACTION\n"
    "-----------------\n"
    "Extract temporal triples from the following text **and output only the JSON array**.\n\n"
    "──────────────────\n"
    "COMPANY NAME: {company}\n"
    "REPORTING YEAR: {year}\n"
    "──────────────────\n\n"
    "Output a valid JSON array, or an empty array [] if nothing found.\n\n"
)


def triple_list_to_graph(triples: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    node_index: Dict[str, int] = {}
    identity_keys_map = get_identity_keys(schema)
    edges: List[Dict[str, Any]] = []

    def _idx(entity: Dict[str, Any]) -> Optional[int]:
        if not isinstance(entity, dict) or "class" not in entity or "properties" not in entity:
            return None
        
        stable_id = get_stable_entity_id(entity, identity_keys_map)
        props = entity["properties"]
        entity_class = entity["class"]
        
        observation_classes = {"KPIObservation", "Emission", "Waste"}
        if entity_class in observation_classes:
            version_key = f"{stable_id}|{json.dumps(props, sort_keys=True)}"
        else:
            valid_from = props.get("valid_from", "")
            valid_to = props.get("valid_to", "")
            version_key = f"{stable_id}|{valid_from}|{valid_to}"
        
        if version_key not in node_index:
            node_index[version_key] = len(nodes)
            nodes.append({
                "class": entity["class"],
                "properties": entity["properties"],
                "stable_id": stable_id
            })
        return node_index[version_key]

    for t in triples:
        if not {"subject", "predicate", "object"} <= t.keys():
            continue
        s_idx = _idx(t["subject"])
        o_idx = _idx(t["object"])
        if s_idx is None or o_idx is None:
            continue
        
        edge = {"subject": s_idx, "predicate": t["predicate"], "object": o_idx}
        if "temporal_metadata" in t:
            edge["temporal_metadata"] = t["temporal_metadata"]
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def build_page_prompt(schema, page_text, page_no, page_kpis, company, year):
    header = TEMPORAL_GRAPH_PROMPT_TEMPLATE.format(
        schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
        company=company,
        year=year
    )
    kpi_section = (
        f"--- KPI OBSERVATIONS (page {page_no}) ---\n```json\n"
        f"{json.dumps(page_kpis, indent=2, ensure_ascii=False)}\n```\n\n"
        if page_kpis else ""
    )
    return f"{header}\n\n--- DOC page {page_no} ---\n\n{page_text}\n\n{kpi_section}"


def _response_to_text(resp) -> str:
    if isinstance(resp, genai.types.GenerateContentResponse):
        buf: list[str] = []
        for cand in resp.candidates or []:
            for part in cand.content.parts or []:
                txt = getattr(part, "text", None)
                if txt:
                    buf.append(txt)
        return "\n".join(buf)
    return str(resp)


def _clean_json_response(resp) -> str:
    text = _response_to_text(resp).strip()
    
    if not text:
        return ""
    
    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Remove common preambles
    if text.startswith("Here") or text.lower().startswith("i'll"):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('[') or line.strip().startswith('{'):
                text = '\n'.join(lines[i:])
                break

    # Find JSON boundaries
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        return text[start:end]

    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end] if start != -1 and end > start else ""


def _parse_json_response(raw_response: str) -> Tuple[Union[Dict, List, str], bool]:
    """Parse JSON response with robust error handling and recovery"""
    cleaned_response = _clean_json_response(raw_response)
    
    if not cleaned_response:
        logger.warning("Empty response after cleaning")
        return [], False

    # Remove trailing commas
    cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
    
    # Remove comments
    cleaned_response = re.sub(r'//.*?$', '', cleaned_response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'/\*.*?\*/', '', cleaned_response, flags=re.DOTALL)

    try:
        parsed = json.loads(cleaned_response)
        return parsed, True
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        logger.debug(f"Problematic JSON (first 500 chars): {cleaned_response[:500]}")
        
        try:
            # Fix single quotes to double quotes
            fixed = cleaned_response.replace("'", '"')
            # Fix unquoted keys (basic attempt)
            fixed = re.sub(r'(\w+):', r'"\1":', fixed)
            parsed = json.loads(fixed)
            logger.info("Recovered JSON with fixes")
            return parsed, True
        except:
            logger.error("Could not recover JSON")
            return [], False


def _validate_extraction_format(data: Any, schema) -> bool:
    if not isinstance(data, list):
        logger.warning(f"Expected list, got {type(data)}")
        return False

    entities, edge_labels = schema_sets(schema)
    valid_count = 0
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"Item {i}: not a dict")
            continue

        required_keys = {"subject", "predicate", "object"}
        if not required_keys.issubset(item.keys()):
            logger.warning(f"Item {i}: missing required keys")
            continue

        valid_item = True
        for entity_key in ["subject", "object"]:
            entity = item[entity_key]
            if not isinstance(entity, dict):
                logger.warning(f"Item {i}: {entity_key} not a dict")
                valid_item = False
                break
            if not {"class", "properties"}.issubset(entity.keys()):
                logger.warning(f"Item {i}: {entity_key} missing class or properties")
                valid_item = False
                break
            if not isinstance(entity["properties"], dict):
                logger.warning(f"Item {i}: {entity_key} properties not a dict")
                valid_item = False
                break
            if entity["class"] not in entities:
                logger.warning(f"Item {i}: {entity_key} class '{entity['class']}' not in schema")
                valid_item = False
                break
        
        if not valid_item:
            continue
            
        # Validate prdicate
        if item["predicate"] not in edge_labels:
            logger.warning(f"Item {i}: predicate '{item['predicate']}' not in schema")
            continue
            
        valid_count += 1
    
    logger.info(f"Validated {valid_count}/{len(data)} triples")
    return valid_count > 0


def company_from_pdf(pdf_path: pathlib.Path) -> str:
    company, _ = parse_company_year_from_filename(pdf_path.stem)
    return company


def year_from_pdf(pdf_path: pathlib.Path) -> int:
    _, year_str = parse_company_year_from_filename(pdf_path.stem)
    try:
        return int(year_str)
    except ValueError:
        logger.warning(f"Could not extract year from {pdf_path.name}, defaulting to 2024")
        return 2024


def create_clients():
    clients = []
    for i in range(1, 7):
        api_key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not api_key:
            raise ValueError(f"GEMINI_API_KEY_{i} not found in environment")
        clients.append(genai.Client(api_key=api_key))
    logger.info(f"Created {len(clients)} Gemini clients")
    return clients


def call_llm(
    prompt: str, 
    client, 
    client_idx: int,
    rate_limiter: RateLimiter,
    schema: Dict[str, Any], 
    retries: int = 3
) -> Tuple[Any, str, bool]:
    last_error = None
    last_raw_text = ""
    rate_limit_failures = 0

    for attempt in range(1, retries + 1):
        try:
            # Wait if needed to respect rate limit
            rate_limiter.wait_if_needed(client_idx)
            
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=CFG_JSON,
            )

            raw_txt = _response_to_text(resp)
            last_raw_text = raw_txt
            parsed, is_valid = _parse_json_response(raw_txt)

            if is_valid:
                if _validate_extraction_format(parsed, schema):
                    logger.info(f"Extracted {len(parsed)} relations")
                    return parsed, raw_txt, False
                else:
                    logger.warning(f"Attempt {attempt}: valid JSON but format issues")
                    return parsed, raw_txt, False
            else:
                logger.warning(f"Attempt {attempt}: could not parse valid JSON")

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate" in error_str or "quota" in error_str or "429" in error_str:
                rate_limit_failures += 1
                logger.warning(f"Attempt {attempt} - Rate limit hit for client {client_idx}: {e}")
            else:
                logger.error(f"Attempt {attempt} failed: {e}")

        if attempt < retries:
            wait_time = 2 ** (attempt - 1)
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    if rate_limit_failures == retries:
        logger.error(f"All {retries} attempts failed due to rate limiting for client {client_idx}")
        return [], last_raw_text, True
    
    logger.error(f"All {retries} attempts failed. Last error: {last_error}")
    return [], last_raw_text, False


def process_page(
    page_data: Dict[str, Any],
    client,
    client_idx: int,
    rate_limiter: RateLimiter,
    schema: Dict[str, Any],
    pdf_stem: str,
    dbg_pdf_dir: pathlib.Path,
    g_pdf_dir: pathlib.Path,
    company: str,
    year: int
) -> Tuple[int, bool, bool]:
    """
    Process a single page. 
    Returns (page_number, success, rate_limited)
    """
    pg = page_data["page_info"]
    page_kpis = page_data["kpis"]
    p_no = pg["page"]
    
    output_file = g_pdf_dir / f"page{p_no}.json"
    bugged_file = g_pdf_dir / f"page{p_no}_bugged.json"
    
    if output_file.exists():
        logger.info(f"Skipping page {p_no} (already exists)")
        return p_no, True, False
    
    logger.info(f"→ Processing page {p_no} with client {client_idx}...")
    
    prompt = build_page_prompt(schema, pg["text"], p_no, page_kpis, company=company, year=year)
    
    max_retries = 2
    for retry in range(max_retries):
        parsed, raw, rate_limited = call_llm(prompt, client, client_idx, rate_limiter, schema, retries=2)
        
        # If rate limited, skip this page entirely
        if rate_limited:
            logger.warning(f"Page {p_no} skipped due to rate limiting on client {client_idx}")
            return p_no, False, True
        
        # Always save debug output
        dbg_path = dbg_pdf_dir / f"{pdf_stem}_p{p_no}.txt"
        dbg_path.write_text(
            f"==== PROMPT ====\n{prompt[:2000]}...\n\n==== RESPONSE ====\n{raw or '[NO RESPONSE]'}",
            encoding="utf-8",
        )
        
        if raw:
            # Convert to graph
            if isinstance(parsed, list) and parsed:
                # Validate triples
                entities, edge_labels = schema_sets(schema)
                valid_triples = []
                invalid_triples = []
                
                for triple in parsed:
                    # Check basic structure
                    if not isinstance(triple, dict):
                        invalid_triples.append(triple)
                        continue
                    if not {"subject", "predicate", "object"}.issubset(triple.keys()):
                        invalid_triples.append(triple)
                        continue
                    
                    # Check entities
                    valid = True
                    for entity_key in ["subject", "object"]:
                        entity = triple.get(entity_key, {})
                        if not isinstance(entity, dict):
                            valid = False
                            break
                        if "class" not in entity or "properties" not in entity:
                            valid = False
                            break
                        if entity["class"] not in entities:
                            valid = False
                            break
                    
                    # Check predicate
                    if triple.get("predicate") not in edge_labels:
                        valid = False
                    
                    if valid:
                        valid_triples.append(triple)
                    else:
                        invalid_triples.append(triple)
                
                # Save bugged triples if any
                if invalid_triples:
                    logger.warning(f"Page {p_no}: {len(invalid_triples)} invalid triples saved to bugged file")
                    bugged_file.write_text(
                        json.dumps(invalid_triples, indent=2, ensure_ascii=False),
                        "utf-8"
                    )
                
                # Convert valid triples to graph
                if valid_triples:
                    graph = triple_list_to_graph(valid_triples, schema)
                else:
                    graph = {"nodes": [], "edges": []}
                
                # Save valid graph
                output_file.write_text(
                    json.dumps(graph, indent=2, ensure_ascii=False),
                    "utf-8"
                )
                
                logger.info(f"Page {p_no}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
                return p_no, True, False
            
            else:
                # Malformed JSON - save as text file for manual fixing
                malformed_file = g_pdf_dir / f"page{p_no}_malformed.txt"
                malformed_content = (
                    f"Company: {company}\n"
                    f"Year: {year}\n"
                    f"Page: {p_no}\n\n"
                    f"==== MALFORMED RESPONSE ====\n"
                    f"{raw}\n\n"
                    f"==== END MALFORMED RESPONSE ====\n"
                )
                malformed_file.write_text(malformed_content, encoding="utf-8")
                logger.warning(f"Page {p_no}: Malformed JSON saved to {malformed_file.name}")
        
        logger.warning(f"Page {p_no} LLM call failed, retry {retry+1}/{max_retries}")
        time.sleep(2)
    
    logger.error(f"𝕏 Page {p_no} failed after {max_retries} retries")
    # DO NOT save empty graph - leave the file missing so it can be retried later
    return p_no, False, False


def process_single_pdf_parallel(
    pdf: pathlib.Path,
    schema: dict,
    out_dir: pathlib.Path,
    input_dir: pathlib.Path,
    max_workers: int = 7,
) -> None:
    """Process a single PDF with parallel page processing and rate limiting"""
    clients = create_clients()
    rate_limiter = RateLimiter(max_calls_per_minute=10)
    
    pages = load_pages_from_text_dir(pdf, input_dir)
    page_kpi_map = load_kpis_by_page(pdf, input_dir)
    
    if not pages:
        logger.error(f"No pages loaded for {pdf.name}")
        return
    
    company = company_from_pdf(pdf)
    year = year_from_pdf(pdf)
    logger.info(f"═══ Processing {pdf.name} - {company} ({year}) ═══")
    
    dbg_pdf_dir = pathlib.Path(DEBUG_DIR) / pdf.stem
    dbg_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    g_dir = out_dir / "graphs"
    g_dir.mkdir(parents=True, exist_ok=True)
    
    g_pdf_dir = g_dir / pdf.stem
    g_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    page_tasks = []
    for pg in pages:
        page_no = pg["page"]
        client_idx = page_no % len(clients)
        page_tasks.append({
            "page_info": pg,
            "kpis": page_kpi_map.get(page_no, []),
            "client": clients[client_idx],
            "client_idx": client_idx
        })
    
    logger.info(f"Processing {len(page_tasks)} pages with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_page,
                page_data,
                page_data["client"],
                page_data["client_idx"],
                rate_limiter,
                schema,
                pdf.stem,
                dbg_pdf_dir,
                g_pdf_dir,
                company,
                year
            ): page_data["page_info"]["page"]
            for page_data in page_tasks
        }
        
        completed = 0
        failed = 0
        rate_limited = 0
        
        for future in as_completed(futures):
            page_no = futures[future]
            try:
                result_page, success, was_rate_limited = future.result()
                completed += 1
                if not success:
                    failed += 1
                    if was_rate_limited:
                        rate_limited += 1
            except Exception as exc:
                failed += 1
                logger.error(f"Page {page_no} exception: {exc}")
    
    success_count = completed - failed
    logger.info(f"═══ Finished {pdf.name}: {success_count}/{completed} successful ═══")
    if rate_limited > 0:
        logger.warning(f"{rate_limited} pages skipped due to rate limiting")
    logger.info("")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract temporal ESG graphs from PDFs")
    ap.add_argument("--input_dir", required=True, help="Directory with PDFs, text, and KPI subdirs")
    ap.add_argument("--out_dir", default="temporal_graphs", help="Output directory")
    ap.add_argument("--limit", type=int, help="Process at most N PDFs")
    ap.add_argument("--workers", type=int, default=7, help="Parallel workers (default: 6)")
    ap.add_argument("--schema", required=True, help="Path to graph schema JSON file")
    args = ap.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    pdfs = sorted(input_dir.glob("*.pdf"))[: args.limit or None]

    if not pdfs:
        raise SystemExit(f"No PDFs found in {input_dir}")

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    schema_file = pathlib.Path(args.schema)
    if not schema_file.exists():
        raise SystemExit("graph_schema.json not found")

    schema = json.loads(schema_file.read_text("utf-8"))
    
    for i in range(1, 8):
        if not os.getenv(f"GEMINI_API_KEY_{i}"):
            raise SystemExit(f"GEMINI_API_KEY_{i} not found in environment")

    logger.info(f"Starting processing of {len(pdfs)} PDFs")
    
    for pdf in pdfs:
        process_single_pdf_parallel(pdf, schema, out, input_dir, args.workers)

    logger.info("All PDFs processed successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
