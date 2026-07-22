#!/usr/bin/env python3

import os, re, json, argparse
import glob
from logging import getLogger, basicConfig, INFO
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
import json, re, textwrap
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import concurrent.futures
import time
import subprocess
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)


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
REPORTS_DIR = PROJECT_ROOT / "reports"
# Regex patterns
CUE_REGEX = re.compile(
    r"\b("
    r"achiev\w*|reached|delivered|actual(?:ly)?|reduc\w*|increase\w*|cut|"
    r"goal|target|aim|pledge|commit\w*|baseline|base\s+year|since|from"
    r")\b",
    flags=re.I,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.M)

home = Path.home()

def _clean_json(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty LLM reply")
    raw = FENCE_RE.sub("", raw).strip()
    start_positions = [pos for pos in (raw.find("["), raw.find("{")) if pos != -1]
    if start_positions:
        start = min(start_positions)
        raw = raw[start:]
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    return json.loads(raw)

def get_embedder():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    return model

def normalize_kpi_response(data: List[Dict]) -> List[Dict]:
    for item in data:
        for obs in item.get("observations", []):
            val = obs.get("value")
            if isinstance(val, str):
                if val.endswith("%"):
                    obs["unit"] = obs.get("unit") or "%"
                    val = val.rstrip("%")
                try:
                    obs["value"] = float(val)
                except ValueError:
                    pass

            yr = obs.get("year")
            if isinstance(yr, str) and yr.isdigit():
                obs["year"] = int(yr)
            yr = obs.get("baseline_year")
            if isinstance(yr, str) and yr.isdigit():
                obs["baseline_year"] = int(yr)
            yr = obs.get("target_year")
            if isinstance(yr, str) and yr.isdigit():
                obs["target_year"] = int(yr)
    return data


def extract_pdf_text_by_page(pdf_path: str) -> List[str]:
    pages_text = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text()
                pages_text.append(text)
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1} from {pdf_path}: {e}")
                pages_text.append("")
        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
    
    return pages_text

def save_page_text(page_text: str, company: str, year: str, page_num: int, reports_dir: str) -> str:
    text_subdir = Path(reports_dir) / f"{company}_{year}_text"
    text_subdir.mkdir(exist_ok=True)
    
    txt_filename = f"{company}_{year}_page_{page_num}.txt"
    txt_path = text_subdir / txt_filename
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(page_text)
        logger.info(f"Saved page {page_num} text to {txt_path}")
        return str(txt_path)
    except Exception as e:
        logger.warning(f"Error saving page text to {txt_path}: {e}")
        return ""

def find_txt_file_for_page(company: str, year: str, page_num: int, reports_dir: str) -> Optional[str]:
    images_dir_pattern = f"{company}*_{year}_*images"
    txt_file_pattern = f"{company}*_{year}_*page*_{page_num}.txt"
    
    images_dirs = glob.glob(os.path.join(reports_dir, images_dir_pattern))
    
    for images_dir in images_dirs:
        txt_files = glob.glob(os.path.join(images_dir, txt_file_pattern))
        if txt_files:
            return txt_files[0]
    
    return None

def read_txt_file(txt_path: str) -> str:
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Error reading {txt_path}: {e}")
        return ""

def parse_company_year_from_filename(pdf_filename: str) -> Tuple[str, str]:
    """Extract company name and year from PDF filename"""
    # Pattern: {company}_*{year}.pdf
    basename = os.path.splitext(pdf_filename)[0] 
    
    year_match = re.search(r'(\d{4})$', basename)
    if year_match:
        year = year_match.group(1)
        company_part = basename[:year_match.start()].rstrip('_-')
        company = re.sub(r'[_\-\s]+$', '', company_part)
        return company, year
    
    # If no clear year pattern, try different approach
    parts = basename.split('_')
    if len(parts) >= 2:
        for i in range(len(parts) - 1, -1, -1):
            if re.match(r'^\d{4}$', parts[i]):
                year = parts[i]
                company = '_'.join(parts[:i])
                return company, year
    
    logger.warning(f"Could not parse company and year from filename: {pdf_filename}")
    return os.path.splitext(pdf_filename)[0], "unknown"

class RateLimitError(Exception):
    pass

class KPIExtractor:
    def __init__(self, kpi_defs_path: str = "./kpi_definitions.json", max_retries: int = 3, retry_delay: float = 2.0):
        load_dotenv()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.llms = []

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        for i in range(1, 8):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        google_api_key=key,
                    )
                    self.llms.append(llm)
                    logger.info(f"Initialized Gemini LLM with GEMINI_API_KEY_{i}")
                except Exception as e:
                    logger.warning(f"Failed to initialize GEMINI_API_KEY_{i}: {e}")

        if not self.llms:
            logger.warning("No GEMINI_API_KEY_1..7 found; " \
             "initializing single LLM with GEMINI_API_KEY_1 (may be None).")
            self.llms = [
                ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    google_api_key=os.getenv("GEMINI_API_KEY_1"),
                )
            ]
        self.n_llms = len(self.llms)

        with open(kpi_defs_path, "r", encoding="utf-8") as f:
            self.all_kpi_defs = json.load(f)
        
        self.embedder = get_embedder()
        self._indices_by_sector = {}
        
        logger.info("KPIExtractor initialized successfully")

    def detect_company_and_sector(self, company: str, text: str) -> Tuple[str, str]:
        SECTORS = [
            "Industrial Transportation",
            "Banks",
            "Nonlife Insurance",
            "Automobiles",
            "Electricity Utilities",
            "None of the listed",
        ]
        schema = '{"company name": "string", "sector": "string"}'
        sector_list = "\n".join(f"- {s}" for s in SECTORS)

        prompt = (
            "Extract the legal company name **exactly as written** and classify it.\n"
            f"Company Name: \n{company}\n\n"
            f"Choose the sector strictly from this list:\n{sector_list}\n\n"
            f'TEXT SOURCE (may be partial):\n"""\n{text}\n"""\n\n'
            f"Respond ONLY with a JSON object that follows this schema:\n{schema}"
        )

        try:
            raw = self.llms[0].invoke(
                [("system", "Company sector extractor"), ("human", prompt)]
            ).content
            reply = raw.strip()
            logger.info("LLM reply: %s", reply[:200])

            try:
                if reply.startswith("```"):
                    reply = reply.lstrip("`").split("```")[0]
                data = json.loads(reply)
                sector = str(data.get("sector", "")).strip()
            except (json.JSONDecodeError, TypeError, ValueError):
                sector_match = re.search(r'"?sector"?\s*:\s*"([^"]+)"', reply, re.I)
                sector = sector_match.group(1).strip() if sector_match else ""

            sector_norm = sector.casefold()
            sector = next(
                (s for s in SECTORS if s.casefold() == sector_norm),
                "None of the listed",
            )

            return company, sector

        except Exception as e:
            logger.warning(f"Sector detection failed: {e}")
            return "Unknown company", "None of the listed"

    def get_sector_view(self, sector: str) -> Tuple[List[Dict], Any]:
        sector_key = sector or "None of the listed"

        if sector_key in self._indices_by_sector:
            return self._indices_by_sector[sector_key]

        # Filter KPI definitions by sector
        kpi_defs = [
            d for d in self.all_kpi_defs
            if sector_key in d.get("sector", []) or "None of the listed" in d.get("sector", [])
        ]
        if not kpi_defs:
            kpi_defs = self.all_kpi_defs

        sentences = [
            f"{d['id']} – {d['name']} – {d.get('definition', '')}" for d in kpi_defs
        ]
        emb = self.embedder.encode(sentences, normalize_embeddings=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        self._indices_by_sector[sector_key] = (kpi_defs, index)
        return kpi_defs, index

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit',
            'quota exceeded',
            'too many requests',
            '429',
            'resource exhausted',
            'rate_limit_exceeded'
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def page_llm_extract(self, page_text: str, figure_desc: str, kpi_subset: List[Dict],
                        company: str, sector: str, page_num: int, doc_name: str,
                        llm_client: Optional[Any] = None) -> List[Dict]:
        llm = llm_client or (self.llms[0] if self.llms else None)
        
        if llm is None:
            logger.error("No LLM client available for page extraction")
            raise RateLimitError("No LLM client available")
        
        extract_text = page_text
        if figure_desc is None:
            figure_desc = "No figure description available."
        
        schema = textwrap.dedent("""
            [
              {
                "kpi_type"      : "string",
                "title"         : "string",
                "observations"  : [
                  {
                    "value"         : "number",
                    "unit"          : "string",
                    "kind"          : "baseline | target | achieved | projection",
                    "direction"     : "absolute | reduction | increase",
                    "year"          : "integer | null",
                    "target_year"   : "integer | null",
                    "baseline_year" : "integer | null",
                    "source_id"     : "{base_doc_name}_{page_num}_{ascending_index}",
                    "snippet"       : "string (≤160 chars)"
                  }
                ],
                "page"     : "integer",
                "doc_name" : "string",
                "company"  : "string",
                "sector"   : "string"
              }
            ]
        """).strip()

        defs = "\n".join(f"{d['id']}: {d.get('definition', '')}" for d in kpi_subset)

        prompt = textwrap.dedent(f"""
            <system>
            You are ESG-KPI-EXTRACTOR-V2. Produce only JSON conforming exactly
            to the schema below. If no KPI can be unambiguously extracted, output [].
            Note: Text source is enhanced at the end with figure captions and descriptions as JSONs from an improved extractor,
            so there may exist duplicates. Keep one entry, trusting the figure captions and descriptions more.
            If there is a discrepancy between the text and the figure descriptions, prefer the figure descriptions.
            
            For each extracted KPI, set:
            - company: "{company}"
            - sector: "{sector}"  
            - page: {page_num}
            - doc_name: "{doc_name}"
            </system>

            **Schema**  
            {schema}

            **Classification rules**
            • baseline  – historic reference (keywords: since, baseline, base year)  
            • target    – ambition or commitment (keywords: goal, target, aim)  
            • achieved  – result already met (keywords: achieved, delivered, reduced)  
            • projection – future estimate not yet committed

            If a sentence contains several numbers for the *same* KPI, create
            separate observation objects.

            If  a metric does not fully match any KPI definition, do not use the KPI types in the definitions.  
            Instead, use a descriptive title and set kpi_type to "other".

            If a set of observations come from a single figure, assign them the same source_id. 

            Do **not** wrap the JSON in Markdown fences or code tags.  
            Return [] rather than guess.

            **KPI_DEFINITIONS (subset)**  
            {defs}

            **TEXT SOURCE**  
            \"\"\"{extract_text}\"\"\"

            **FIGURE DESCRIPTIONS**
            \"\"\"{figure_desc}\"\"\"
        """).strip()

        for attempt in range(self.max_retries):
            try:
                raw = llm.invoke(
                    [("system", "KPI extractor"), ("human", prompt)]
                ).content
                raw = raw.strip()
                logger.info("RAW LLM reply: %r", raw[:300])
                data = normalize_kpi_response(_clean_json(raw))
                return data
            except Exception as err:
                is_rate_limit = self._is_rate_limit_error(err)
                
                if is_rate_limit:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit on attempt {attempt + 1}/{self.max_retries}. "
                                     f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit exceeded after {self.max_retries} attempts. Skipping page {page_num}.")
                        raise RateLimitError(f"Rate limit exceeded after {self.max_retries} attempts")
                else:
                    logger.warning(f"page_llm_extract error (attempt {attempt + 1}/{self.max_retries}): {err}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
        
        return []

    def process_pdf_report_with_page_outputs(self, pdf_path: str, reports_dir: str, max_workers: Optional[int] = None) -> int:
        pdf_filename = os.path.basename(pdf_path)
        pdf_name_no_ext = os.path.splitext(pdf_filename)[0]
        company_from_filename, year = parse_company_year_from_filename(pdf_filename)

        logger.info(f"Processing {pdf_filename} (Company: {company_from_filename}, Year: {year})")

        output_subdir = Path(reports_dir) / f"{pdf_name_no_ext}_kpis"
        output_subdir.mkdir(exist_ok=True)

        pdf_pages = extract_pdf_text_by_page(pdf_path)
        if not pdf_pages:
            logger.warning(f"No text extracted from {pdf_path}")
            return 0

        # Detect company and sector from first few pages
        sample_text = " ".join(pdf_pages[:3])[:2000]
        company, sector = self.detect_company_and_sector(company_from_filename, sample_text)

        logger.info(f"Detected - Company: {company}, Sector: {sector}")

        # Get relevant KPIs for the sector
        kpi_defs, kpi_index = self.get_sector_view(sector)
        logger.info(f"Using {len(kpi_defs)} KPI definitions for sector '{sector}'")

        total_kpis = 0
        failed_pages = []

        n_workers = max(1, min(self.n_llms, (max_workers or self.n_llms)))
        logger.info(f"Processing pages in parallel with {n_workers} worker(s) and {self.n_llms} LLM client(s)")

        def _process_one(page_idx_1based: int, page_text: str) -> Tuple[int, List[Dict], bool]:
            try:
                page_output_file = output_subdir / f"page_{page_idx_1based:03d}_kpis.json"
                if page_output_file.exists():
                    logger.info(f"Skipping page {page_idx_1based} - output file already exists")
                    with open(page_output_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    return page_idx_1based, existing_results, True
                
                save_page_text(page_text, company_from_filename, year, page_idx_1based, reports_dir)
                
                txt_path = find_txt_file_for_page(company_from_filename, year, page_idx_1based, reports_dir)
                figure_desc = read_txt_file(txt_path) if txt_path else ""
                combined_text = page_text

                llm_client = self.llms[(page_idx_1based - 1) % self.n_llms]

                results = self.page_llm_extract(
                    combined_text, figure_desc, kpi_defs,
                    company, sector, page_idx_1based, pdf_filename,
                    llm_client=llm_client
                )
                return page_idx_1based, results, True
            except RateLimitError as e:
                logger.error(f"Rate limit error on page {page_idx_1based}: {e}")
                return page_idx_1based, [], False
            except Exception as e:
                logger.exception(f"Unhandled exception processing page {page_idx_1based}: {e}")
                return page_idx_1based, [], False

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as exe:
            for page_num, page_text in enumerate(pdf_pages, 1):
                if not page_text.strip():
                    save_page_text("", company_from_filename, year, page_num, reports_dir)
                    out_file = output_subdir / f"page_{page_num:03d}_kpis.json"
                    with open(out_file, 'w', encoding='utf-8') as f:
                        json.dump([], f, indent=2, ensure_ascii=False)
                    continue
                futures[exe.submit(_process_one, page_num, page_text)] = page_num

            # Gather results as tasks complete
            for fut in concurrent.futures.as_completed(futures):
                page_num = futures[fut]
                try:
                    pn, page_results, success = fut.result()
                except Exception as e:
                    logger.error(f"Future failed for page {page_num}: {e}")
                    pn, page_results, success = page_num, [], False

                if not success:
                    failed_pages.append(pn)
                    logger.warning(f"Skipping page {pn} due to rate limit or persistent errors - no output file created")
                    continue

                page_output_file = output_subdir / f"page_{pn:03d}_kpis.json"
                with open(page_output_file, 'w', encoding='utf-8') as f:
                    json.dump(page_results, f, indent=2, ensure_ascii=False)

                if page_results:
                    total_kpis += len(page_results)
                    logger.info(f"Extracted {len(page_results)} KPIs from page {pn}, saved to {page_output_file}")
                else:
                    logger.info(f"No KPIs found on page {pn}, saved empty file {page_output_file}")

        if failed_pages:
            logger.warning(f"WARNING: {len(failed_pages)} page(s) failed due to rate limits or errors: {failed_pages}")
            logger.warning(f"These pages were SKIPPED and no output files were created for them.")
        
        logger.info(f"Total KPIs extracted from {pdf_filename}: {total_kpis}")
        return total_kpis

    def process_reports_directory(self, reports_dir: str):
        reports_dir = Path(reports_dir)
        
        # Find all PDF files and sort alphabetically
        pdf_files = sorted(list(reports_dir.glob("*.pdf")))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {reports_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        total_kpis = 0
        
        for pdf_path in pdf_files:
            try:
                kpis_count = self.process_pdf_report_with_page_outputs(str(pdf_path), str(reports_dir))
                total_kpis += kpis_count
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        
        logger.info(f"Processing complete! Extracted {total_kpis} total KPIs across all PDFs")

def main():
    parser = argparse.ArgumentParser(description="Extract KPIs from PDF reports")
    parser.add_argument("--reports-dir", "-r", default=f"{REPORTS_DIR}", 
                       help="Directory containing PDF reports and image subdirectories")
    parser.add_argument("--kpi-defs", "-k", default=f"{PROJECT_ROOT}/kpi_definitions.json",
                       help="Path to KPI definitions JSON file")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries for LLM calls (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                       help="Initial delay between retries in seconds (default: 2.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.reports_dir):
        logger.error(f"Reports directory not found: {args.reports_dir}")
        return
    
    if not os.path.exists(args.kpi_defs):
        logger.error(f"KPI definitions file not found: {args.kpi_defs}")
        return
    
    extractor = KPIExtractor(args.kpi_defs, max_retries=args.max_retries, retry_delay=args.retry_delay)
    extractor.process_reports_directory(args.reports_dir)

if __name__ == "__main__":
    main()