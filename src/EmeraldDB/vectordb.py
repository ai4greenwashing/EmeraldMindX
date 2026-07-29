from __future__ import annotations

import argparse
import base64
import json
import os
import re
import uuid
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# Directory of ESG report PDFs named company_YYYY.pdf or company_YYYY_YYYY.pdf.
REPORTS_DIR = "path/to/reports"

# Persistent ChromaDB directory. The same path is passed to emx_rag.py.
CHROMADB_PATH = "path/to/chromadb"

# Cache of parsed page text, so a re-run skips the expensive parsing stage.
PARSED_DIR = "path/to/parsed"

COLLECTION_NAME = "esg_reports"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
VISION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

# Chunking is character-based; at roughly four characters per token, 800/150 is
# about 200 tokens with 40 of overlap.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# A page is sent to the vision model if it embeds an image at least this many
# pixels on a side, draws at least this many vector operations, or yields less
# than this much text.
MIN_IMAGE_SIZE = 100
MIN_DRAWING_OPS = 20
MIN_TEXT_LENGTH = 100

PAGE_RENDER_DPI = 300
MAX_RENDER_PIXELS = 1024
VISION_BATCH_SIZE = 8

FILENAME_PATTERN = re.compile(r"^(.+?)_(\d{4})(?:_(\d{4}))?\.pdf$", re.IGNORECASE)

VISION_PROMPT = (
    "This image is from an ESG report and may contain charts, bar graphs, "
    "tables, or infographics. Extract ALL data shown including: titles, "
    "axis labels, every data point with its label and value, and any "
    "footnotes or annotations. Structure the output as a readable table "
    "or list. Output ONLY the extracted data. "
    "If the image contains no charts, tables, graphs, or meaningful data, "
    "return exactly an empty string - do not explain, do not comment, "
    "do not say 'no data found'."
)

# Vision models sometimes narrate their refusal instead of returning nothing.
EMPTY_MARKERS = (
    "no data",
    "no chart",
    "no table",
    "no meaningful",
    "does not contain",
    "no relevant",
    "empty string",
)


class ReportStore:

    def __init__(
        self, db_path: str, embedding_model: str = EMBEDDING_MODEL, create: bool = False
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )
        client = chromadb.PersistentClient(path=db_path)
        get = client.get_or_create_collection if create else client.get_collection
        self.collection = get(
            name=COLLECTION_NAME, embedding_function=self.embedding_function
        )

    def add(
        self,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
        batch_size: int = 5000,
    ) -> None:
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self.collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    def query(
        self,
        text: str,
        n_results: int = 8,
        company: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[str]:

        filters = []
        if company:
            filters.append({"company": company})
        if year is not None:
            filters.append({"years": {"$contains": year}})

        arguments = {"query_texts": [text], "n_results": n_results}
        if len(filters) == 1:
            arguments["where"] = filters[0]
        elif filters:
            arguments["where"] = {"$and": filters}

        results = self.collection.query(**arguments)
        documents = results.get("documents") or []
        return documents[0] if documents else []


class VisionExtractor:

    def __init__(self, model: str, gpu_memory: float, max_model_len: int):
        self.model = model
        self.gpu_memory = gpu_memory
        self.max_model_len = max_model_len
        self._llm = None
        self._sampling_params = None

    def _load(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams

            print(f"Loading vision model: {self.model}")
            self._llm = LLM(
                model=self.model,
                dtype="float16",
                quantization="awq",
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_memory,
                max_model_len=self.max_model_len,
                limit_mm_per_prompt={"image": 1},
            )
            self._sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
        return self._llm

    def extract(self, images: List) -> List[str]:
        if not images:
            return []

        llm = self._load()
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_png(image)}"
                            },
                        },
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }
            ]
            for image in images
        ]

        texts = []
        for output in llm.chat(conversations, self._sampling_params, use_tqdm=False):
            text = output.outputs[0].text.strip()
            lowered = text.lower()
            if not text or any(marker in lowered for marker in EMPTY_MARKERS):
                texts.append("")
            else:
                texts.append(text)
        return texts


def encode_png(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_is_large(document, image_info) -> bool:
    try:
        image = document.extract_image(image_info[0])
        return image["width"] >= MIN_IMAGE_SIZE and image["height"] >= MIN_IMAGE_SIZE
    except Exception:
        return False


def needs_vision(page, text: str) -> Tuple[bool, str]:

    large_images = [
        i for i in page.get_images(full=True) if image_is_large(page.parent, i)
    ]
    if large_images:
        return True, f"{len(large_images)} embedded image(s)"

    operations = len(page.get_drawings())
    if operations >= MIN_DRAWING_OPS:
        return True, f"{operations} drawing operations"

    if not text:
        return True, "no extractable text"
    if len(text) < MIN_TEXT_LENGTH:
        return True, f"only {len(text)} characters of text"

    return False, "text only"


def render_page(page):
    from PIL import Image

    pixmap = page.get_pixmap(dpi=PAGE_RENDER_DPI)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    width, height = image.size
    if max(width, height) > MAX_RENDER_PIXELS:
        ratio = MAX_RENDER_PIXELS / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
    return image


def parse_report(pdf_path: str, extractor: Optional[VisionExtractor]) -> List[dict]:

    import fitz

    document = fitz.open(pdf_path)
    pages, to_render = [], []

    for number, page in enumerate(document, start=1):
        text = page.get_text().strip()
        visual, reason = needs_vision(page, text)
        record = {
            "page": number,
            "pymupdf_text": text,
            "vision_text": "",
            "vision_used": visual,
            "vision_reason": reason,
        }

        if visual and extractor is not None:
            try:
                record["image"] = render_page(page)
                to_render.append(len(pages))
            except Exception as error:
                record["vision_used"] = False
                record["vision_reason"] = f"render failed: {error}"
        pages.append(record)

    document.close()
    print(f"  {len(pages)} pages, {len(to_render)} needing the vision model")

    for start in range(0, len(to_render), VISION_BATCH_SIZE):
        batch = to_render[start : start + VISION_BATCH_SIZE]
        print(
            f"    vision batch {start // VISION_BATCH_SIZE + 1}: "
            f"pages {[pages[i]['page'] for i in batch]}"
        )
        try:
            texts = extractor.extract([pages[i]["image"] for i in batch])
            for index, text in zip(batch, texts):
                pages[index]["vision_text"] = text
        except Exception as error:
            print(f"    vision batch failed: {error}")

    for record in pages:
        record.pop("image", None)
        parts = [record["pymupdf_text"], record["vision_text"]]
        record["text"] = "\n\n".join(part for part in parts if part)

    return pages


def chunk(text: str, size: int, overlap: int) -> List[str]:
    stride = size - overlap
    chunks = []
    for start in range(0, len(text), stride):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
    return chunks


def parse_filename(filename: str) -> Optional[Dict[str, object]]:
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    first = int(match.group(2))
    last = int(match.group(3)) if match.group(3) else first
    return {"company": match.group(1), "years": list(range(first, last + 1))}


def cache_path(parsed_dir: str, company: str, years: List[int]) -> str:
    label = f"{years[0]}_{years[-1]}" if len(years) > 1 else str(years[0])
    return os.path.join(parsed_dir, f"{company}_{label}.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ESG report PDFs and ingest them into ChromaDB."
    )
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--db-path", default=CHROMADB_PATH)
    parser.add_argument("--parsed-dir", default=PARSED_DIR)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--vision-model", default=VISION_MODEL)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--no-vision", action="store_true", help="Use the text channel only."
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Parse and cache pages without writing to ChromaDB.",
    )
    arguments = parser.parse_args()

    pdfs = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(arguments.reports_dir)
        for name in names
        if name.lower().endswith(".pdf")
    )
    if not pdfs:
        print(f"No PDFs found in {arguments.reports_dir}")
        return
    print(f"Found {len(pdfs)} PDF(s)")

    os.makedirs(arguments.parsed_dir, exist_ok=True)
    extractor = (
        None
        if arguments.no_vision
        else VisionExtractor(
            arguments.vision_model, arguments.gpu_memory, arguments.max_model_len
        )
    )

    documents, metadatas, ids = [], [], []

    for pdf_path in pdfs:
        filename = os.path.basename(pdf_path)
        parsed_name = parse_filename(filename)
        if parsed_name is None:
            print(f"Skipping {filename}: expected company_YYYY[_YYYY].pdf")
            continue

        company, years = parsed_name["company"], parsed_name["years"]
        print(f"\n{filename} -> company={company}, years={years}")

        cache = cache_path(arguments.parsed_dir, company, years)
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as handle:
                pages = json.load(handle)["pages"]
            print(f"  loaded {len(pages)} cached pages")
        else:
            try:
                pages = parse_report(pdf_path, extractor)
            except Exception as error:
                print(f"  parsing failed: {error}")
                continue
            with open(cache, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source": filename,
                        "company": company,
                        "years": years,
                        "pages": pages,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  cached to {cache}")

        if arguments.no_ingest:
            continue

        for page in pages:
            if not page["text"].strip():
                continue
            for index, piece in enumerate(
                chunk(page["text"], arguments.chunk_size, arguments.chunk_overlap)
            ):
                documents.append(piece)
                metadatas.append(
                    {
                        "company": company,
                        "years": years,
                        "page": page["page"],
                        "chunk_id": index,
                        "vision_used": page["vision_used"],
                        "source": filename,
                    }
                )
                ids.append(str(uuid.uuid4()))

    if arguments.no_ingest:
        print(f"\nParsed pages cached in {os.path.abspath(arguments.parsed_dir)}")
        return

    if not documents:
        print("\nNothing to ingest.")
        return

    print(f"\nIngesting {len(documents)} chunks into {arguments.db_path}")
    store = ReportStore(arguments.db_path, arguments.embedding_model, create=True)
    store.add(documents, metadatas, ids)
    print("Done.")


if __name__ == "__main__":
    main()
