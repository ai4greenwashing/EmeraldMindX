import os
import re
import uuid
import subprocess
import chromadb
from chromadb.utils import embedding_functions

def get_git_root():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return root
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_git_root()
SOURCE_DIR = os.path.join(PROJECT_ROOT, "reports")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def get_text_chunks(text, chunk_size, overlap):
    stride = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), stride):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks

def process_and_ingest():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=DB_PATH)
    
    collection = client.get_or_create_collection(
        name="esg_reports",
        embedding_function=sentence_transformer_ef
    )

    documents = []
    metadatas = []
    ids = []
    
    filename_pattern = re.compile(r"^(.*)_(\d{4})_page_(\d+)\.txt$")

    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue

            match = filename_pattern.match(file)
            if not match:
                continue

            company = match.group(1)
            year = match.group(2)
            page = match.group(3)
            source_pdf = f"{company}_{year}.pdf"
            
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                text_chunks = get_text_chunks(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
                
                for idx, chunk in enumerate(text_chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "company": company,
                        "year": int(year),
                        "page": int(page),
                        "chunk_id": idx,
                        "source": source_pdf
                    })
                    ids.append(str(uuid.uuid4()))
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if documents:
        print(f"Ingesting {len(documents)} chunks...")
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            end = i + batch_size
            collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        print("Ingestion complete.")
    else:
        print("No valid documents found.")

class ReportParser:
    def __init__(self, db_path, embedding_model_name="sentence-transformers/all-mpnet-base-v2"):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.collection = self.client.get_collection(
            name="esg_reports", 
            embedding_function=self.ef
        )

    def query(self, query_text, n_results=5, company_filter=None):
        query_args = {
            "query_texts": [query_text],
            "n_results": n_results,
        }

        if company_filter:
            query_args["where"] = {"company": company_filter}

        return self.collection.query(**query_args)

if __name__ == "__main__":
    process_and_ingest()