from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path


if os.path.exists("gv.env"):
    load_dotenv('./gv.env')
else:
    documents_path = Path.home() / ".env"
    load_dotenv(os.path.join(documents_path, 'gv.env'))


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# config.py
DOCUMENT_DIR = "data"
EMBEDDING_MODEL_NAME=os.environ.get('EMBEDDING_MODEL_NAME') or "intfloat/multilingual-e5-large"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250
CLUSTER_THRESHOLD = 0.1
CLUSTER_DIM = 10
RECURSION_LEVELS = 3
FAISS_INDEX_PATH = "index/neuro_index"
RERANKING_MODEL=os.environ.get('RERANKING_MODEL') or '/models/bge-reranker-large'

# Root data directory
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "test_data/Notion0code")  

# Where to save/load the FAISS index
INDEX_DIR = os.getenv("INDEX_DIR", "multi_index")  
IMAGE_DATA_DIR = RAW_DATA_DIR

ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY') or "ND"


