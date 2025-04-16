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
DOCUMENT_DIR = "test_data"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 0
CLUSTER_THRESHOLD = 0.1
CLUSTER_DIM = 10
RECURSION_LEVELS = 3
FAISS_INDEX_PATH = "faiss_index"
