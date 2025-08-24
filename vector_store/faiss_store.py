# faiss_store.py
from typing import List, Any
import logging, sys, time

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from embeddings.embedder import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/loader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_faiss_index(texts: List[str]):
    """
    Builds a FAISS index from the given texts using the HuggingFace embedding model.
    """
    embed_model = get_embedding_model()
    # Use the FAISS wrapper from LangChain to build the vector store.
    return FAISS.from_texts(texts, embed_model)

def build_faiss_index_from_docs_chunked(documents: list[Document], batch_size: int = 500, max_retries: int = 3):
    """
    Builds a FAISS index from the given texts using the HuggingFace embedding model.
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    # Step 2: Initialize the embedding model
    embeddings = get_embedding_model()
    first_batch = documents[:batch_size]
    logging.info(f"Initializing FAISS vectorstore with the first batch of {len(first_batch)} documents.")
    vectorstore = FAISS.from_documents(documents=first_batch, embedding=embeddings)

    # Step 4: Transfer FAISS index to GPU if CUDA is available
    if device == "cuda":
        import faiss
        if hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            cpu_index = vectorstore.index
            vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logging.info("FAISS index transferred to GPU.")

    # Step 5: Process and add remaining documents in batches
    remaining_documents = documents[batch_size:]
    total_batches = (len(remaining_documents) + batch_size - 1) // batch_size
    logging.info(f"Adding remaining documents in {total_batches} batches of size {batch_size}.")

    for batch_num, i in enumerate(range(0, len(remaining_documents), batch_size), start=1):
        batch = remaining_documents[i:i + batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                logging.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents.")
                vectorstore.add_documents(batch)
                logging.info(f"Batch {batch_num}/{total_batches} added successfully.")
                break
            except Exception as e:
                attempt += 1
                wait_time = 2 ** attempt
                logging.warning(f"Attempt {attempt}/{max_retries} failed for batch {batch_num}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if attempt == max_retries:
                    logging.error(f"Batch {batch_num} failed after {max_retries} attempts. Skipping this batch.")

    logging.info("All batches processed. Vector store is ready.")
    return vectorstore



def build_faiss_index_from_docs(docs: list[Document]):
    """
    Builds a FAISS index from the given texts using the HuggingFace embedding model.
    """
    embed_model = get_embedding_model()
    # Use the FAISS wrapper from LangChain to build the vector store.
    return FAISS.from_documents(docs, embed_model)

def save_faiss_index(index, path: str):
    """
    Saves the FAISS index locally.
    """
    index.save_local(path)

def load_faiss_index(path: str):
    """
    Loads a FAISS index from the specified local directory.
    """
    embed_model = get_embedding_model()
    return FAISS.load_local(path, embed_model, allow_dangerous_deserialization=True)

def query_faiss_index(index, query: str, k: int = 5):
    """
    Runs a similarity search on the FAISS index.
    """
    return index.similarity_search(query, k=k)
