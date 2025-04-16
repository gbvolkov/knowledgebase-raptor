# faiss_store.py
from langchain_community.vectorstores import FAISS
from embeddings.embedder import get_embedding_model
from typing import List

def build_faiss_index(texts: List[str]):
    """
    Builds a FAISS index from the given texts using the HuggingFace embedding model.
    """
    embed_model = get_embedding_model()
    # Use the FAISS wrapper from LangChain to build the vector store.
    return FAISS.from_texts(texts, embed_model)

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
