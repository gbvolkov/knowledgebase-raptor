# embedder.py
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

def get_embedding_model():
    """
    Returns a HuggingFaceEmbeddings object using the specified model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
