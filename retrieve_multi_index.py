# src/image_retrieval.py

import argparse
from pathlib import Path
from typing import List
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import INDEX_DIR, FAISS_INDEX_PATH, RERANKING_MODEL
from vector_store.faiss_store import load_faiss_index, query_faiss_index


def load_image_index() -> FAISS:
    """
    Reload the FAISS index and its CLIP embedder from disk.
    """
    clip_embedder = OpenCLIPEmbeddings(
        model_name="ViT-B-32",
        checkpoint="laion2b_s34b_b79k",
        load_fn_kwargs={"device": "cpu"},
    )  # Embedder must match indexing settings :contentReference[oaicite:8]{index=8}

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        clip_embedder,
        allow_dangerous_deserialization=True,
    )  # Reloads FAISS index + metadata :contentReference[oaicite:9]{index=9}

    return vectorstore

def retrieve_similar_images(
    query: str,
    k: int = 5
) -> List[str]:
    """
    Given an image file path or text query, embed it into CLIP space,
    perform a vector lookup, and return top-k image URIs.
    """
    vectorstore = load_image_index()

    # 1) Determine if query is an image file
    ext = Path(query).suffix.lower()
    clip_embedder = vectorstore.embeddings  # reuse embedder instance

    if ext in (".png", ".jpg", ".jpeg"):
        # Embed the query image
        query_embedding = clip_embedder.embed_image([query])[0]  # :contentReference[oaicite:10]{index=10}
    else:
        # Embed a text query
        query_embedding = clip_embedder.embed_query(query)        # :contentReference[oaicite:11]{index=11}

    # 2) Find top-k nearest neighbors by vector
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)  # :contentReference[oaicite:12]{index=12}

    # 3) Extract and return image URIs from metadata
    return [doc.metadata["source"] for doc in docs]


def test_vectorstore_retrieval():
    # Load the FAISS index stored at FAISS_INDEX_PATH.
    index = load_faiss_index(FAISS_INDEX_PATH)
    
    # Define your query for testing.
    query_text = "Как получить zerocoins?"
    print("Query:", query_text)
    
    # Run a similarity search on the vector store. k determines the number of results.
    results = query_faiss_index(index, query_text, k=5)
    
    # Print the retrieved document contents.
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content)
        print("-" * 80)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(
    #    description="Image-to-image (or text-to-image) search with CLIP+FAISS"
    #)
    #parser.add_argument("query", type=str, help="Path to query image or text string")
    #parser.add_argument("--k", type=int, default=5, help="Number of results")
    #args = parser.parse_args()
    image_vs = load_image_index()
    text_vs = load_faiss_index(FAISS_INDEX_PATH)
    chats_vs = load_faiss_index("chats_index")

    k = 5
    ensemble = EnsembleRetriever(
        retrievers=[text_vs.as_retriever(search_kwargs={"k": k}), chats_vs.as_retriever(search_kwargs={"k": k})],
        weights=[0.5, 0.5]                  # adjust to favor text vs. images
    )
    reranker_model = HuggingFaceCrossEncoder(model_name=RERANKING_MODEL)
    RERANKER = CrossEncoderReranker(model=reranker_model, top_n=3)
    retriever = ContextualCompressionRetriever(
            base_compressor=RERANKER, base_retriever=ensemble
            )

    query = "Когда вебминар"

    results = retriever.get_relevant_documents(query, search_kwargs={"k": k})
    for doc in results:
        if doc.metadata.get("type") == "image":
            print(doc.metadata["source"])
        else:
            print(doc.page_content)
    