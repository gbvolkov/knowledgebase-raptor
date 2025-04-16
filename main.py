# main.py
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from raptor.tree_builder import recursive_embed_cluster_summarize
from vector_store.faiss_store import build_faiss_index, save_faiss_index
from config import DOCUMENT_DIR, FAISS_INDEX_PATH

def main():
    # Recursively load documents from the local test_data folder.
    print("Loading documents from:", DOCUMENT_DIR)
    documents = load_documents(DOCUMENT_DIR)
    print(f"Loaded {len(documents)} documents.")

    # Optionally, split (chunk) documents if they are too long.
    chunked_docs = chunk_documents(documents)
    print(f"Chunked into {len(chunked_docs)} document pieces.")

    # Get raw text from all chunks.
    leaf_texts = [doc.page_content for doc in chunked_docs if doc.page_content.strip()]
    
    # Build the RAPTOR tree recursively.
    print("Building RAPTOR tree...")
    tree_results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

    # Flatten the tree by gathering leaf texts and all summaries.
    all_texts = leaf_texts.copy()
    for level in sorted(tree_results.keys()):
        summaries = tree_results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)
    print(f"Total number of texts for vector index: {len(all_texts)}")

    # Build the FAISS vector store.
    print("Building FAISS index...")
    index = build_faiss_index(all_texts)
    save_faiss_index(index, FAISS_INDEX_PATH)
    print("FAISS index built and saved at:", FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()
