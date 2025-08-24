# main.py
from loader import load_documents
from ingestion.chunker import chunk_documents
from raptor.tree_builder import recursive_embed_cluster_summarize
from vector_store.faiss_store import build_faiss_index_from_docs_chunked, save_faiss_index
from config import DOCUMENT_DIR, FAISS_INDEX_PATH
import pickle, os

FAISS_INDEX_PATH = "index/itil_index"
def main():
    # Recursively load documents from the local test_data folder.
    if os.path.isfile("data/documents/itil_docstore.pkl"):
        with open('data/documents/itil_docstore.pkl', 'rb') as file:
            documents = pickle.load(file)
    print(f"Loaded {len(documents)} documents.")

    # Optionally, split (chunk) documents if they are too long.
    chunked_docs = chunk_documents(documents)
    print(f"Chunked into {len(chunked_docs)} document pieces.")


    # Build the FAISS vector store.
    print("Building FAISS index...")
    index = build_faiss_index_from_docs_chunked(chunked_docs)
    save_faiss_index(index, FAISS_INDEX_PATH)
    print("FAISS index built and saved at:", FAISS_INDEX_PATH)

    # Optionally, save the loaded documents to a pickle file.
    with open(f'{FAISS_INDEX_PATH}/docstore.pkl', 'wb') as file:
        pickle.dump(documents, file)

if __name__ == "__main__":
    main()
