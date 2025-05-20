# main.py
#from loader import load_documents
from ingestion.chunker import chunk_documents
from raptor.tree_builder import recursive_embed_cluster_summarize
from vector_store.faiss_store import build_faiss_index, save_faiss_index
import pickle
from config import DOCUMENT_DIR, FAISS_INDEX_PATH

DOCUMENT_DIR = "data"
FAISS_INDEX_PATH = "index/qa_index"
def main():
    from langchain_community.document_loaders.csv_loader import CSVLoader

    fileds = "Status,cluster_summary_question,cluster_summary_article,Комментарий,questions,cluster_summary_answer,cluster_summary_article_question,answers,cluster".split(",")
    print("Loading documents from:", DOCUMENT_DIR)
    loader = CSVLoader(
        file_path="./data/00_QA_clustered_ready.csv",
        csv_args={
            "fieldnames": fileds,
        },
        encoding="utf-8",
        metadata_columns=["cluster"],
        content_columns = ["cluster_summary_question","cluster_summary_article"]
    )
    documents = loader.load()[1:]
    print(f"Loaded {len(documents)} documents.")

    # Optionally, save the loaded documents to a pickle file.
    with open(f'{FAISS_INDEX_PATH}/docstore.pkl', 'wb') as file:
        pickle.dump(documents, file)

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
