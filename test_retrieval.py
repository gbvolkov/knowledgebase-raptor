# test_retrieval.py
from vector_store.faiss_store import load_faiss_index, query_faiss_index
from config import FAISS_INDEX_PATH

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
    test_vectorstore_retrieval()
