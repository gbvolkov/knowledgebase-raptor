# run_test_ingestion.py
import os
from loader import load_documents_from_directory
from ingestion.chunker import chunk_documents

# Specify the path to your test folder
test_folder = os.path.join(os.getcwd(), "test_data")

# Load documents using the ingestion module
documents = load_documents_from_directory(test_folder)
print(f"Loaded {len(documents)} documents from {test_folder}")

# Optionally, run them through the chunker to see the output chunks
chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=100)
print(f"Created {len(chunks)} document chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (source: {chunk.metadata.get('source')}):")
    print(chunk.page_content[:200] + '...')  # print first 200 characters for preview
    print("\n---\n")
