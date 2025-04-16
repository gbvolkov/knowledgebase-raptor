# ingestion/loader.py
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredPDFLoader, 
    UnstructuredWordDocumentLoader
)

def load_documents(directory_path: str) -> list[Document]:
    """
    Recursively scans the given directory, loads supported files, and returns a list of LangChain Documents.
    
    Supported file types:
      - .txt, .md     : Loaded using TextLoader
      - .pdf          : Loaded using UnstructuredPDFLoader
      - .doc, .docx   : Loaded using UnstructuredWordDocumentLoader
    
    Each document gets assigned metadata including:
       - source: file name
       - relative_path: file path relative to the base directory
    
    Args:
        directory_path (str): The path to the base directory containing documents.
    
    Returns:
        List[Document]: List of loaded LangChain Document objects.
    """
    documents = []
    # Walk through directories recursively
    for root, _, files in os.walk(directory_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            try:
                # Choose the loader based on file extension
                if ext in ['.txt', '.md']:
                    loader = TextLoader(full_path, encoding="utf8")
                elif ext == '.pdf':
                    loader = UnstructuredPDFLoader(full_path)
                elif ext in ['.docx', '.doc']:
                    loader = UnstructuredWordDocumentLoader(full_path)
                else:
                    print(f"Unsupported file type {ext} for file {filename}")
                    continue

                docs = loader.load()
                # Compute relative path from the base directory
                rel_path = os.path.relpath(full_path, directory_path)
                # Set metadata for each loaded document
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["relative_path"] = rel_path
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    return documents
