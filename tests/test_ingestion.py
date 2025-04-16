# tests/test_ingestion.py
import os
import shutil
import tempfile
import unittest

from ingestion.loader import load_documents_from_directory
from ingestion.chunker import chunk_documents
from langchain.docstore.document import Document

class TestIngestion(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and sample text file
        self.test_dir = tempfile.mkdtemp()
        self.sample_text = (
            "This is a sample document. " * 100  # repeat to create a long text for chunking
        )
        self.file_path = os.path.join(self.test_dir, "example.txt")
        with open(self.file_path, "w", encoding="utf8") as f:
            f.write(self.sample_text)
    
    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
    
    def test_load_documents(self):
        # Test document loading from directory
        documents = load_documents_from_directory(self.test_dir)
        self.assertTrue(len(documents) > 0, "No documents were loaded.")
        # Check metadata and content
        for doc in documents:
            self.assertIn("source", doc.metadata)
            self.assertEqual(doc.metadata["source"], "example.txt")
            self.assertTrue(len(doc.page_content) > 0)
    
    def test_chunk_documents(self):
        # Load the document first
        documents = load_documents_from_directory(self.test_dir)
        # Chunk documents (with a small chunk size for testing)
        chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
        self.assertTrue(len(chunks) > 1, "Document was not split into multiple chunks.")
        # Check that metadata is preserved in chunks
        for chunk in chunks:
            self.assertIn("source", chunk.metadata)
            self.assertEqual(chunk.metadata["source"], "example.txt")
    
if __name__ == "__main__":
    unittest.main()
