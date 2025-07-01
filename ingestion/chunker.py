# chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document
from ingestion import utils
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #model_name=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    For each document, split its text into chunks (if needed) and return a new list of Documents.
    """
    chunked_docs = []
    for doc in docs:
        chunks = chunk_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(page_content=chunk, metadata=doc.metadata)
            )
    return chunked_docs
