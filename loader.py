# ingestion/loader.py
from json import JSONDecodeError
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    AssemblyAIAudioTranscriptLoader,
    UnstructuredCSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEmailLoader,
    JSONLoader,
)
from pydub import AudioSegment
import config


def convert_audio_to_wav(input_file, output_file, audio_type):
    """
    Converts an M4A (or similar) file to WAV format.
    """
    audio = AudioSegment.from_file(input_file, format=audio_type)
    audio.export(output_file, format='wav')

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
                #if ext in ['.txt', '.md', '.py']:
                if ext in ['.txt', '.py']:
                    loader = TextLoader(full_path, encoding="utf8")
                elif ext == '.json':
                    loader = JSONLoader(full_path, jq_schema=".", text_content=False)
                elif ext == '.pdf':
                    loader = UnstructuredPDFLoader(full_path)
                elif ext in ['.docx', '.doc']:
                    loader = UnstructuredWordDocumentLoader(full_path)
                elif ext in ['.pptx', '.ppt']:
                    # PowerPoint loader: returns a single Document by default
                    loader = UnstructuredPowerPointLoader(full_path, mode="single")
                elif ext in ['.xlsx', '.xls']:
                    # Excel loader: returns a single Document by default
                    loader = UnstructuredExcelLoader(full_path, mode="single")
                elif ext in ['.csv']:
                    loader = UnstructuredCSVLoader(full_path, mode="single")
                elif ext in ['.html']:
                    loader = UnstructuredHTMLLoader(full_path, mode="single")
                elif ext in ['.xml']:
                    loader = UnstructuredXMLLoader(full_path, mode="single")
                elif ext in ['.rtf']:
                    loader = UnstructuredRTFLoader(full_path, mode="single")
                elif ext in ['.md']:
                    loader = UnstructuredMarkdownLoader(full_path, mode="single")
                elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                    loader = UnstructuredImageLoader(full_path, mode="single")
                elif ext in ['.msg', '.eml']:
                    loader = UnstructuredEmailLoader(full_path)
                elif ext in ['.mp4', '.mov', '.avi']:
                    wav_name = filename
                    name, ext = os.path.splitext(filename)
                    ext = ext.replace('.', '')
                    wav_name = f"temp/{name}.wav"

                    convert_audio_to_wav(full_path, wav_name, ext)
                    import assemblyai
                    config = assemblyai.TranscriptionConfig(
                        language_code='ru'
                    )
                    loader = AssemblyAIAudioTranscriptLoader(wav_name, config=config)
                else:
                    print(f"Unsupported file type {ext} for file {filename}")
                    continue
                try:
                    docs = loader.load()
                except Exception as e:
                    if not isinstance(loader, JSONLoader) or not isinstance(
                        e, JSONDecodeError
                    ):
                        raise e
                    from pathlib import Path
                    src = Path(full_path)
                    text = src.read_text(encoding="utf-8-sig")
                    tmp = src.with_suffix(".nobom.json")
                    tmp.write_text(text, encoding="utf-8")
                    try:
                        loader = JSONLoader(file_path=str(tmp), jq_schema=".", text_content=False)
                        docs = loader.load()
                    finally:
                         tmp.unlink(missing_ok=True)
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


if __name__ == "__main__":
    DOCUMENT_DIR = "data/itil"
    documents = load_documents(DOCUMENT_DIR)
    import pickle
    with open('data/documents/itil_docstore.pkl', 'wb') as file:
        pickle.dump(documents, file)
    with open('data/documents/itil_docstore.pkl', 'rb') as file:
        docs = pickle.load(file)

    print(f"Loaded {len(documents)} documents.")