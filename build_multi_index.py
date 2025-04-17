# src/image_indexing.py

import os
from typing import List
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from utils_multi_index import generate_clip_embeddings
from config import IMAGE_DATA_DIR, INDEX_DIR

def create_faiss_index(images_path: str) -> FAISS:
    """
    Build and save a FAISS index from CLIP embeddings of images under `images_path`.
    """
    # 1) Initialize the CLIP embedder (ViT-B-32 / laion2b_s34b_b79k)
    clip_embedder = OpenCLIPEmbeddings(
        model_name="ViT-B-32",
        checkpoint="laion2b_s34b_b79k",
        load_fn_kwargs={"device": "cpu"},
    )  # Multi-modal CLIP embeddings :contentReference[oaicite:4]{index=4}

    # 2) Generate embeddings + URI list
    embeddings, image_paths = generate_clip_embeddings(images_path, clip_embedder)

    # 3) Pair each URI with its vector for FAISS
    text_embeddings = list(zip(image_paths, embeddings))  # Iterable[(str, List[float])] :contentReference[oaicite:5]{index=5}

    # 4) Build FAISS index from precomputed embeddings
    faiss_index = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=clip_embedder,
        metadatas=[{"source": uri, "type": "image"} for uri in image_paths],
        ids=list(range(len(image_paths))),
    )  # Constructs a brute‑force FAISS index :contentReference[oaicite:6]{index=6}

    # 5) Persist index + metadata to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss_index.save_local(INDEX_DIR)  # Saves index files under INDEX_DIR :contentReference[oaicite:7]{index=7}

    print(f"Indexed {len(image_paths)} images and saved FAISS index to {INDEX_DIR}")
    return faiss_index


if __name__ == "__main__":
    create_faiss_index(IMAGE_DATA_DIR)
