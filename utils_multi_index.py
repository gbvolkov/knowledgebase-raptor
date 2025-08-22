import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from langchain_experimental.open_clip import OpenCLIPEmbeddings

def is_image(doc) -> bool:
    """Detect if a Document represents an image by file extension."""
    src = doc.metadata.get("source", "").lower()
    return src.endswith((".png", ".jpg", ".jpeg"))

def display_image_base64(b64_string: str):
    """
    Decode a Base64-encoded image string (with optional data URI prefix)
    and display it using PIL, correcting padding errors if needed.
    """
    # 1) Remove data URI prefix if present
    if b64_string.startswith("data:") and "," in b64_string:
        # Split off the header, keep only the base64 data
        _, b64_string = b64_string.split(",", 1)

    # 2) Strip whitespace and newlines
    b64_string = "".join(b64_string.split())

    if missing_padding := len(b64_string) % 4:
        b64_string += "=" * (4 - missing_padding)

    # 4) Decode and display
    img_data = base64.b64decode(b64_string)
    img = Image.open(BytesIO(img_data))
    img.show()


def generate_clip_embeddings(
    images_path: str,
    model: OpenCLIPEmbeddings
) -> Tuple[List[List[float]], List[str]]:
    """
    Recursively traverse `images_path` for .jpg/.jpeg/.png files
    and compute CLIP embeddings for each image URI.
    """
    # 1) Recursive discovery of image files
    image_paths = [
        str(p) for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in Path(images_path).rglob(ext)
    ]  # Path.rglob finds files in nested folders 

    # 2) Batch-embed images into CLIP feature vectors
    embeddings = model.embed_image(image_paths)  # returns List[List[float]] :contentReference[oaicite:3]{index=3}

    return embeddings, image_paths
