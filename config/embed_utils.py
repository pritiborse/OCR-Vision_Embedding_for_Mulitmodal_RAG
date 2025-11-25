import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def hybrid_embed(text=None, image_path=None):
    """Create a 896-D embedding (384 text + 512 image)."""
    text_emb = np.zeros((1, 384), dtype="float32")
    img_emb = np.zeros((1, 512), dtype="float32")

    if text:
        text_emb = text_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**inputs).cpu().numpy().astype("float32")

    return np.concatenate([text_emb, img_emb], axis=1).astype("float32")
