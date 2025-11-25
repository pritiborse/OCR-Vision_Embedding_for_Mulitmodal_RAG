import os
import faiss
import numpy as np
import json


class VectorStore:
    """
    Handles FAISS index creation, storage, and retrieval of hybrid embeddings
    (text + image) with associated metadata.
    """

    def __init__(self, index_dir=None, index_type="IndexFlatIP"):
        self.index = None
        self.metadata = []
        self.dim = None
        self.index_type = index_type

        if index_dir and os.path.exists(os.path.join(index_dir, "faiss_index.bin")):
            self.load(index_dir)

    def _create_index(self):
        if self.dim is None:
            raise ValueError("Embedding dimension must be set before creating index.")

        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def add(self, embeddings, metadata_list):
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings provided to add().")

        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.dim = embeddings.shape[1]
            self._create_index()

        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}")

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, top_k=5, min_score=0.2):
        """
        Search for similar embeddings and return results with scores.
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("The FAISS index is empty.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search more candidates initially
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype("float32"), search_k)

        results = []
        seen_texts = set()
        
        for dist, idx in zip(distances[0], indices[0]):
            # Filter by minimum score
            if dist < min_score:
                continue

            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            text = meta.get("text", "").strip()
            
            # Skip duplicates and empty results
            if not text or text in seen_texts:
                continue
            
            seen_texts.add(text)
            results.append({
                "score": float(dist),
                "text": text,
                "page": meta.get("page", 0),
                "type": meta.get("type", "text")
            })
            
            # Stop when we have enough unique results
            if len(results) >= top_k:
                break

        return results

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.bin"))

        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"[VectorStore] Saved FAISS index ({self.dim}-D) and metadata to {save_dir}")

    def load(self, load_dir):
        index_path = os.path.join(load_dir, "faiss_index.bin")
        metadata_path = os.path.join(load_dir, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found in {load_dir}")

        self.index = faiss.read_index(index_path)
        self.dim = self.index.d

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        print(f"[VectorStore] Loaded index with dim={self.dim}, entries={self.index.ntotal}")