"""
Script to view and export embeddings from the FAISS index
"""
import os
import json
import numpy as np
import faiss
from config.vectorstore import VectorStore

index_dir = "outputs/hybrid_index"
output_dir = "outputs/embeddings_export"


def export_embeddings():
    """Export embeddings and metadata to readable formats."""
    
    # Check if index exists
    if not os.path.exists(os.path.join(index_dir, "faiss_index.bin")):
        print(" Index not found. Please run: python build_index.py")
        return
    
    print("="*60)
    print("Embedding Viewer & Exporter")
    print("="*60)
    
    # Load vector store
    store = VectorStore(index_dir=index_dir)
    
    # Get embeddings from FAISS index - Fixed method
    n_vectors = store.index.ntotal
    embeddings = np.zeros((n_vectors, store.dim), dtype=np.float32)
    
    for i in range(n_vectors):
        embeddings[i] = store.index.reconstruct(i)
    
    print(f"\n  Index Statistics:")
    print(f"   → Total embeddings: {n_vectors}")
    print(f"   → Embedding dimension: {store.dim}")
    print(f"   → Text embeddings (384-D): {embeddings[:, :384].shape}")
    print(f"   → Image embeddings (512-D): {embeddings[:, 384:].shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save embeddings as numpy array
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    print(f"\n Saved embeddings to: {output_dir}/embeddings.npy")
    
    # 2. Save metadata with embeddings info
    export_data = []
    for i, meta in enumerate(store.metadata):
        export_data.append({
            "index": i,
            "page": meta.get("page", 0),
            "type": meta.get("type", "unknown"),
            "text_preview": meta.get("text", "")[:200] + "..." if len(meta.get("text", "")) > 200 else meta.get("text", ""),
            "full_text": meta.get("text", ""),
            "confidence": meta.get("confidence", 0),
            "embedding_shape": [store.dim],
            "text_embedding_norm": float(np.linalg.norm(embeddings[i, :384])),
            "image_embedding_norm": float(np.linalg.norm(embeddings[i, 384:])),
        })
    
    # Save as JSON
    with open(os.path.join(output_dir, "embeddings_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f" Saved metadata to: {output_dir}/embeddings_metadata.json")
    
    # 3. Create summary statistics
    summary = {
        "total_embeddings": int(n_vectors),
        "embedding_dimension": int(store.dim),
        "text_dimension": 384,
        "image_dimension": 512,
        "embeddings_by_type": {},
        "embeddings_by_page": {},
        "average_text_norm": float(np.mean([np.linalg.norm(embeddings[i, :384]) for i in range(len(embeddings))])),
        "average_image_norm": float(np.mean([np.linalg.norm(embeddings[i, 384:]) for i in range(len(embeddings))])),
    }
    
    # Count by type
    for meta in store.metadata:
        chunk_type = meta.get("type", "unknown")
        summary["embeddings_by_type"][chunk_type] = summary["embeddings_by_type"].get(chunk_type, 0) + 1
    
    # Count by page
    for meta in store.metadata:
        page = meta.get("page", 0)
        summary["embeddings_by_page"][f"page_{page}"] = summary["embeddings_by_page"].get(f"page_{page}", 0) + 1
    
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f" Saved summary to: {output_dir}/summary.json")
    
    # 4. Display sample embeddings
    print("\n" + "="*60)
    print("Sample Embeddings (First 3)")
    print("="*60)
    
    for i in range(min(3, len(export_data))):
        item = export_data[i]
        print(f"\n Index {i} | Page {item['page']} | Type: {item['type']}")
        print(f"   Text: {item['text_preview']}")
        print(f"   Text Embedding Norm: {item['text_embedding_norm']:.4f}")
        print(f"   Image Embedding Norm: {item['image_embedding_norm']:.4f}")
        print(f"   First 10 dimensions: {embeddings[i, :10].tolist()}")
    
    # 5. Display summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"\n Embeddings by Type:")
    for chunk_type, count in summary["embeddings_by_type"].items():
        print(f"   → {chunk_type}: {count}")
    
    print(f"\n Embeddings by Page (top 5):")
    page_counts = sorted(summary["embeddings_by_page"].items(), key=lambda x: x[1], reverse=True)
    for page, count in page_counts[:5]:
        print(f"   → {page}: {count}")
    
    print(f"\n Average Norms:")
    print(f"   → Text embeddings: {summary['average_text_norm']:.4f}")
    print(f"   → Image embeddings: {summary['average_image_norm']:.4f}")
    
    print("\n" + "="*60)
    print(f" All files saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    export_embeddings()