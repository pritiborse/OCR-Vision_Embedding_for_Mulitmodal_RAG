"""
Script to build the hybrid FAISS index from PDF.
Run this once before using main.py
"""
from techniques.Hybrid_OCR_vision_embedding import build_hybrid_index

pdf_path = "data/Construction.pdf"
output_dir = "outputs/hybrid_index"

if __name__ == "__main__":
    print("\n🔨 Building Hybrid Index (Text + Vision + OCR)\n")
    build_hybrid_index(pdf_path, output_dir)
    print("\n✅ Indexing complete! You can now run main.py\n")