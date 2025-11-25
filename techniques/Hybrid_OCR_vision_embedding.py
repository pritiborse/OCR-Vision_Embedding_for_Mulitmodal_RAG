import os
import numpy as np
from tqdm import tqdm
import fitz
from config.embed_utils import hybrid_embed
from config.vectorstore import VectorStore
from config.detect_utils import extract_images_and_captions


def extract_text_with_sections(pdf_path):
    """Extract text preserving semantic sections and headers."""
    pdf_document = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(pdf_document, start=1):
        blocks = page.get_text("blocks")
        
        current_section = ""
        current_header = ""
        
        for block in blocks:
            if len(block) >= 5:
                text = block[4].strip()
                
                if not text:
                    continue
                
                # Detect headers (uppercase, short lines)
                if text.isupper() and len(text) < 100 and '\n' not in text:
                    # Save previous section
                    if current_section:
                        chunks.append({
                            "page": page_num,
                            "text": f"{current_header} {current_section}".strip(),
                            "type": "text"
                        })
                    current_header = text
                    current_section = ""
                else:
                    current_section += " " + text
                    
                    # If section gets large, save it
                    if len(current_section) > 600:
                        chunks.append({
                            "page": page_num,
                            "text": f"{current_header} {current_section}".strip(),
                            "type": "text"
                        })
                        current_section = ""
        
        # Save remaining content
        if current_section:
            chunks.append({
                "page": page_num,
                "text": f"{current_header} {current_section}".strip(),
                "type": "text"
            })
    
    pdf_document.close()
    return chunks


def build_hybrid_index(pdf_path, output_dir="outputs/hybrid_index"):
    """Extract text + images, create embeddings, and build FAISS index."""
    print("[1/4] Extracting text chunks from PDF...")
    text_chunks = extract_text_with_sections(pdf_path)
    print(f"   → Extracted {len(text_chunks)} text chunks")

    print("[2/4] Extracting images and OCR...")
    image_data = extract_images_and_captions(pdf_path, output_dir=os.path.join(output_dir, "extracted_images"))
    print(f"   → Extracted {len(image_data)} images")

    # Clean output directory
    if os.path.exists(os.path.join(output_dir, "faiss_index.bin")):
        os.remove(os.path.join(output_dir, "faiss_index.bin"))
    if os.path.exists(os.path.join(output_dir, "metadata.json")):
        os.remove(os.path.join(output_dir, "metadata.json"))
    os.makedirs(output_dir, exist_ok=True)

    print("[3/4] Creating hybrid embeddings...")
    embeddings = []
    metadata_list = []

    # Process text chunks
    for chunk in tqdm(text_chunks, desc="Embedding text"):
        if len(chunk["text"]) < 30:
            continue
        
        emb = hybrid_embed(text=chunk["text"])
        embeddings.append(emb)
        metadata_list.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "type": "text",
            "confidence": 1.0
        })

    # Process images with OCR
    for img_data in tqdm(image_data, desc="Embedding images"):
        text_content = img_data["combined_text"]
        
        if len(text_content) < 10:
            continue
        
        emb = hybrid_embed(text=text_content, image_path=img_data["image_path"])
        embeddings.append(emb)
        metadata_list.append({
            "page": img_data["page"],
            "text": text_content,
            "type": "image",
            "image_path": img_data["image_path"],
            "confidence": img_data["ocr_conf"] / 100.0
        })

    print("[4/4] Building FAISS index...")
    embeddings = np.vstack(embeddings)
    store = VectorStore()
    store.add(embeddings, metadata_list)
    store.save(output_dir)
    
    print(f"  Successfully indexed {len(metadata_list)} items")
    print(f"   - Text chunks: {len(text_chunks)}")
    print(f"   - Image chunks: {len(image_data)}")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")