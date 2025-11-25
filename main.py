import os
import json
import re
from config.vectorstore import VectorStore
from config.embed_utils import hybrid_embed

# --- Path setup ---
index_dir = "outputs/hybrid_index"
queries_file = "data/test_queries.json"
output_json = "outputs/query_results.json"


def extract_key_info(text, query):
    """Extract most relevant information from text based on query."""
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into bullet points or sentences
    parts = []
    
    # Check for bullet points (lines starting with special chars)
    if '•' in text or text.count('\n') > 3:
        parts = [p.strip() for p in re.split(r'[•\n]', text) if p.strip()]
    else:
        parts = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 15]
    
    # Filter relevant parts based on query keywords
    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    
    scored_parts = []
    for part in parts:
        part_lower = part.lower()
        # Calculate relevance score
        matches = sum(1 for word in query_keywords if word in part_lower)
        if matches > 0 or len(query_keywords) < 3:
            scored_parts.append((matches, part))
    
    # Sort by relevance
    scored_parts.sort(reverse=True, key=lambda x: x[0])
    
    # Take top 5 most relevant parts
    selected = [part for score, part in scored_parts[:5]]
    
    return ' '.join(selected) if selected else parts[:3]


def format_answer(query, retrieved_chunks):
    """Format answer by combining top relevant chunks."""
    if not retrieved_chunks:
        return "No relevant information found in the document."
    
    # Take top 3 chunks with good scores
    top_chunks = [c for c in retrieved_chunks[:5] if c["score"] > 0.28]
    
    if not top_chunks:
        return "No sufficiently relevant information found."
    
    # Combine texts
    combined = ' '.join([c['text'] for c in top_chunks[:3]])
    
    # Clean and extract key information
    answer = extract_key_info(combined, query)
    
    # Ensure proper ending
    if isinstance(answer, list):
        answer = ' '.join(answer)
    
    if not answer.endswith('.'):
        answer += '.'
    
    return answer


def main():
    print("="*60)
    print("Construction Document QA System")
    print("="*60)
    
    # Check if index exists
    if not os.path.exists(os.path.join(index_dir, "faiss_index.bin")):
        print("\n Index not found. Please run: python build_index.py")
        return
    
    # Load queries from JSON
    if not os.path.exists(queries_file):
        print(f"\n Queries file not found: {queries_file}")
        return
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    # Load vector store
    print("\n[Loading] Vector store...")
    store = VectorStore(index_dir=index_dir)
    print(f"   → Index dimension: {store.dim}")
    print(f"   → Total entries: {len(store.metadata)}")
    
    # Process queries
    results = {}
    
    print("\n" + "="*60)
    print("Processing Queries")
    print("="*60)
    
    for query_item in queries_data:
        for key, query in query_item.items():
            print(f"\n {key}: {query}")
            
            # Create query embedding
            q_emb = hybrid_embed(query)
            
            # Retrieve similar chunks
            retrieved = store.search(q_emb, top_k=8, min_score=0.25)
            
            # Format answer
            answer = format_answer(query, retrieved)
            
            # Store result
            results[key] = {
                "question": query,
                "answer": answer
            }
            
            print(f" Answer: {answer[:150]}...")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f" Results saved to: {output_json}")
    print("="*60)


if __name__ == "__main__":
    main()