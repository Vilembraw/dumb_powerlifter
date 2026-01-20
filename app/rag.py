import json
import os
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer

from app.rag_loader import load_pdf, chunk_pages

EMBEDDER_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 600
OVERLAP = 100
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.json"

embedder = SentenceTransformer(EMBEDDER_MODEL)
dimension = 384

index = None
chunks_db = []

def build_knowledge_base(pdf_path: str, force_rebuild=False):
    """Builds the knowledge base from the given PDF file."""
    global index, chunks_db
    if not force_rebuild and os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
        # Load existing index and chunks
        print("Loading existing knowledge base...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks_db = json.load(f)
        return

    # Initialize FAISS index
    print("Building knowledge base...")
    # Load and chunk the PDF
    pages = load_pdf(pdf_path)
    chunks_db = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    # Save chunks to file
    print("Creating embeddings...")
    texts = [c["text"] for c in chunks_db]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Build FAISS index
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))

    # Save index and chunks
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_db, f, ensure_ascii=False, indent=2)

    print("Knowledge base built and saved.")



def search_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Searches the knowledge base for relevant chunks."""
    if index is None or not chunks_db:
        print("Knowledge base not loaded.")
        return []

    query_vector = embedder.encode(
        [query],
        convert_to_numpy=True,
    ).astype('float32')

    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks_db):
            chunk = chunks_db[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": float(distances[0][i])
            })
    return results


def init_rag(pdf_path: str = "data/poliquin_picp_level_1.pdf", force_rebuild: bool = False):
    """Initializes the RAG system by building/loading the knowledge base."""
    global index, chunks_db

    # Convert to absolute path if necessary
    if not os.path.isabs(pdf_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        pdf_path = os.path.join(project_root, pdf_path)

    print(f"Checking PDF:  {pdf_path}")
    print(f"Exists: {os.path.exists(pdf_path)}")

    if os.path.exists(pdf_path):
        print(f"PDF found, initializing...")
        build_knowledge_base(pdf_path, force_rebuild=force_rebuild)

        # === DEBUG INFO ===
        print(f"\nPost-build status:")
        print(f"- index type: {type(index)}")
        print(f"- index is None: {index is None}")
        if index is not None:
            print(f"- index.ntotal: {index.ntotal}")
        print(f"- chunks_db type: {type(chunks_db)}")
        print(f"- chunks_db length: {len(chunks_db)}")

    else:
        print(f"PDF not found:  {pdf_path}")
        print("Using fallback knowledge base")

        # Fallback
        chunks_db = [
            {"text": "RPE 10 = maksymalne obciążenie (0 powtórzeń w zapasie)",
             "page": 1, "source": "fallback", "chunk_id": 0},
            {"text": "Zasada przysiadu IPF:  biodra poniżej kolan",
             "page": 1, "source": "fallback", "chunk_id": 1}
        ]

        index = faiss.IndexFlatL2(dimension)
        fallback_texts = [c["text"] for c in chunks_db]
        fallback_embs = embedder.encode(fallback_texts, convert_to_numpy=True)
        faiss.normalize_L2(fallback_embs)
        index.add(fallback_embs.astype('float32'))
        print(f"Fallback KB loaded with {len(chunks_db)} entries, {index.ntotal} vectors")