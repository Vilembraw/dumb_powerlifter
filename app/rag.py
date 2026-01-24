import json
import os
import re
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
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
bm25 = None

def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25"""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_knowledge_base(pdf_path: str, force_rebuild=False):
    """Builds the knowledge base from the given PDF file."""
    global index, chunks_db, bm25
    if not force_rebuild and os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
        # Load existing index and chunks
        print("Loading existing knowledge base...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks_db = json.load(f)
        bm25_corpus = [tokenize(c["text"]) for c in chunks_db]
        bm25 = BM25Okapi(bm25_corpus)
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

    # Build BM25 index
    bm25_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(bm25_corpus)
    print(f"BM25 index built with {len(bm25_corpus)} documents")

    # Save index and chunks
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_db, f, ensure_ascii=False, indent=2)

    print("Knowledge base built and saved.")


def search_dense(query: str, top_k: int = 10) -> List[tuple]:
    """Dense vector search (FAISS)"""
    if index is None or not chunks_db:
        return []

    query_vector = embedder.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, top_k)

    return [(float(distances[0][i]), idx) for i, idx in enumerate(indices[0]) if 0 <= idx < len(chunks_db)]


def search_bm25(query: str, top_k: int = 10) -> List[tuple]:
    """Sparse text search (BM25)"""
    if bm25 is None or not chunks_db:
        return []

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(float(scores[idx]), idx) for idx in top_indices if scores[idx] > 0]


def rrf_fusion(dense_results: List[tuple], sparse_results: List[tuple], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion of dense and sparse search results."""
    scores = {}

    # Score from dense search
    for rank, (score, idx) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)

    # Score from sparse search
    for rank, (score, idx) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)

    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in ranked]


def search_knowledge(query: str, top_k: int = 3, use_hybrid: bool = True) -> List[Dict[str, Any]]:
    """
    Hybrid search combining dense vectors and BM25
    """
    if index is None or not chunks_db:
        print("Knowledge base not loaded.")
        return []

    # Encode query ONCE
    query_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_vec)

    dense_idxs = set()
    sparse_idxs = set()
    idx_to_score = {}

    if not use_hybrid or bm25 is None:
        # Pure dense search
        distances, indices_arr = index.search(query_vec, top_k)
        indices = [int(idx) for idx in indices_arr[0] if 0 <= idx < len(chunks_db)]
        idx_to_score = {int(indices_arr[0][i]): float(distances[0][i]) for i in range(len(indices_arr[0]))}
    else:
        # Dense search for fusion
        distances, indices_arr = index.search(query_vec, top_k * 2)
        dense_results = [(float(distances[0][i]), int(indices_arr[0][i]))
                         for i in range(len(indices_arr[0])) if 0 <= indices_arr[0][i] < len(chunks_db)]

        # Sparse search
        sparse_results = search_bm25(query, top_k=top_k * 2)

        # Fusion
        indices = rrf_fusion(dense_results, sparse_results, k=60)[:top_k]


        dense_idxs = {idx for _, idx in dense_results}

        sparse_idxs = {idx for _, idx in sparse_results}

        # Store dense scores
        idx_to_score = {idx: score for score, idx in dense_results}

    results = []
    for idx in indices:
        if 0 <= idx < len(chunks_db):
            chunk = chunks_db[idx]

            if idx in dense_idxs and idx in sparse_idxs:
                method = "both"  # Hybrid
            elif idx in sparse_idxs:
                method = "bm25"  # Keyword
            else:
                method = "dense"  # Semantic


            # Use cached score from FAISS, or compute if needed
            if idx in idx_to_score:
                score = idx_to_score[idx]
            else:
                # Compute similarity score
                chunk_vec = embedder.encode([chunk["text"]], convert_to_numpy=True).astype('float32')
                faiss.normalize_L2(chunk_vec)
                score = float(np.dot(query_vec[0], chunk_vec[0]))

            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "retrieved_by": method
            })

    return results


def init_rag(pdf_path: str = "data/poliquin_picp_level_1.pdf", force_rebuild: bool = False):
    """Initializes the RAG system by building/loading the knowledge base."""
    global index, chunks_db, bm25

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
        print(f"\n{'=' * 60}")
        print(f"POST-BUILD STATUS")
        print(f"{'=' * 60}")
        print(f"- index: {type(index).__name__ if index else 'None'}")
        print(f"- index is None: {index is None}")
        if index is not None:
            print(f"- index.ntotal: {index.ntotal}")
        print(f"- chunks_db: {type(chunks_db).__name__}")
        print(f"- chunks_db length: {len(chunks_db)}")
        print(f"- bm25: {'✓ Ready' if bm25 else '✗ Not initialized'}")
        print(f"{'=' * 60}\n")

        if index is None:
            raise RuntimeError("❌ FAISS index failed to load!")
        if not chunks_db:
            raise RuntimeError("❌ chunks_db is empty!")
        if bm25 is None:
            raise RuntimeError("❌ BM25 index failed to initialize!")

        print(f"RAG system ready with {len(chunks_db)} chunks\n")

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

        fallback_texts = [c["text"] for c in chunks_db]
        fallback_embs = embedder.encode(fallback_texts, convert_to_numpy=True)
        faiss.normalize_L2(fallback_embs)
        index = faiss.IndexFlatL2(dimension)
        index.add(fallback_embs.astype('float32'))
        bm25_corpus = [tokenize(text) for text in fallback_texts]
        bm25 = BM25Okapi(bm25_corpus)
        print(f"Fallback KB loaded with {len(chunks_db)} entries, {index.ntotal} vectors")