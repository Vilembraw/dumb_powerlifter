import os
import re
from typing import List, Dict

from pypdf import PdfReader


def load_pdf(pdf_path: str) -> List[Dict]:
    """Loads a PDF file and extracts text from each page."""
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            # Normalize whitespace (replace multiple spaces/newlines with single space)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:
                pages.append({
                    "source": os.path.basename(pdf_path),
                    "page": i,
                    "text": text
                })
        except Exception as e:
            print(f"Error extracting page {i} from {pdf_path}: {e}")
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits a text into overlapping chunks for better context preservation."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to end at a sentence boundary
        if end < text_len:
            last_period = text.rfind('.', start, end)
            # Only break at period if it's reasonably close to the chunk end
            if last_period > start + chunk_size // 2:
                end = last_period + 1

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move start forward with overlap
        start = end - overlap if end < text_len else text_len
    return chunks

def chunk_pages(pages: List[Dict], chunk_size: int = 600, overlap: int = 100) -> List[Dict]:
    """Chunks the text of each page into smaller overlapping pieces."""
    all_chunks = []

    for page_data in pages:
        page_text = page_data["text"]
        text_chunks = chunk_text(page_text, chunk_size, overlap)

        for idx, chunk_content in enumerate(text_chunks):
            all_chunks.append({
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_id": idx + 1,
                "text": chunk_content
            })

    print(f"Found {len(all_chunks)} chunks")
    return all_chunks



