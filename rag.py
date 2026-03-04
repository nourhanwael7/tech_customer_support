"""
rag.py — Retrieval-Augmented Generation pipeline
Loads text/PDF files, creates embeddings with FAISS, retrieves relevant context.
"""

import os
import re
import pickle
from typing import Optional

import faiss
import numpy as np

# ── Optional PDF support ───────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── Embedding model ────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _embedder = None  # lazy load

    def get_embedder() -> SentenceTransformer:
        global _embedder
        if _embedder is None:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return _embedder

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    def get_embedder():
        raise RuntimeError("sentence-transformers not installed.")


KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
INDEX_CACHE = os.path.join(os.path.dirname(__file__), "data", "rag_index.pkl")


class RAGPipeline:
    """
    Manages document loading, chunking, embedding, and retrieval.
    """

    def __init__(self):
        self.chunks: list[str] = []         # raw text chunks
        self.sources: list[str] = []        # source filename per chunk
        self.index: Optional[faiss.IndexFlatL2] = None
        self.dimension: int = 384           # all-MiniLM-L6-v2 dimension

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_text_file(self, filepath: str) -> list[str]:
        """Load a plain-text or .txt file, return list of chunks."""
        try:
            with open(filepath, encoding="utf-8") as f:
                text = f.read()
            return self._chunk_text(text, source=os.path.basename(filepath))
        except Exception as e:
            print(f"[RAG] Error loading {filepath}: {e}")
            return []

    def load_pdf_file(self, filepath: str) -> list[str]:
        """Load a PDF file, return list of chunks."""
        if not PDF_SUPPORT:
            print("[RAG] pypdf not installed — skipping PDF.")
            return []
        try:
            reader = PdfReader(filepath)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages)
            return self._chunk_text(text, source=os.path.basename(filepath))
        except Exception as e:
            print(f"[RAG] Error loading PDF {filepath}: {e}")
            return []

    def load_knowledge_base(self, directory: Optional[str] = None) -> int:
        """
        Load all .txt and .pdf files from the knowledge_base directory.
        Returns total number of chunks loaded.
        """
        directory = directory or KNOWLEDGE_BASE_DIR
        os.makedirs(directory, exist_ok=True)

        new_chunks: list[str] = []
        new_sources: list[str] = []

        for fname in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fname)
            if fname.endswith(".txt"):
                chunks = self.load_text_file(fpath)
            elif fname.endswith(".pdf"):
                chunks = self.load_pdf_file(fpath)
            else:
                continue

            new_chunks.extend(chunks)
            new_sources.extend([fname] * len(chunks))

        self.chunks.extend(new_chunks)
        self.sources.extend(new_sources)

        if new_chunks:
            self._build_index()

        return len(new_chunks)

    def add_document(self, text: str, source: str = "uploaded") -> int:
        """
        Add a document (raw text) to the pipeline at runtime.
        Returns number of chunks added.
        """
        chunks = self._chunk_text(text, source=source)
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        if chunks:
            self._build_index()
        return len(chunks)

    def add_pdf_bytes(self, pdf_bytes: bytes, source: str = "uploaded.pdf") -> int:
        """
        Add a PDF from raw bytes (from Streamlit file_uploader).
        """
        if not PDF_SUPPORT:
            return 0
        try:
            import io
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
            return self.add_document(text, source=source)
        except Exception as e:
            print(f"[RAG] Error adding PDF bytes: {e}")
            return 0

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        source: str = "unknown",
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> list[str]:
        """
        Split text into overlapping chunks of ~chunk_size words.
        Returns list of chunk strings (each prefixed with [source]).
        """
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(f"[Source: {source}]\n{chunk}")
            if end == len(words):
                break
            start += chunk_size - overlap

        return chunks

    # ── Indexing ───────────────────────────────────────────────────────────────

    def _build_index(self):
        """Build or rebuild FAISS index from all current chunks."""
        if not EMBEDDING_AVAILABLE or not self.chunks:
            return

        try:
            embedder = get_embedder()
            embeddings = embedder.encode(self.chunks, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype="float32")

            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            self.dimension = embeddings.shape[1]
        except Exception as e:
            print(f"[RAG] Error building FAISS index: {e}")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """
        Retrieve top_k most relevant chunks for a query.
        Returns list of chunk strings.
        """
        if not self.chunks:
            return []

        if self.index is None or not EMBEDDING_AVAILABLE:
            # Fallback: simple keyword matching
            return self._keyword_search(query, top_k)

        try:
            embedder = get_embedder()
            q_emb = embedder.encode([query], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype="float32")

            distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    results.append(self.chunks[idx])
            return results
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[str]:
        """Fallback: BM25-style keyword matching."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k] if _ > 0]

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return len(self.chunks) > 0

    @property
    def document_count(self) -> int:
        return len(set(self.sources))

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def get_sources(self) -> list[str]:
        return sorted(set(self.sources))

    def clear(self):
        self.chunks = []
        self.sources = []
        self.index = None


# ── Singleton ──────────────────────────────────────────────────────────────────

_rag_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Return the singleton RAG pipeline, initialized if needed."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
        loaded = _rag_instance.load_knowledge_base()
        print(f"[RAG] Loaded {loaded} chunks from knowledge base.")
    return _rag_instance


def reset_rag_pipeline() -> RAGPipeline:
    """Force re-initialization of the RAG pipeline."""
    global _rag_instance
    _rag_instance = None
    return get_rag_pipeline()
