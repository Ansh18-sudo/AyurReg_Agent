"""Regulatory knowledge base (ChromaDB). Holds ONLY official documents, never personal chats."""
from __future__ import annotations

import hashlib
import io
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

import config as cfg
import llm


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=str(cfg.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def _collection():
    return get_client().get_or_create_collection(
        cfg.REG_COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )


def count() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0


def indexed_sources() -> set[str]:
    try:
        metas = _collection().get(include=["metadatas"])["metadatas"] or []
        return {m.get("source", "") for m in metas}
    except Exception:
        return set()


def chunk_text(text: str, size: int = cfg.CHUNK_SIZE, overlap: int = cfg.CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) >= 40:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def index_pdf(data: bytes, filename: str) -> int:
    """Index one PDF. Re-indexing the same file just updates it. Returns number of chunks."""
    reader = PdfReader(io.BytesIO(data))
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(hashlib.sha1(f"{filename}|{page_no}|{i}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": filename, "page": page_no})
    if not docs:
        return 0  # scanned/image-only PDF: needs OCR, which this app does not do
    col = _collection()
    for s in range(0, len(docs), 32):
        col.upsert(ids=ids[s:s + 32], documents=docs[s:s + 32],
                   metadatas=metas[s:s + 32], embeddings=llm.embed(docs[s:s + 32]))
    return len(docs)


def index_folder() -> dict[str, int]:
    """Index PDFs in data/knowledge_base that are not indexed yet."""
    done, results = indexed_sources(), {}
    for pdf in sorted(Path(cfg.KB_DIR).glob("*.pdf")):
        if pdf.name not in done:
            try:
                results[pdf.name] = index_pdf(pdf.read_bytes(), pdf.name)
            except Exception:
                results[pdf.name] = -1
    return results


def search(query: str, k: int = cfg.TOP_K) -> list[dict]:
    col = _collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_embeddings=llm.embed([query]), n_results=min(k, n),
                    include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= cfg.RAG_MAX_DISTANCE:
            out.append({"text": doc, "source": meta.get("source", "?"),
                        "page": meta.get("page"), "distance": float(dist)})
    return out
