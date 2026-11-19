"""Central configuration for A1. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"      # put regulatory PDFs here (or upload in the sidebar)
CHROMA_DIR = DATA_DIR / "chroma_db"       # vector store (regulatory docs + per-user memory)
DB_PATH = DATA_DIR / "a1_memory.db"       # SQLite: users, conversations, messages, memories

for _d in (DATA_DIR, KB_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Ollama / LLM -----------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("A1_LLM_MODEL", "llama3.2:3b")
KEEP_ALIVE = "30m"  # keep the model loaded in memory for 30 minutes
LLM_OPTIONS = {"num_ctx": 4096, "num_predict": 800, "temperature": 0.2, "top_p": 0.9}
SUMMARY_OPTIONS = {"num_ctx": 3072, "num_predict": 320, "temperature": 0.1}

# ---- Embeddings -------------------------------------------------------------
# "default": Chroma's built-in MiniLM model (no extra setup, English-centred).
# "ollama":  an Ollama embedding model. For Hindi + English documents use:
#            `ollama pull bge-m3`, then set A1_EMBED_BACKEND=ollama and re-index.
EMBED_BACKEND = os.getenv("A1_EMBED_BACKEND", "default").lower()
OLLAMA_EMBED_MODEL = os.getenv("A1_EMBED_MODEL", "bge-m3")
_tag = "minilm" if EMBED_BACKEND == "default" else OLLAMA_EMBED_MODEL
EMBED_TAG = re.sub(r"[^a-zA-Z0-9]+", "_", _tag).strip("_").lower()
REG_COLLECTION = f"regulatory_{EMBED_TAG}"       # official documents (shared, read-only knowledge)
MEMORY_COLLECTION = f"user_memory_{EMBED_TAG}"   # personal conversation memory (per user)

# ---- Retrieval --------------------------------------------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 6                    # regulatory chunks sent to the LLM
RAG_MAX_DISTANCE = 0.85      # cosine distance above this = treated as irrelevant
MEMORY_TOP_K = 2             # personal memories sent to the LLM
MEMORY_MAX_DISTANCE = 0.75
MEMORY_CHARS = 500

# ---- Session memory ---------------------------------------------------------
HISTORY_MESSAGES = 6         # last N messages of this session sent to the LLM
HISTORY_CHARS = 600          # each old message is trimmed to this length
SUMMARY_EVERY = 10           # summarise after this many new messages
SUPPORTED_LANGUAGES = ["Auto", "English", "Hindi", "Marathi", "Bengali", "Punjabi",
                       "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam"]
