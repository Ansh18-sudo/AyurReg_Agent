"""All Ollama calls: streaming chat, one-shot chat, embeddings, warm-up."""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Iterator

import ollama

import config as cfg


@lru_cache(maxsize=1)
def client() -> ollama.Client:
    return ollama.Client(host=cfg.OLLAMA_HOST)


def is_available() -> bool:
    try:
        client().list()
        return True
    except Exception:
        return False


def warmup_async() -> None:
    """Load the model into memory in the background so the first question is fast."""
    def _run():
        try:
            client().generate(model=cfg.LLM_MODEL, prompt="", keep_alive=cfg.KEEP_ALIVE)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def stream_chat(messages: list[dict]) -> Iterator[str]:
    for chunk in client().chat(
        model=cfg.LLM_MODEL,
        messages=messages,
        stream=True,
        keep_alive=cfg.KEEP_ALIVE,
        options=cfg.LLM_OPTIONS,
    ):
        piece = chunk["message"]["content"]
        if piece:
            yield piece


def chat_once(messages: list[dict], options: dict | None = None) -> str:
    resp = client().chat(
        model=cfg.LLM_MODEL,
        messages=messages,
        stream=False,
        keep_alive=cfg.KEEP_ALIVE,
        options=options or cfg.LLM_OPTIONS,
    )
    return resp["message"]["content"].strip()


@lru_cache(maxsize=1)
def _default_ef():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def embed(texts: list[str]) -> list[list[float]]:
    if cfg.EMBED_BACKEND == "ollama":
        resp = client().embed(model=cfg.OLLAMA_EMBED_MODEL, input=texts, keep_alive=cfg.KEEP_ALIVE)
        return [list(map(float, v)) for v in resp["embeddings"]]
    return [[float(x) for x in v] for v in _default_ef()(texts)]
