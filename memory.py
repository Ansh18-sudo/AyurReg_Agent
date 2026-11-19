"""Personal memory: summaries in SQLite + vectors in a SEPARATE Chroma collection.
Every query is filtered by user_id, so one person's memory is never visible to another."""
from __future__ import annotations

import config as cfg
import database
import llm
import rag


def _col():
    return rag.get_client().get_or_create_collection(
        cfg.MEMORY_COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )


def count_memories(user_id: str) -> int:
    return database.get_conn().execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=?", (user_id,)).fetchone()[0]


def upsert_memory(user_id: str, conversation_id: str, title: str, summary: str) -> None:
    conn, ts = database.get_conn(), database.now()
    with conn:
        conn.execute(
            "INSERT INTO memories(id,user_id,conversation_id,title,summary,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "title=excluded.title, summary=excluded.summary, updated_at=excluded.updated_at "
            "WHERE memories.user_id=excluded.user_id",
            (conversation_id, user_id, conversation_id, title, summary, ts, ts),
        )
    _col().upsert(
        ids=[conversation_id],
        documents=[summary],
        embeddings=llm.embed([f"{title}\n{summary}"]),
        metadatas=[{"user_id": user_id, "conversation_id": conversation_id, "title": title}],
    )


def retrieve(user_id: str, query: str, exclude_conversation_id: str | None = None) -> list[dict]:
    """Return only the few most relevant memories of THIS user."""
    if count_memories(user_id) == 0:
        return []
    where = {"user_id": user_id}
    if exclude_conversation_id:
        where = {"$and": [{"user_id": user_id}, {"conversation_id": {"$ne": exclude_conversation_id}}]}
    try:
        res = _col().query(query_embeddings=llm.embed([query]), n_results=cfg.MEMORY_TOP_K,
                           where=where, include=["documents", "metadatas", "distances"])
    except Exception:
        return []
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= cfg.MEMORY_MAX_DISTANCE:
            out.append({"title": meta.get("title", ""), "text": doc[:cfg.MEMORY_CHARS],
                        "distance": float(dist)})
    return out


def list_memories(user_id: str) -> list[dict]:
    rows = database.get_conn().execute(
        "SELECT conversation_id, title, summary, updated_at FROM memories "
        "WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_conversation_memory(user_id: str, conversation_id: str) -> None:
    conn = database.get_conn()
    with conn:
        conn.execute("DELETE FROM memories WHERE id=? AND user_id=?", (conversation_id, user_id))
    try:
        _col().delete(ids=[conversation_id], where={"user_id": user_id})
    except Exception:
        pass


def clear_user_memory(user_id: str) -> None:
    """Delete this user's memories and summaries. Chat transcripts are kept."""
    conn = database.get_conn()
    with conn:
        conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        conn.execute("UPDATE conversations SET summary='', summarized_upto=0 WHERE user_id=?", (user_id,))
    try:
        _col().delete(where={"user_id": user_id})
    except Exception:
        pass


# ---- preferences ----------------------------------------------------------
def get_pref(user_id: str, key: str, default: str = "") -> str:
    row = database.get_conn().execute(
        "SELECT value FROM preferences WHERE user_id=? AND key=?", (user_id, key)).fetchone()
    return row["value"] if row else default


def set_pref(user_id: str, key: str, value: str) -> None:
    conn = database.get_conn()
    with conn:
        conn.execute(
            "INSERT INTO preferences(user_id,key,value) VALUES(?,?,?) "
            "ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value", (user_id, key, value))
