"""Persistent conversations (SQLite), search, export and automatic summaries."""
from __future__ import annotations

import json
import re
import threading
import uuid

import config as cfg
import database
import llm
import memory

SUMMARY_PROMPT = (
    "You maintain compact memory notes for an Indian AYUSH / Ayurveda regulatory assistant. "
    "Update the notes using the previous notes and the new messages. Write in English, at most "
    "170 words, using only these headings when they have content: Topics / User info / Decisions / "
    "Key Q&A / Pending.\n"
    "Rules: copy Act names, section and rule numbers, dates and notification numbers EXACTLY as "
    "they appear in the messages; never add any that are not there; never state that something is "
    "currently in force; keep it factual."
)


_FOOTER = re.compile(r"\n\n\*\*(?:Sources|स्रोत):\*\*")


def _like(q: str) -> str:
    q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{q}%"


def make_title(question: str) -> str:
    t = " ".join(question.split())
    return (t[:57] + "...") if len(t) > 60 else (t or "New conversation")


def create_conversation(user_id: str, title: str) -> str:
    cid, ts = uuid.uuid4().hex, database.now()
    conn = database.get_conn()
    with conn:
        conn.execute("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                     (cid, user_id, title, ts, ts))
    return cid


def get_conversation(cid: str, user_id: str):
    return database.get_conn().execute(
        "SELECT * FROM conversations WHERE id=? AND user_id=?", (cid, user_id)).fetchone()


def list_conversations(user_id: str, query: str = "", limit: int = 30) -> list[dict]:
    conn = database.get_conn()
    if query.strip():
        p = _like(query.strip())
        rows = conn.execute(
            "SELECT c.id,c.title,c.updated_at FROM conversations c WHERE c.user_id=? AND ("
            "c.title LIKE ? ESCAPE '\\' OR c.summary LIKE ? ESCAPE '\\' OR EXISTS ("
            "SELECT 1 FROM messages m WHERE m.conversation_id=c.id AND m.content LIKE ? ESCAPE '\\')) "
            "ORDER BY c.updated_at DESC LIMIT ?", (user_id, p, p, p, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT id,title,updated_at FROM conversations WHERE user_id=? "
            "ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_messages(cid: str, user_id: str) -> list[dict]:
    if not get_conversation(cid, user_id):
        return []
    rows = database.get_conn().execute(
        "SELECT role,content,meta,created_at FROM messages WHERE conversation_id=? AND user_id=? "
        "ORDER BY id", (cid, user_id)).fetchall()
    out = []
    for r in rows:
        try:
            meta = json.loads(r["meta"] or "{}")
        except ValueError:
            meta = {}
        out.append({"role": r["role"], "content": r["content"], "meta": meta, "created_at": r["created_at"]})
    return out


def add_message(cid: str, user_id: str, role: str, content: str, meta: dict | None = None) -> None:
    if not get_conversation(cid, user_id):
        raise PermissionError("Conversation does not belong to this user.")
    ts, conn = database.now(), database.get_conn()
    with conn:
        conn.execute("INSERT INTO messages(conversation_id,user_id,role,content,meta,created_at) "
                     "VALUES(?,?,?,?,?,?)", (cid, user_id, role, content, json.dumps(meta or {}, ensure_ascii=False), ts))
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?", (ts, cid, user_id))


def count_messages(cid: str, user_id: str) -> int:
    return database.get_conn().execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id=? AND user_id=?", (cid, user_id)).fetchone()[0]


def delete_conversation(cid: str, user_id: str) -> None:
    conn = database.get_conn()
    with conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=? AND user_id=?", (cid, user_id))
        conn.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, user_id))
    memory.delete_conversation_memory(user_id, cid)


# ---- export ----------------------------------------------------------------
def export_all_json(user_id: str) -> str:
    data = []
    for c in list_conversations(user_id, limit=100000):
        data.append({"id": c["id"], "title": c["title"], "updated_at": c["updated_at"],
                     "messages": [{"role": m["role"], "content": m["content"], "time": m["created_at"]}
                                  for m in get_messages(c["id"], user_id)]})
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_conversation_text(cid: str, user_id: str) -> str:
    conv = get_conversation(cid, user_id)
    if not conv:
        return ""
    lines = [f"# {conv['title']}", f"Started: {conv['created_at']}", ""]
    for m in get_messages(cid, user_id):
        lines.append(f"[{m['created_at']}] {m['role'].upper()}:\n{m['content']}\n")
    return "\n".join(lines)


# ---- automatic summaries ---------------------------------------------------
_running: set[str] = set()
_lock = threading.Lock()


def _short(text: str, n: int = 700) -> str:
    return text if len(text) <= n else text[:n] + "..."


def maybe_summarize(cid: str, user_id: str, force: bool = False) -> bool:
    """Summarise new messages. Normal mode waits for SUMMARY_EVERY messages; force needs >= 2."""
    conv = get_conversation(cid, user_id)
    if not conv:
        return False
    msgs = get_messages(cid, user_id)
    new = msgs[conv["summarized_upto"]:]
    if len(new) < (2 if force else cfg.SUMMARY_EVERY):
        return False
    transcript = "\n".join(f"{m['role'].upper()}: {_short(_FOOTER.split(m['content'])[0])}" for m in new)
    prompt = (f"Previous notes:\n{conv['summary'] or '(none)'}\n\nNew messages:\n{transcript}\n\nUpdated notes:")
    try:
        summary = llm.chat_once(
            [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": prompt}],
            cfg.SUMMARY_OPTIONS)
    except Exception:
        return False
    docs = []  # code-generated, so document names/pages are always accurate
    for m in msgs:
        for s in m["meta"].get("sources", []) if m["role"] == "assistant" else []:
            ref = f"{s.get('source')} p.{s.get('page')}"
            if ref not in docs:
                docs.append(ref)
    if docs:
        summary += "\nDocuments retrieved: " + "; ".join(docs[:8])
    conn = database.get_conn()
    with conn:
        conn.execute("UPDATE conversations SET summary=?, summarized_upto=? WHERE id=? AND user_id=?",
                     (summary, len(msgs), cid, user_id))
    memory.upsert_memory(user_id, cid, conv["title"], summary)
    return True


def summarize_async(cid: str | None, user_id: str, force: bool = False) -> None:
    """Run the summary in the background so the chat never waits for it."""
    if not cid:
        return
    with _lock:
        if cid in _running:
            return
        _running.add(cid)

    def _run():
        try:
            maybe_summarize(cid, user_id, force)
        except Exception:
            pass
        finally:
            with _lock:
                _running.discard(cid)

    threading.Thread(target=_run, daemon=True).start()
