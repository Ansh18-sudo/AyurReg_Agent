"""Orchestrates one answer: language, follow-up resolution, RAG, personal memory, prompt."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

import config as cfg
import llm
import memory
import rag

_SCRIPTS = [(lang, re.compile(p)) for lang, p in [
    ("Hindi", r"[\u0900-\u097F]"), ("Bengali", r"[\u0980-\u09FF]"), ("Punjabi", r"[\u0A00-\u0A7F]"),
    ("Gujarati", r"[\u0A80-\u0AFF]"), ("Tamil", r"[\u0B80-\u0BFF]"), ("Telugu", r"[\u0C00-\u0C7F]"),
    ("Kannada", r"[\u0C80-\u0CFF]"), ("Malayalam", r"[\u0D00-\u0D7F]")]]
_LATIN = re.compile(r"[A-Za-z]")
_FOOTER_SPLIT = re.compile(r"\n\n\*\*(?:Sources|स्रोत):\*\*")
_PRONOUNS = re.compile(r"\b(it|its|this|that|these|those|they|them|their|above|same|there|such)\b", re.I)
_CITE = re.compile(r"\[S(\d+)\]")

CATEGORIES = [
    ("Licensing", ["licen", "manufactur", "permit", "registration", "लाइसेंस", "अनुमति"]),
    ("Labeling", ["label", "packag", "ingredient", "लेबल"]),
    ("Advertising", ["advertis", "claim", "magic remed", "विज्ञापन"]),
    ("Intellectual Property", ["patent", "trademark", "tkdl", "geographical", "copyright", "पेटेंट"]),
    ("GMP / Quality", ["gmp", "quality", "good manufacturing", "pharmacopoeia", "schedule t"]),
    ("Import / Export", ["import", "export", "customs"]),
    ("Clinical / Safety", ["clinical", "trial", "adverse", "pharmacovigilance", "safety"]),
]

FOOT = {
    "English": ("Sources", "No specific source was cited.",
                "Please verify against the latest official Gazette notification or the "
                "Ministry of AYUSH / CDSCO before relying on this."),
    "Hindi": ("स्रोत", "कोई विशिष्ट स्रोत उद्धृत नहीं किया गया।",
              "कृपया उपयोग से पहले नवीनतम राजपत्र अधिसूचना या आयुष मंत्रालय / CDSCO से सत्यापित करें।"),
}


def detect_language(text: str) -> str:
    """Script-based detection in code (no LLM call). Latin script -> English."""
    best, best_n = None, 0
    for lang, pat in _SCRIPTS:
        n = len(pat.findall(text))
        if n > best_n:
            best, best_n = lang, n
    if best and best_n >= len(_LATIN.findall(text)):
        return best
    return "English"


def classify(question: str) -> str:
    q = question.lower()
    for name, words in CATEGORIES:
        if any(w in q for w in words):
            return name
    return "General"


def _clean(text: str) -> str:
    return _FOOTER_SPLIT.split(text)[0].strip()


def build_history(history: list[dict], lang: str) -> list[dict]:
    """Recent turns of THIS session, skipping turns in another language (prevents language drift)."""
    out = []
    for m in history[-cfg.HISTORY_MESSAGES:]:
        content = _clean(m["content"])
        if content and detect_language(content) == lang:
            out.append({"role": m["role"], "content": content[:cfg.HISTORY_CHARS]})
    return out


def retrieval_query(question: str, history: list[dict]) -> str:
    """Resolve follow-ups like 'Explain its licensing' without an extra LLM call."""
    prev = [m["content"] for m in history if m["role"] == "user"]
    if prev and (len(question.split()) <= 6 or _PRONOUNS.search(question)):
        return f"{prev[-1]} {question}"
    return question


def confidence(sources: list[dict]) -> str:
    if not sources:
        return "None"
    best = min(s["distance"] for s in sources)
    return "High" if best <= 0.45 else "Moderate" if best <= 0.65 else "Low"


SYSTEM = """You are A1, an assistant for Ayurveda and Indian AYUSH regulatory questions. Today is {today}.
Rules:
1. Give a complete, detailed answer. Explain the actual content of the sources: list every relevant requirement, type, condition or step they mention. Never just point to a page or document; write out what it says.
2. Use ONLY the numbered REGULATORY SOURCES for legal or regulatory statements, and put a citation like [S1] after each point.
3. If the sources do not cover part of the question, say which part is missing. NEVER invent Acts, sections, rules, schedules or notifications.
4. Keep current requirements separate from historical information, and say when something needs verification.
5. PERSONAL MEMORY holds notes from this user's earlier chats. Use it only for context. It is NOT evidence that any regulation is valid or current.
6. Use the earlier messages of this conversation to resolve words like "it", "its" or "that".
7. Reply in {lang}. Do not write a source list; it is added automatically."""


@dataclass
class Prepared:
    messages: list[dict]
    sources: list[dict]
    memories: list[dict]
    category: str
    confidence: str
    language: str

    def meta(self) -> dict:
        return {
            "category": self.category, "confidence": self.confidence, "language": self.language,
            "memory_used": len(self.memories),
            "sources": [{"label": f"S{i}", "source": s["source"], "page": s["page"],
                         "text": s["text"][:400], "distance": round(s["distance"], 3)}
                        for i, s in enumerate(self.sources, 1)],
        }


def prepare(user_id: str, conversation_id: str | None, question: str,
            history: list[dict], language_pref: str = "Auto") -> Prepared:
    lang = detect_language(question) if language_pref in ("", "Auto") else language_pref
    rq = retrieval_query(question, history)
    sources = rag.search(rq)
    memories = memory.retrieve(user_id, rq, conversation_id)

    reg = "\n\n".join(f"[S{i}] ({s['source']}, page {s['page']})\n{s['text']}" for i, s in enumerate(sources, 1)) \
        or "(No relevant regulatory source was retrieved. Say that you cannot confirm this from the knowledge base.)"
    parts = []
    if memories:
        parts.append("PERSONAL MEMORY (earlier chats of this user; not legal evidence):\n" +
                     "\n".join(f"- {m['title']}: {m['text']}" for m in memories))
    parts += [f"REGULATORY SOURCES:\n{reg}", f"QUESTION: {question}",
                f"Reply ONLY in {lang}. Give a full, detailed answer in bullet points from the sources. "
              f"Do not just name pages. Cite [S1], [S2] after each point."]
    messages = [{"role": "system", "content": SYSTEM.format(today=date.today().isoformat(), lang=lang)}]
    messages += build_history(history, lang)
    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return Prepared(messages, sources, memories, classify(question), confidence(sources), lang)


def stream_tokens(p: Prepared) -> Iterator[str]:
    yield from llm.stream_chat(p.messages)


def build_footer(answer: str, p: Prepared) -> str:
    """Source list generated by code from what the model actually cited."""
    label, none_cited, verify = FOOT.get(p.language, FOOT["English"])
    cited = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(p.sources)})
    lines = [f"- [S{n}] {p.sources[n - 1]['source']}, page {p.sources[n - 1]['page']}" for n in cited]
    body = "\n".join(lines) if lines else f"- {none_cited}"
    return f"\n\n**{label}:**\n{body}\n\n*{verify}*"
