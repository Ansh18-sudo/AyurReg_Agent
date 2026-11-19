"""A1 | Ayurveda Regulation Assistant - Streamlit UI.  Run: streamlit run app.py"""
from __future__ import annotations

import time

import streamlit as st

import agent
import auth
import config as cfg
import conversation
import database
import llm
import memory
import rag

st.set_page_config(page_title="A1 | Ayurveda Regulation Assistant", page_icon="🌿", layout="wide")


@st.cache_resource
def bootstrap() -> bool:
    database.init_db()
    rag.index_folder()      # indexes new PDFs found in data/knowledge_base
    llm.warmup_async()      # load the model in the background
    return True


@st.cache_data(ttl=20)
def ollama_ok() -> bool:
    return llm.is_available()


# ---------------------------------------------------------------- session helpers
def reset_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []


def leave_conversation(uid: str) -> None:
    """Summarise the conversation we are leaving (in the background) so it becomes memory."""
    conversation.summarize_async(st.session_state.get("conversation_id"), uid, force=True)


def open_conversation(uid: str, cid: str) -> None:
    leave_conversation(uid)
    st.session_state.conversation_id = cid
    st.session_state.messages = conversation.get_messages(cid, uid)


# ---------------------------------------------------------------- rendering
def render_meta(meta: dict) -> None:
    if not meta:
        return
    c1, c2, c3 = st.columns(3)
    c1.caption(f"Category: {meta.get('category', '-')}")
    c2.caption(f"Evidence confidence: {meta.get('confidence', '-')}")
    c3.caption(f"Personal memories used: {meta.get('memory_used', 0)}")
    sources = meta.get("sources") or []
    if sources:
        with st.expander("Retrieved source documents"):
            for s in sources:
                st.markdown(f"**[{s['label']}] {s['source']}**, page {s['page']}  (distance {s['distance']})")
                st.caption(s["text"])


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_meta(m.get("meta", {}))


# ---------------------------------------------------------------- login screen
def login_screen() -> None:
    st.title("🌿 A1 | Ayurveda Regulation Assistant")
    st.caption("Log in so A1 can keep your own private chat history and memory.")
    tab_in, tab_up = st.tabs(["Log in", "Create account"])
    with tab_in:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                user = auth.login(u, p)
                if user:
                    st.session_state.user = user
                    reset_chat()
                    st.rerun()
                time.sleep(0.8)
                st.error("Invalid username or password.")
    with tab_up:
        with st.form("register_form"):
            name = st.text_input("Your name (optional)")
            u = st.text_input("Choose a username")
            p = st.text_input("Choose a password (min 8 characters)", type="password")
            if st.form_submit_button("Create account"):
                ok, msg, user = auth.register(u, p, name)
                if ok:
                    st.session_state.user = user
                    reset_chat()
                    st.rerun()
                st.error(msg)


# ---------------------------------------------------------------- sidebar
def sidebar(user: dict) -> None:
    uid = user["id"]
    with st.sidebar:
        st.markdown(f"**👤 {user['display_name']}**  \n`{user['username']}`")
        if st.button("Log out"):
            leave_conversation(uid)
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.subheader("Knowledge base")
        st.write(f"Indexed document chunks: **{rag.count()}**")
        st.write("Ollama: " + ("🟢 Available" if ollama_ok() else "🔴 Not reachable"))
        files = st.file_uploader("Add regulatory documents", type=["pdf"], accept_multiple_files=True,
                                 key=f"uploader_{st.session_state.get('upload_n', 0)}")
        if st.button("Index uploaded documents", disabled=not files):
            with st.spinner("Indexing..."):
                for f in files:
                    data = f.getvalue()
                    (cfg.KB_DIR / f.name).write_bytes(data)
                    try:
                        n = rag.index_pdf(data, f.name)
                        (st.success if n else st.warning)(
                            f"{f.name}: {n} chunks" if n else f"{f.name}: no text found (scanned PDF?)")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
            st.session_state.upload_n = st.session_state.get("upload_n", 0) + 1

        st.divider()
        st.subheader("Conversation")
        if st.button("Start new conversation"):
            leave_conversation(uid)
            reset_chat()
            st.rerun()

        st.divider()
        st.subheader("Chat history")
        search = st.text_input("Search chat history", key="hist_search", placeholder="Search...")
        pending = st.session_state.get("pending_delete")
        if pending:
            conv = conversation.get_conversation(pending, uid)
            st.warning(f"Permanently delete “{conv['title'] if conv else ''}”?")
            a, b = st.columns(2)
            if a.button("Yes, delete", key="del_yes"):
                conversation.delete_conversation(pending, uid)
                if st.session_state.get("conversation_id") == pending:
                    reset_chat()
                st.session_state.pending_delete = None
                st.rerun()
            if b.button("Cancel", key="del_no"):
                st.session_state.pending_delete = None
                st.rerun()
        for c in conversation.list_conversations(uid, search, limit=25):
            col1, col2 = st.columns([5, 1])
            label = ("▶ " if c["id"] == st.session_state.get("conversation_id") else "") + c["title"]
            if col1.button(label, key=f"open_{c['id']}", help=c["updated_at"]):
                open_conversation(uid, c["id"])
                st.rerun()
            if col2.button("🗑", key=f"del_{c['id']}", help="Delete this conversation"):
                st.session_state.pending_delete = c["id"]
                st.rerun()

        st.divider()
        st.subheader("Memory")
        with st.expander("View saved memory"):
            mems = memory.list_memories(uid)
            if not mems:
                st.caption("No saved memory yet. Memory is created from your conversations.")
            for m in mems:
                st.markdown(f"**{m['title']}**  \n*{m['updated_at']}*")
                st.caption(m["summary"])
        if st.session_state.get("pending_clear"):
            st.warning("Delete all your saved memory and summaries? Chat transcripts are kept.")
            a, b = st.columns(2)
            if a.button("Yes, clear", key="clr_yes"):
                memory.clear_user_memory(uid)
                st.session_state.pending_clear = False
                st.rerun()
            if b.button("Cancel", key="clr_no"):
                st.session_state.pending_clear = False
                st.rerun()
        elif st.button("Clear personal memory"):
            st.session_state.pending_clear = True
            st.rerun()

        st.divider()
        st.subheader("Export & preferences")
        if st.button("Prepare export of my history"):
            st.session_state.export_data = conversation.export_all_json(uid)
        if st.session_state.get("export_data"):
            st.download_button("Download history (JSON)", st.session_state.export_data,
                               file_name="a1_chat_history.json", mime="application/json")
        cid = st.session_state.get("conversation_id")
        if cid:
            st.download_button("Download this chat (TXT)", conversation.export_conversation_text(cid, uid),
                               file_name="a1_conversation.txt", mime="text/plain")
        current = memory.get_pref(uid, "language", "Auto")
        idx = cfg.SUPPORTED_LANGUAGES.index(current) if current in cfg.SUPPORTED_LANGUAGES else 0
        choice = st.selectbox("Answer language", cfg.SUPPORTED_LANGUAGES, index=idx,
                              help="Auto = same language as your question.")
        if choice != current:
            memory.set_pref(uid, "language", choice)
        st.caption("Chat history and memory are stored locally in SQLite, separately for each user.")


# ---------------------------------------------------------------- main
def main() -> None:
    bootstrap()
    user = st.session_state.get("user")
    if not user:
        login_screen()
        return
    if "messages" not in st.session_state:
        reset_chat()
    uid = user["id"]
    sidebar(user)

    st.title("🌿 A1 | Ayurveda Regulation Assistant")
    st.caption("A multilingual, document-grounded assistant for licensing, labeling and IP questions.")
    for m in st.session_state.messages:
        render_message(m)

    question = st.chat_input("Ask about AYUSH licensing, labeling, or traditional knowledge...")
    if not question:
        return

    history = list(st.session_state.messages)      # only turns BEFORE this question
    if not st.session_state.get("conversation_id"):
        st.session_state.conversation_id = conversation.create_conversation(uid, conversation.make_title(question))
    cid = st.session_state.conversation_id

    conversation.add_message(cid, uid, "user", question)
    st.session_state.messages.append({"role": "user", "content": question, "meta": {}})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            prepared = agent.prepare(uid, cid, question, history, memory.get_pref(uid, "language", "Auto"))
            text = st.write_stream(agent.stream_tokens(prepared))
        except Exception as e:
            st.error(f"Could not get an answer. Is Ollama running and is '{cfg.LLM_MODEL}' installed? ({e})")
            return
        footer = agent.build_footer(text, prepared)
        st.markdown(footer)
        meta = prepared.meta()
        render_meta(meta)

    full = text + footer
    conversation.add_message(cid, uid, "assistant", full, meta)
    st.session_state.messages.append({"role": "assistant", "content": full, "meta": meta})
    conversation.summarize_async(cid, uid)          # only acts after SUMMARY_EVERY new messages


main()
