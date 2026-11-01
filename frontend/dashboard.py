"""Streamlit dashboard for the Company Knowledge Assistant.

Every page reads and writes through the backend API (over HTTP, or in-process in embedded mode);
the UI never touches API keys, the vector store or documents directly.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from frontend.api_client import APIError, BackendClient
from frontend.components import (
    bar_chart, empty_state, html_table, inject_css, kpi_tiles, page_header, pill, section_title, skeletons,
    status_pill,
)

# Unset or "embedded": run the API inside this Streamlit process (needed on Streamlit Community Cloud,
# which runs only `streamlit run app.py`). Otherwise talk to a separately started API server over HTTP.
BACKEND_URL = os.getenv("BACKEND_URL", "").strip() or "embedded"
EMBEDDED = BACKEND_URL.lower() == "embedded"
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_history.json"
# One shared server process serves every visitor in embedded mode, so a history *file* would show each
# visitor everyone else's conversations. There, history lives in the browser session only.
HISTORY_MODE = os.getenv("CHAT_HISTORY", "session" if EMBEDDED else "file").strip().lower()

NAV = [
    ("assistant", "💬  Assistant"),
    ("history", "🕘  Conversations"),
    ("kb", "📊  Knowledge Base"),
    ("upload", "⬆️  Upload Documents"),
    ("settings", "⚙️  Settings"),
]

DATA_POLICY_WARNING = (
    "**Data handling notice:** uploaded files are stored and indexed on this server, and relevant excerpts are "
    "sent to the configured AI provider to generate answers. Only upload documents you are authorised to share, and "
    "handle them according to your company's data classification and retention policies. Do not upload "
    "personal data, credentials, or regulated information unless your policies permit it."
)


# --------------------------------------------------------------------------- backend access
@st.cache_resource(show_spinner="Starting the knowledge base (indexing documents, loading models)…")
def _embedded_backend():
    """One in-process API per Streamlit server, shared by all sessions."""
    from fastapi.testclient import TestClient

    from main import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    client.__enter__()  # runs the app's startup: builds services and syncs the knowledge_base folder
    return client


def api() -> BackendClient:
    if EMBEDDED:
        return BackendClient("http://embedded", session=_embedded_backend())
    return BackendClient(BACKEND_URL)


@st.cache_data(ttl=10, show_spinner=False)
def cached_settings() -> Optional[Dict]:
    try:
        return api().settings()
    except APIError:
        return None


@st.cache_data(ttl=5, show_spinner=False)
def cached_documents() -> List[Dict]:
    try:
        return api().documents()
    except APIError:
        return []


@st.cache_data(ttl=15, show_spinner=False)
def cached_stats() -> Optional[Dict]:
    try:
        return api().stats()
    except APIError:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def cached_suggestions(limit: int = 6) -> Dict:
    return api().suggestions(limit)


def refresh_data() -> None:
    for fn in (cached_documents, cached_settings, cached_stats, cached_suggestions):
        fn.clear()


def kb_read_only() -> bool:
    """Fixed knowledge base: no uploads or document management in the UI (the API refuses them too)."""
    return bool((cached_settings() or {}).get("kb_read_only", True))


def indexed_docs() -> List[Dict]:
    return [d for d in cached_documents() if d.get("status") == "indexed"]


# --------------------------------------------------------------------------- conversations
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_history() -> List[Dict]:
    if HISTORY_MODE != "file":
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_history() -> None:
    if HISTORY_MODE != "file":
        return
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    convs = [c for c in st.session_state.conversations if c["messages"]]
    HISTORY_FILE.write_text(json.dumps(convs[:200], indent=1), encoding="utf-8")


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("dark_mode", False)
    ss.setdefault("page", "assistant")
    ss.setdefault("conversations", load_history())
    ss.setdefault("top_k", None)
    ss.setdefault("threshold", None)
    if "current_id" not in ss:
        new_conversation()


def new_conversation() -> None:
    conv = {"id": uuid.uuid4().hex[:12], "title": "New conversation", "created_at": _now(),
            "updated_at": _now(), "messages": []}
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_id = conv["id"]


# Navigation changes run as widget callbacks: the "page" radio's state may only be set before it renders.
def start_new_chat() -> None:
    current = current_conversation()
    if current["messages"]:
        new_conversation()
    st.session_state.page = "assistant"


def open_conversation(conv_id: str) -> None:
    st.session_state.current_id = conv_id
    st.session_state.page = "assistant"


def delete_conversation(conv_id: str) -> None:
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != conv_id]
    if st.session_state.current_id == conv_id:
        new_conversation()
    save_history()


def current_conversation() -> Dict:
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_id:
            return conv
    new_conversation()
    return st.session_state.conversations[0]


def ask(question: str) -> None:
    """Queue a question (from a suggestion or retry button) for the assistant page to send."""
    st.session_state.pending_question = question
    st.session_state.page = "assistant"


def retry_last(conv_id: str) -> None:
    """Drop the failed exchange and resend its question."""
    for conv in st.session_state.conversations:
        if conv["id"] == conv_id and len(conv["messages"]) >= 2 and conv["messages"][-1].get("error"):
            question = conv["messages"][-2]["content"]
            del conv["messages"][-2:]
            ask(question)


# --------------------------------------------------------------------------- sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='brand'><div class='brand-logo'>K</div><div><div class='brand-title'>Knowledge Assistant</div>"
            "<div class='brand-sub'>Grounded answers from company documents</div></div></div>",
            unsafe_allow_html=True,
        )
        st.button("＋  New chat", use_container_width=True, type="primary", on_click=start_new_chat)
        st.write("")
        labels = {k: v for k, v in NAV if not (k == "upload" and kb_read_only())}
        if st.session_state.get("page") not in labels:
            st.session_state.page = "assistant"
        st.radio("Navigation", list(labels), format_func=labels.get, key="page", label_visibility="collapsed")

        recent = [c for c in st.session_state.conversations if c["messages"]][:6]
        if recent:
            st.caption("Recent conversations")
            for conv in recent:
                label = ("● " if conv["id"] == st.session_state.current_id else "") + conv["title"][:34]
                st.button(label, key=f"recent-{conv['id']}", use_container_width=True,
                          on_click=open_conversation, args=(conv["id"],))

        st.divider()
        try:
            health = api().health()
            status = pill("● Online", "ok")
            if not health.get("llm_configured"):
                status += pill("API key missing", "warn")
            elif health.get("api_key_warning"):
                status += pill("API key looks invalid", "warn")
            if kb_read_only():
                status += pill("Fixed knowledge base", "info")
            st.markdown(status, unsafe_allow_html=True)
            st.caption(f"{health['documents']} documents · {health['chunks']} passages indexed · {health.get('model', '')}")
        except APIError:
            st.markdown(pill("● Backend offline", "err"), unsafe_allow_html=True)
            st.caption("The embedded backend failed to start - see the app logs." if EMBEDDED
                       else f"Expected at {BACKEND_URL}")
        st.toggle("Dark mode", key="dark_mode")


# --------------------------------------------------------------------------- answers & citations
_CITE = re.compile(r"\[(S\d+)\]")


def numbered_sources(msg: Dict) -> List[Dict]:
    """Sources with a display number. Older saved messages may predate the `number` field."""
    return [dict(s, number=s.get("number") or i) for i, s in enumerate(msg.get("sources") or [], start=1)]


def source_label(src: Dict) -> str:
    """Citation text built only from the source's stored metadata."""
    if src.get("citation"):
        return src["citation"]
    parts = [src["document"]] + ([f"p. {src['page']}"] if src.get("page") else []) + ([src["section"]] if src.get("section") else [])
    return " · ".join(parts)


def render_answer_text(msg: Dict) -> None:
    """The answer as Markdown (headings, lists, tables) with [S#] turned into numbered, hoverable citations."""
    by_id = {s["source_id"]: s for s in numbered_sources(msg)}
    # Escape HTML in the model's text first; only our own <sup> tags are allowed through. Markdown syntax
    # (#, -, |, **) is unaffected, so headings, lists and tables still render.
    text = escape(msg["content"], quote=False)

    def cite(match: re.Match) -> str:
        src = by_id.get(match.group(1))
        if not src:
            return ""
        return f"<sup class='cite' title='{escape(source_label(src))}'>[{src['number']}]</sup>"

    st.markdown(_CITE.sub(cite, text), unsafe_allow_html=True)


def highlighted_excerpt(src: Dict) -> str:
    html = escape(src["excerpt"])
    for sentence in src.get("highlights") or []:
        html = html.replace(escape(sentence), f"<mark>{escape(sentence)}</mark>", 1)
    return f"<div class='excerpt'>{html}</div>"


def neighbour_chunks(doc_id: str, index: Optional[int]) -> List[Dict]:
    """The chunks just before and after a cited chunk, for reading it in context."""
    if index is None:
        return []
    cache = st.session_state.setdefault("chunk_cache", {})
    if doc_id not in cache:
        cache[doc_id] = api().chunks(doc_id).get("chunks", [])
    by_index = {c.get("metadata", {}).get("chunk_index", c.get("chunk_index")): c for c in cache[doc_id]}
    return [by_index[i] for i in (index - 1, index + 1) if i in by_index]


def render_source_details(src: Dict, key: str) -> None:
    label = "Supporting sentences highlighted" if src.get("highlights") else "Retrieved passage"
    st.markdown(f"<div class='src-meta'>{label}</div>{highlighted_excerpt(src)}", unsafe_allow_html=True)
    st.write("")
    rows = [("Document", src["document"])]
    if src.get("page"):
        rows.append(("Page", f"{src['page']} of {src['total_pages']}" if src.get("total_pages") else str(src["page"])))
    else:
        rows.append(("Page", f"— ({src['page_note']})" if src.get("page_note") else "—"))
    rows.append(("Section", src.get("section") or "— (text before the first heading)"))
    if src.get("category"):
        rows.append(("Category", src["category"]))
    if src.get("chunk_index") is not None and src.get("chunk_total"):
        rows.append(("Passage", f"{src['chunk_index'] + 1} of {src['chunk_total']} in this document"))
    rows.append(("Similarity", f"{src['score']:.2f}"))
    if src.get("uploaded_at"):
        rows.append(("Indexed", src["uploaded_at"][:10]))
    table = "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>" for k, v in rows)
    st.markdown(f"<table class='src-table'>{table}</table>", unsafe_allow_html=True)

    if src.get("doc_id") and st.toggle("Show surrounding text", key=f"ctx-{key}"):
        try:
            for chunk in neighbour_chunks(src["doc_id"], src.get("chunk_index")):
                meta = chunk.get("metadata", {})
                where = " · ".join(str(x) for x in (meta.get("section"), f"p. {meta['page']}" if meta.get("page") and src.get("page") else None) if x)
                st.markdown(f"<div class='src-meta'>{escape(where or 'Adjacent passage')}</div>"
                            f"<div class='excerpt muted-excerpt'>{escape(chunk.get('text', ''))}</div>", unsafe_allow_html=True)
        except APIError as exc:
            st.caption(f"Could not load surrounding text: {exc}")


def source_card(src: Dict, key: str) -> None:
    """A citation card: number, document, location, category and match, with the excerpt one click away."""
    where = []
    if src.get("section"):
        where.append(escape(src["section"]))
    where.append(f"p. {src['page']}" + (f" of {src['total_pages']}" if src.get("total_pages") else "") if src.get("page")
                 else escape(src.get("page_note") or "No page numbers"))
    match = max(0.0, min(1.0, float(src.get("score") or 0)))
    with st.container(border=True):
        st.markdown(
            f"<div class='src-card'><div class='src-num'>{src['number']}</div><div style='min-width:0;flex:1'>"
            f"<div class='src-doc'>{escape(src['document'])}</div>"
            f"<div class='src-where'>{' · '.join(where)}</div>"
            + (pill(src["category"]) if src.get("category") else "")
            + f"<div class='meter' title='Semantic similarity {match:.2f}'><span style='width:{max(4, int(match * 100))}%'></span></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Show excerpt"):
            render_source_details(src, key)


def render_sources(msg: Dict, key: str) -> None:
    sources = numbered_sources(msg)
    by_id = {s["source_id"]: s for s in sources}
    if msg.get("conflicts"):
        items = "".join(
            f"<li>{escape(c['description'])} <span class='muted'>(see "
            + ", ".join(f"[{by_id[s]['number']}] {escape(by_id[s]['document'])}" if s in by_id else escape(s) for s in c["sources"])
            + ")</span></li>"
            for c in msg["conflicts"]
        )
        st.markdown(f"<div class='callout'><div class='title'>⚠ Conflicting information</div><ul>{items}</ul>"
                    "<div class='muted'>Confirm with the owning department before acting.</div></div>",
                    unsafe_allow_html=True)
    if sources:
        docs = len({s["document"] for s in sources})
        st.markdown(f"<div class='section-title' style='margin-top:.9rem'>Sources · {len(sources)} passage"
                    f"{'s' if len(sources) != 1 else ''} from {docs} document{'s' if docs != 1 else ''}</div>",
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, src in enumerate(sources):
            with cols[i % 2]:
                source_card(src, f"{key}-{src['source_id']}")
    retrieved = msg.get("retrieved") or []
    if retrieved:
        cited_ids = {s["chunk_id"] for s in sources}
        with st.expander(f"All retrieved passages ({len(retrieved)})"):
            st.caption("Everything the search sent to the model for this question, including passages the answer did not cite.")
            for r in retrieved:
                mark = " · ✅ cited" if r["chunk_id"] in cited_ids else ""
                st.markdown(f"**{escape(source_label(r))}** · similarity {r['score']:.2f}{mark}")
                st.markdown(f"<div class='excerpt'>{escape(r['excerpt'])}</div>", unsafe_allow_html=True)
    if msg.get("meta"):
        st.markdown(f"<div class='answer-meta'>{escape(msg['meta'])}</div>", unsafe_allow_html=True)


def render_message(msg: Dict, idx: int, conv_id: str = "", last: bool = False) -> None:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✨"):
        if msg.get("error"):
            st.error(msg["content"], icon="⚠️")
            if last:
                st.button("↻ Try again", key=f"retry-{idx}", on_click=retry_last, args=(conv_id,))
            return
        if msg["role"] == "assistant" and msg.get("found") is False:
            st.info(msg["content"], icon="🔎")
            st.caption("Try rephrasing, or check the Knowledge Base page to see which documents are indexed.")
        elif msg["role"] == "assistant":
            render_answer_text(msg)
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg, f"m{idx}")


def transcript_markdown(conv: Dict) -> str:
    """Markdown export that keeps each answer's verified sources."""
    blocks = [f"# {conv['title']}\n_{conv.get('created_at', '')[:16].replace('T', ' ')}_"]
    for m in conv["messages"]:
        if m["role"] == "user":
            blocks.append(f"**You:** {m['content']}")
        else:
            text = f"**Assistant:**\n\n{m['content']}"
            if m.get("sources"):
                refs = "\n".join(f"- [{s['number']}] {source_label(s)} ({s['source_id']})" for s in numbered_sources(m))
                text += f"\n\nSources:\n{refs}"
            blocks.append(text)
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- assistant page
def scope_filters(docs: List[Dict]):
    active = len(st.session_state.get("filter_cats") or []) + len(st.session_state.get("filter_docs") or [])
    with st.popover(f"🔎 Scope{f' · {active}' if active else ''}", use_container_width=True):
        st.caption("Limit answers to some categories or documents.")
        cats = sorted({d.get("category", "General") for d in docs})
        sel_cats = st.multiselect("Categories", cats, key="filter_cats")
        doc_opts = {d["filename"]: d["doc_id"] for d in docs if not sel_cats or d.get("category") in sel_cats}
        sel_docs = st.multiselect("Documents", list(doc_opts), key="filter_docs")
    return sel_cats, [doc_opts[n] for n in sel_docs if n in doc_opts]


def welcome(docs: List[Dict]) -> None:
    cats = sorted({d.get("category", "General") for d in docs})
    chips = "".join(pill(c) for c in cats[:8]) + (pill(f"+{len(cats) - 8} more", "info") if len(cats) > 8 else "")
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Company knowledge assistant</div>"
        "<h2>Ask anything about company policies and procedures</h2>"
        f"<p>Answers come only from the {len(docs)} indexed documents, with every statement cited. "
        "If the documents don't cover something, the assistant says so.</p>"
        f"{chips}</div>",
        unsafe_allow_html=True,
    )
    section_title("Suggested questions")
    placeholder = st.empty()
    placeholder.markdown(skeletons(3), unsafe_allow_html=True)
    try:
        suggestions = cached_suggestions(6).get("questions", [])
    except APIError:
        suggestions = []
    placeholder.empty()
    if not suggestions:
        st.caption("Suggestions are unavailable right now; type a question below.")
        return
    titles = {d["filename"]: d.get("category", "") for d in docs}
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                st.button(f"{s['question']}  \n:gray[{titles.get(s['document'], s.get('category', ''))}]",
                          key=f"sugg-{i}", use_container_width=True, on_click=ask, args=(s["question"],))


def page_assistant() -> None:
    conv = current_conversation()
    docs = indexed_docs()
    head, actions = st.columns([3, 2], vertical_alignment="bottom")
    with head:
        page_header("Assistant", conv["title"] if conv["messages"] else "New conversation", eyebrow="Chat")
    with actions:
        a1, a2, a3 = st.columns(3)
        with a1:
            sel_cats, sel_doc_ids = scope_filters(docs)
        a2.download_button("⬇ Export", transcript_markdown(conv), file_name=f"conversation-{conv['id']}.md",
                           mime="text/markdown", use_container_width=True, disabled=not conv["messages"])
        if a3.button("🗑 Clear", use_container_width=True, disabled=not conv["messages"]):
            conv["messages"], conv["title"] = [], "New conversation"
            save_history()
            st.rerun()

    if not conv["messages"]:
        if not docs:
            hint = ("Add documents to the <code>knowledge_base/</code> folder and restart the app." if kb_read_only()
                    else "Go to <b>Upload Documents</b> to add policies, SOPs or handbooks.")
            if cached_settings() is None:
                empty_state("⚠️", "The knowledge base is unavailable", "The backend isn't responding. Check the app logs.")
            else:
                empty_state("📭", "The knowledge base is empty", hint)
        else:
            welcome(docs)

    for i, msg in enumerate(conv["messages"]):
        render_message(msg, i, conv["id"], last=i == len(conv["messages"]) - 1)

    question = st.chat_input("Ask about policies, SOPs or guidelines…", disabled=not docs)
    question = question or st.session_state.pop("pending_question", None)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"] if not m.get("error")]
    user_msg = {"role": "user", "content": question}
    conv["messages"].append(user_msg)
    if conv["title"] == "New conversation":
        conv["title"] = question[:60] + ("…" if len(question) > 60 else "")
    render_message(user_msg, len(conv["messages"]))

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner(f"Searching {len(docs)} documents and drafting a cited answer…"):
            try:
                result = api().chat(
                    question, history,
                    categories=sel_cats or None,
                    doc_ids=sel_doc_ids or None,
                    top_k=st.session_state.top_k,
                    relevance_threshold=st.session_state.threshold,
                )
                n = len(result["sources"])
                meta = (f"{result.get('model') or 'answered without the AI model'} · "
                        f"{result.get('latency_ms', 0) / 1000:.1f}s" + (f" · {n} citation{'s' if n != 1 else ''}" if n else ""))
                msg = {"role": "assistant", "content": result["answer"], "found": result["found"],
                       "sources": result["sources"], "retrieved": result["retrieved"],
                       "conflicts": result["conflicts"], "meta": meta}
            except APIError as exc:
                msg = {"role": "assistant", "content": f"{exc}", "error": True}
    conv["messages"].append(msg)
    conv["updated_at"] = _now()
    save_history()
    st.rerun()


# --------------------------------------------------------------------------- conversations page
def _group(conv: Dict) -> str:
    try:
        when = datetime.fromisoformat(conv.get("updated_at") or conv.get("created_at"))
    except (TypeError, ValueError):
        return "Older"
    today = datetime.now().date()
    if when.date() == today:
        return "Today"
    if when.date() == today - timedelta(days=1):
        return "Yesterday"
    if when.date() > today - timedelta(days=7):
        return "Previous 7 days"
    return "Older"


def page_history() -> None:
    page_header("Conversations", "Reopen, export or delete past chats.", eyebrow="History")
    if HISTORY_MODE != "file":
        st.caption("🔒 Conversations are private to this browser session and are cleared when you close or reload the page.")
    convs = [c for c in st.session_state.conversations if c["messages"]]
    if not convs:
        empty_state("💬", "No conversations yet", "Ask a question on the Assistant page and it will appear here.")
        return
    query = st.text_input("Search conversations", placeholder="Search titles and messages…", label_visibility="collapsed")
    q = query.lower().strip()
    if q:
        convs = [c for c in convs if q in c["title"].lower() or any(q in m["content"].lower() for m in c["messages"])]
        if not convs:
            empty_state("🔍", "No matching conversations", f"Nothing matches “{escape(query)}”.")
            return
    groups: Dict[str, List[Dict]] = {}
    for c in convs:
        groups.setdefault(_group(c), []).append(c)
    for name in ("Today", "Yesterday", "Previous 7 days", "Older"):
        if name not in groups:
            continue
        section_title(name)
        for conv in groups[name]:
            questions = sum(1 for m in conv["messages"] if m["role"] == "user")
            answer = next((m for m in conv["messages"] if m["role"] == "assistant" and not m.get("error")), None)
            preview = re.sub(r"\[S\d+\]|[#*|>`]", "", answer["content"])[:140] if answer else ""
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([6, 1.2, 1.2, 1.2], vertical_alignment="center")
                when = (conv.get("updated_at") or conv["created_at"])[:16].replace("T", " ")
                c1.markdown(f"<div class='conv-title'>{escape(conv['title'])}</div>"
                            f"<div class='conv-meta'>{when} · {questions} question{'s' if questions != 1 else ''}"
                            + (f" · {escape(preview)}…" if preview else "") + "</div>", unsafe_allow_html=True)
                c2.button("Open", key=f"open-{conv['id']}", use_container_width=True,
                          on_click=open_conversation, args=(conv["id"],))
                c3.download_button("Export", transcript_markdown(conv), file_name=f"conversation-{conv['id']}.md",
                                   mime="text/markdown", key=f"dl-{conv['id']}", use_container_width=True)
                c4.button("Delete", key=f"del-{conv['id']}", use_container_width=True,
                          on_click=delete_conversation, args=(conv["id"],))
    st.write("")
    with st.expander("Delete all conversations"):
        st.caption("This permanently removes every conversation" + (" saved on this server." if HISTORY_MODE == "file" else " in this session."))
        if st.button("Delete all", type="primary"):
            st.session_state.conversations = []
            new_conversation()
            save_history()
            st.toast("All conversations deleted", icon="🗑")
            st.rerun()


# --------------------------------------------------------------------------- knowledge base page
def citation_usage() -> List[Dict]:
    counts: Dict[str, int] = {}
    for conv in st.session_state.conversations:
        for m in conv["messages"]:
            for doc in {s["document"] for s in m.get("sources") or []}:
                counts[doc] = counts.get(doc, 0) + 1
    return [{"document": d, "answers": n} for d, n in sorted(counts.items(), key=lambda x: -x[1])]


def page_knowledge_base() -> None:
    read_only = kb_read_only()
    page_header("Knowledge Base", "What the assistant knows: documents, coverage and processing status.",
                eyebrow="Analytics")
    stats = cached_stats()
    if stats is None:
        st.error("Couldn't load knowledge base statistics: the backend isn't responding.", icon="⚠️")
        if st.button("↻ Retry"):
            refresh_data()
            st.rerun()
        return
    docs = cached_documents()
    if not docs:
        hint = ("Add documents to the <code>knowledge_base/</code> folder and restart the app." if read_only
                else "Use <b>Upload Documents</b> to add some.")
        empty_state("📭", "No documents yet", hint)
        return

    size_kb = stats.get("total_size_bytes", 0) / 1024
    kpi_tiles([
        ("Documents", stats["indexed_documents"], f"{stats['total_documents']} total"),
        ("Indexed passages", stats["total_chunks"], f"{stats.get('avg_chunks_per_document', 0)} per document"),
        ("Categories", len(stats.get("by_category", {})), ", ".join(sorted(stats.get("by_file_type", {}))).upper() or "—"),
        ("Content size", f"{size_kb:,.0f} KB" if size_kb < 1024 else f"{size_kb / 1024:,.1f} MB", "source files"),
        ("Needs attention", stats["failed_documents"] + stats.get("processing_documents", 0),
         f"{stats['failed_documents']} failed · {stats.get('processing_documents', 0)} processing"),
    ])

    dark = st.session_state.dark_mode
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Passages per document**")
        rows = [{"document": d["filename"].rsplit(".", 1)[0], "passages": d["chunks"], "category": d["category"]}
                for d in stats.get("chunks_by_document", [])]
        if rows:
            bar_chart(rows, "document", "passages", "Passages", dark, ["category"])
    with c2.container(border=True):
        st.markdown("**Documents by category**")
        rows = [{"category": k, "documents": v} for k, v in stats.get("by_category", {}).items()]
        if rows:
            bar_chart(rows, "category", "documents", "Documents", dark)
        usage = citation_usage()
        st.markdown("**Most cited in your conversations**")
        if usage:
            bar_chart([dict(u, document=u["document"].rsplit(".", 1)[0]) for u in usage[:8]],
                      "document", "answers", "Answers citing it", dark)
        else:
            st.caption("No cited answers yet. Ask a few questions and this fills in.")

    section_title("Documents")
    f1, f2 = st.columns([3, 2])
    search = f1.text_input("Search documents", placeholder="Search by name…", label_visibility="collapsed")
    cats = sorted({d.get("category", "General") for d in docs})
    cat = f2.selectbox("Category", ["All categories"] + cats, label_visibility="collapsed")
    shown = [d for d in docs if (not search or search.lower() in d["filename"].lower())
             and (cat == "All categories" or d.get("category") == cat)]
    if not shown:
        empty_state("🔍", "No documents match", "Clear the search or pick another category.")
    else:
        html_table([
            {
                "Document": escape(d["filename"]),
                "Category": escape(d.get("category", "")),
                "Status": status_pill(d.get("status", "")),
                "Passages": d.get("chunks", 0),
                "Pages": d.get("pages") or "—",
                "Size": f"{d.get('size_bytes', 0) / 1024:.1f} KB",
                "Indexed": (d.get("indexed_at") or d.get("updated_at") or "")[:16].replace("T", " "),
                "Notes": escape(d.get("error") or ""),
            }
            for d in shown
        ], raw_html_cols=["Document", "Category", "Status", "Notes"])
    if any(d.get("status") in {"processing", "reindexing"} for d in docs):
        st.info("Some documents are still processing. This page refreshes their status when you reopen it.", icon="⏳")

    if not read_only:
        a1, a2, _ = st.columns([1.6, 1.6, 3])
        if a1.button("🔄 Re-index all", use_container_width=True):
            with st.spinner("Re-indexing every document…"):
                try:
                    results = api().reindex()["results"]
                    st.toast(f"Re-indexed {sum(r['status'] == 'reindexed' for r in results)} documents", icon="✅")
                except APIError as exc:
                    st.error(str(exc))
            refresh_data()
        if a2.button("📂 Sync folder", use_container_width=True, help="Index files copied directly into knowledge_base/"):
            with st.spinner("Scanning knowledge_base folder…"):
                try:
                    results = api().sync()["results"]
                    st.toast(f"Sync complete: {len(results)} change(s)", icon="✅")
                except APIError as exc:
                    st.error(str(exc))
            refresh_data()

    section_title("Inspect a document")
    names = {d["filename"]: d for d in docs}
    selected = st.selectbox("Document", list(names), label_visibility="collapsed")
    doc = names[selected]
    b1, b2, b3 = st.columns(3)
    if b1.button("View indexed passages", use_container_width=True):
        st.session_state.inspect = doc["doc_id"]
    if not read_only:
        if b2.button("Re-index", use_container_width=True):
            with st.spinner(f"Re-indexing {selected}…"):
                try:
                    res = api().reindex(doc["doc_id"])
                    (st.success if res["status"] == "reindexed" else st.error)(res.get("message", res["status"]))
                except APIError as exc:
                    st.error(str(exc))
            refresh_data()
        confirm = b3.checkbox("Confirm delete", key=f"confirm-{doc['doc_id']}")
        if b3.button("Delete document", type="primary", disabled=not confirm, use_container_width=True):
            try:
                api().delete(doc["doc_id"])
                st.toast(f"Deleted {selected}", icon="🗑")
                refresh_data()
                st.rerun()
            except APIError as exc:
                st.error(str(exc))

    if st.session_state.get("inspect") == doc["doc_id"]:
        try:
            chunks = api().chunks(doc["doc_id"])["chunks"]
        except APIError as exc:
            st.error(f"Couldn't load passages: {exc}")
            return
        st.caption(f"{len(chunks)} passages, in document order")
        for ch in chunks:
            md = ch["metadata"]
            title = f"Passage {md.get('chunk_index', 0) + 1}" + (f" · page {md['page']}" if md.get("page") else "")
            title += f" · {md.get('section') or 'Before first heading'}"
            with st.expander(title):
                st.markdown(f"<div class='excerpt'>{escape(ch['text'])}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- upload page
def page_upload() -> None:
    page_header("Upload Documents", "Add PDF, DOCX, TXT or Markdown files. Each is extracted, split into passages, "
                "embedded and indexed.", eyebrow="Knowledge base")
    st.warning(DATA_POLICY_WARNING, icon="🔐")
    settings = cached_settings() or {}
    categories = settings.get("categories", ["General"])
    with st.form("upload", clear_on_submit=True, border=True):
        files = st.file_uploader("Drop files here or browse", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
                                 help=f"Max {settings.get('max_upload_mb', 25)} MB per file")
        category = st.selectbox("Category", categories, index=categories.index("General") if "General" in categories else 0)
        submitted = st.form_submit_button("Upload and index", type="primary", use_container_width=True)

    if submitted and not files:
        st.error("Choose at least one file to upload.", icon="⚠️")
    elif submitted:
        results = []
        progress = st.progress(0.0, text=f"Processing 0 of {len(files)} files…")
        for i, f in enumerate(files, start=1):
            with st.status(f"{f.name}: uploading and indexing…", expanded=False) as status:
                st.write("Validating, extracting text, splitting into passages and embedding…")
                try:
                    res = api().upload([(f.name, f.getvalue(), f.type)], category)["results"][0]
                except APIError as exc:
                    res = {"status": "rejected", "filename": f.name, "message": str(exc)}
                results.append(res)
                ok = res["status"] in {"indexed", "updated", "duplicate"}
                st.write(res.get("message", ""))
                status.update(label=f"{f.name}: {res['status'].replace('_', ' ')}", state="complete" if ok else "error")
            progress.progress(i / len(files), text=f"Processed {i} of {len(files)} files")
        refresh_data()
        indexed = sum(r["status"] in {"indexed", "updated"} for r in results)
        failed = sum(r["status"] in {"failed", "rejected"} for r in results)
        dupes = sum(r["status"] == "duplicate" for r in results)
        kpi_tiles([("Indexed", indexed, "new or updated"), ("Skipped", dupes, "identical content already indexed"),
                   ("Failed", failed, "see details below")])
        if failed:
            st.error(f"{failed} file{'s' if failed != 1 else ''} could not be indexed.", icon="⚠️")
        if indexed:
            st.success(f"{indexed} file{'s' if indexed != 1 else ''} ready to answer questions.", icon="✅")
        html_table([{"File": escape(r["filename"]), "Result": status_pill(r["status"]),
                     "Passages": r.get("chunks", "—"), "Details": escape(r.get("message", ""))} for r in results],
                   raw_html_cols=["File", "Result", "Details"])

    section_title("Recent activity")
    docs = sorted(cached_documents(), key=lambda d: d.get("updated_at") or "", reverse=True)[:8]
    if not docs:
        empty_state("📄", "Nothing uploaded yet", "Files you upload appear here with their processing status.")
    else:
        html_table([{"Document": escape(d["filename"]), "Status": status_pill(d.get("status", "")),
                     "Passages": d.get("chunks", 0), "Updated": (d.get("updated_at") or "")[:16].replace("T", " ")}
                    for d in docs], raw_html_cols=["Document", "Status"])


# --------------------------------------------------------------------------- settings page
def page_settings() -> None:
    page_header("Settings", "Backend configuration and per-session retrieval controls.", eyebrow="System")
    settings = cached_settings()
    if settings is None:
        st.error("The embedded backend failed to start - see the app logs." if EMBEDDED
                 else f"Cannot reach the backend at {BACKEND_URL}.", icon="⚠️")
        return
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Language model**")
        provider = settings.get("provider", "anthropic")
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
        st.write(f"Provider: `{provider}`")
        st.write(f"Model: `{settings['model']}`")
        if provider == "anthropic":
            st.write(f"Effort: `{settings['effort']}`")
            st.write(f"Refusal fallback: `{'on' if settings['fallbacks_enabled'] else 'off'}`")
        st.write("API key: " + ("✅ configured on server" if settings["llm_configured"] else f"❌ missing (set {key_name})"))
        if settings.get("api_key_warning"):
            st.warning(settings["api_key_warning"])
    with c2.container(border=True):
        st.markdown("**Indexing & retrieval**")
        st.write(f"Embeddings: `{settings['embedding_model']}`")
        st.write(f"Chunk size / overlap: `{settings['chunk_size']}` / `{settings['chunk_overlap']}` tokens (split at headings first)")
        search = "hybrid (semantic + keyword)" if settings.get("hybrid_search", True) else "semantic only"
        rerank = f"reranked by `{settings['reranker_model']}`" if settings.get("reranker_model") else "no reranking"
        st.write(f"Search: {search}, {settings.get('candidate_pool', 20)} candidates, {rerank}")
        if settings.get("kb_read_only", True):
            st.write("Knowledge base: fixed (read-only), loaded from `knowledge_base/`")
        else:
            st.write(f"Supported types: {', '.join(settings['supported_extensions'])} · max {settings['max_upload_mb']} MB")
        st.caption("Change these in `.env` and restart; documents are re-indexed automatically when chunk settings change.")

    section_title("Retrieval (this session)")
    top_k = st.slider("Passages sent to the model (top-k)", 1, 20, st.session_state.top_k or settings["top_k"])
    threshold = st.slider("Minimum relevance score", 0.0, 0.9, float(st.session_state.threshold or settings["relevance_threshold"]), 0.05,
                          help="If no passage reaches this similarity, the assistant answers with the fallback message "
                               "without calling the AI model. Higher = more questions rejected up front.")
    st.session_state.top_k = top_k if top_k != settings["top_k"] else None
    st.session_state.threshold = threshold if abs(threshold - settings["relevance_threshold"]) > 1e-9 else None

    section_title("Appearance & data")
    st.caption("Switch between light and dark theme with the **Dark mode** toggle in the sidebar. "
               + ("Conversations are kept for this browser session only." if HISTORY_MODE != "file"
                  else "Conversations are saved on this server in `data/chat_history.json`."))


PAGES = {"assistant": page_assistant, "history": page_history, "kb": page_knowledge_base,
         "upload": page_upload, "settings": page_settings}


def main() -> None:
    st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚", layout="wide",
                       initial_sidebar_state="auto")
    init_state()
    inject_css(st.session_state.dark_mode)
    sidebar()
    page = st.session_state.page
    if page == "upload" and kb_read_only():
        page = "assistant"
    PAGES[page]()
