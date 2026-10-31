"""Streamlit dashboard for the Company Knowledge Assistant.

The UI only talks to the FastAPI backend over HTTP; it never loads the Anthropic key.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from frontend.api_client import APIError, BackendClient

# Unset or "embedded": run the API inside this Streamlit process (needed on Streamlit Community Cloud,
# which runs only `streamlit run app.py`). Otherwise talk to a separately started API server over HTTP.
BACKEND_URL = os.getenv("BACKEND_URL", "").strip() or "embedded"
EMBEDDED = BACKEND_URL.lower() == "embedded"
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_history.json"

NAV = [
    ("assistant", "💬  AI Knowledge Assistant"),
    ("history", "🕘  Chat History"),
    ("kb", "📚  Knowledge Base"),
    ("upload", "⬆️  Upload Documents"),
    ("settings", "⚙️  System Settings"),
]

DATA_POLICY_WARNING = (
    "**Data handling notice:** uploaded files are stored and indexed on this server, and relevant excerpts are "
    "sent to the Anthropic API to generate answers. Only upload documents you are authorised to share, and "
    "handle them according to your company's data classification and retention policies. Do not upload "
    "personal data, credentials, or regulated information unless your policies permit it."
)

THEMES = {
    "light": {
        "bg": "#F6F7FB", "surface": "#FFFFFF", "surface2": "#F1F3F9", "text": "#141824", "muted": "#5D6475",
        "border": "#E2E6EF", "accent": "#4F46E5", "accent_soft": "#EEF0FF", "ok": "#0F9D58", "warn": "#B26A00",
    },
    "dark": {
        "bg": "#0E1117", "surface": "#161A23", "surface2": "#1D2230", "text": "#E7E9EF", "muted": "#9AA2B4",
        "border": "#2A3040", "accent": "#8B85FF", "accent_soft": "#23234A", "ok": "#3DDC97", "warn": "#F5B54A",
    },
}


# --------------------------------------------------------------------------- state & persistence
@st.cache_resource(show_spinner="Loading the knowledge base and embedding model…")
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


def load_history() -> List[Dict]:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_history() -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    convs = [c for c in st.session_state.conversations if c["messages"]]
    HISTORY_FILE.write_text(json.dumps(convs[:100], indent=1), encoding="utf-8")


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
    conv = {"id": uuid.uuid4().hex[:12], "title": "New conversation",
            "created_at": datetime.now().isoformat(timespec="minutes"), "messages": []}
    st.session_state.conversations.insert(0, conv)
    st.session_state.current_id = conv["id"]


# Navigation changes run as widget callbacks: the "page" radio's state may only be set before it renders.
def start_new_chat() -> None:
    new_conversation()
    st.session_state.page = "assistant"


def open_conversation(conv_id: str) -> None:
    st.session_state.current_id = conv_id
    st.session_state.page = "assistant"


def current_conversation() -> Dict:
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_id:
            return conv
    new_conversation()
    return st.session_state.conversations[0]


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


def refresh_data() -> None:
    cached_documents.clear()
    cached_settings.clear()


# --------------------------------------------------------------------------- styling
def inject_css(theme: str) -> None:
    t = THEMES[theme]
    st.markdown(
        f"""
<style>
:root {{
  --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --text:{t['text']}; --muted:{t['muted']};
  --border:{t['border']}; --accent:{t['accent']}; --accent-soft:{t['accent_soft']}; --ok:{t['ok']}; --warn:{t['warn']};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: var(--bg) !important; color: var(--text); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: var(--surface) !important; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] * {{ color: var(--text); }}
.block-container {{ padding-top: 2rem; max-width: 1100px; }}
h1, h2, h3, h4, p, li, label, span, div {{ color: inherit; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color: var(--text); }}
[data-testid="stCaptionContainer"], .muted {{ color: var(--muted) !important; }}
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem 1rem; }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--border) !important; border-radius: 10px; }}
[data-testid="stExpander"] summary {{ color: var(--text) !important; }}
[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem 1rem; }}
[data-testid="stMetricValue"] {{ color: var(--text); }}
[data-testid="stMetricLabel"] p {{ color: var(--muted) !important; }}
textarea, input, [data-baseweb="select"] > div, [data-baseweb="input"] {{ background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }}
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div {{ background: var(--bg) !important; }}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{ background: var(--surface) !important; border-color: var(--border) !important; border-radius: 12px; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface) !important; border: 1.5px dashed var(--border); }}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 10px; border: 1px solid var(--border); background: var(--surface); color: var(--text);
}}
.stButton button:hover, .stDownloadButton button:hover {{ background: var(--surface2) !important; border-color: var(--accent) !important; color: var(--accent) !important; }}
.stButton button:disabled {{ opacity: .45; background: var(--surface) !important; color: var(--muted) !important; }}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind^="primary"] {{ background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }}
.doc-table {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); margin-bottom: 1rem; }}
.doc-table table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
.doc-table th {{ text-align: left; font-weight: 600; color: var(--muted); background: var(--surface2); padding: .55rem .75rem; border-bottom: 1px solid var(--border); white-space: nowrap; }}
.doc-table td {{ padding: .5rem .75rem; border-bottom: 1px solid var(--border); color: var(--text); white-space: nowrap; }}
.doc-table tr:last-child td {{ border-bottom: none; }}
.brand {{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem; }}
.brand-logo {{ width:34px; height:34px; border-radius:10px; background:var(--accent); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; }}
.brand-title {{ font-weight:700; font-size:1.02rem; line-height:1.1; }}
.brand-sub {{ font-size:.78rem; color:var(--muted); }}
.page-title {{ font-size:1.6rem; font-weight:700; margin-bottom:.1rem; }}
.page-sub {{ color:var(--muted); margin-bottom:1.2rem; }}
.pill {{ display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600; background:var(--accent-soft); color:var(--accent); margin-right:.3rem; }}
.pill.ok {{ color:var(--ok); background:transparent; border:1px solid var(--ok); }}
.pill.warn {{ color:var(--warn); background:transparent; border:1px solid var(--warn); }}
.excerpt {{ background:var(--surface2); border-left:3px solid var(--accent); border-radius:6px; padding:.6rem .8rem; font-size:.88rem; white-space:pre-wrap; color:var(--text); }}
.src-meta {{ font-size:.82rem; color:var(--muted); margin-bottom:.4rem; }}
.excerpt mark {{ background:color-mix(in srgb, var(--accent) 22%, transparent); color:var(--text); padding:0 .1rem; border-radius:3px; font-weight:600; }}
.muted-excerpt {{ border-left-color:var(--border); color:var(--muted); margin-bottom:.5rem; }}
.src-table {{ border-collapse:collapse; font-size:.84rem; margin:.1rem 0 .8rem; width:100%; }}
.src-table th {{ text-align:left; color:var(--muted); font-weight:600; padding:.2rem .8rem .2rem 0; white-space:nowrap; vertical-align:top; width:7rem; }}
.src-table td {{ padding:.2rem 0; color:var(--text); word-break:break-word; }}
sup.cite {{ color:var(--accent); font-weight:700; cursor:help; margin-left:1px; }}
.empty {{ text-align:center; padding:2.5rem 1rem; color:var(--muted); border:1px dashed var(--border); border-radius:14px; background:var(--surface); }}
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{escape(title)}</div><div class='page-sub'>{escape(subtitle)}</div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='brand'><div class='brand-logo'>K</div><div><div class='brand-title'>AI Knowledge Assistant</div>"
            "<div class='brand-sub'>Company knowledge, grounded answers</div></div></div>",
            unsafe_allow_html=True,
        )
        labels = dict(NAV)
        st.radio("Navigation", list(labels), format_func=labels.get, key="page", label_visibility="collapsed")

        st.divider()
        st.button("＋  New chat", use_container_width=True, on_click=start_new_chat)

        recent = [c for c in st.session_state.conversations if c["messages"]][:5]
        if recent:
            st.caption("Recent")
            for conv in recent:
                st.button(conv["title"][:38], key=f"recent-{conv['id']}", use_container_width=True,
                          on_click=open_conversation, args=(conv["id"],))

        st.divider()
        try:
            health = api().health()
            status = "<span class='pill ok'>● Backend online</span>"
            if not health.get("llm_configured"):
                status += "<br><span class='pill warn'>API key missing</span>"
            elif health.get("api_key_warning"):
                status += "<br><span class='pill warn'>API key looks invalid</span>"
            st.markdown(status, unsafe_allow_html=True)
            st.caption(f"{health['documents']} documents · {health['chunks']} chunks")
        except APIError:
            st.markdown("<span class='pill warn'>● Backend offline</span>", unsafe_allow_html=True)
            st.caption("Embedded backend failed to start - see the app logs." if EMBEDDED else f"Expected at {BACKEND_URL}")
        st.toggle("Dark mode", key="dark_mode")


# --------------------------------------------------------------------------- chat rendering
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
    """Show the answer with [S#] markers turned into numbered citations that name their source on hover."""
    by_id = {s["source_id"]: s for s in numbered_sources(msg)}
    # Escape any HTML in the model's text first; only our own <sup> tags are allowed through.
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
    rows.append(("Relevance", f"{src['score']:.2f}"))
    if src.get("uploaded_at"):
        rows.append(("Uploaded", src["uploaded_at"][:10]))
    table = "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>" for k, v in rows)
    st.markdown(f"<table class='src-table'>{table}</table>", unsafe_allow_html=True)

    label = "Supporting sentences highlighted" if src.get("highlights") else "Retrieved passage"
    st.markdown(f"<div class='src-meta'>{label}</div>{highlighted_excerpt(src)}", unsafe_allow_html=True)

    if src.get("doc_id") and st.toggle("Show surrounding text", key=f"ctx-{key}"):
        try:
            for chunk in neighbour_chunks(src["doc_id"], src.get("chunk_index")):
                meta = chunk.get("metadata", {})
                where = " · ".join(str(x) for x in (meta.get("section"), f"p. {meta['page']}" if meta.get("page") and src.get("page") else None) if x)
                st.markdown(f"<div class='src-meta'>{escape(where or 'Adjacent passage')}</div>"
                            f"<div class='excerpt muted-excerpt'>{escape(chunk.get('text', ''))}</div>", unsafe_allow_html=True)
        except APIError as exc:
            st.caption(f"Could not load surrounding text: {exc}")


def render_sources(msg: Dict, key: str) -> None:
    sources = numbered_sources(msg)
    by_id = {s["source_id"]: s for s in sources}
    if msg.get("conflicts"):
        lines = "\n".join(
            f"- {c['description']} _(see "
            + ", ".join(f"[{by_id[s]['number']}] {by_id[s]['document']}" if s in by_id else s for s in c["sources"])
            + ")_"
            for c in msg["conflicts"]
        )
        st.warning(f"**Conflicting information detected**\n\n{lines}")
    if sources:
        st.markdown(f"**Sources ({len(sources)})**")
        for src in sources:
            with st.expander(f"[{src['number']}] {source_label(src)}"):
                render_source_details(src, f"{key}-{src['source_id']}")
    retrieved = msg.get("retrieved") or []
    if retrieved:
        cited_ids = {s["chunk_id"] for s in sources}
        with st.expander(f"Inspect all retrieved passages ({len(retrieved)})"):
            st.caption("Everything the search returned for this question, including passages the answer did not use.")
            for r in retrieved:
                mark = " · ✅ cited" if r["chunk_id"] in cited_ids else ""
                st.markdown(f"**{escape(source_label(r))}** · relevance {r['score']:.2f}{mark}")
                st.markdown(f"<div class='excerpt'>{escape(r['excerpt'])}</div>", unsafe_allow_html=True)
    if msg.get("meta"):
        st.caption(msg["meta"])


def render_message(msg: Dict, idx: int) -> None:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✨"):
        if msg.get("error"):
            st.error(msg["content"])
            return
        if msg["role"] == "assistant" and msg.get("found") is False:
            st.info(msg["content"])
        elif msg["role"] == "assistant":
            render_answer_text(msg)
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg, f"m{idx}")


def transcript_text(conv: Dict) -> str:
    """Plain-text export that keeps each answer's verified sources."""
    blocks = []
    for m in conv["messages"]:
        text = f"{m['role'].upper()}: {m['content']}"
        if m["role"] == "assistant" and m.get("sources"):
            refs = "\n".join(f"  [{s['number']}] {source_label(s)} ({s['source_id']})" for s in numbered_sources(m))
            text += f"\nSOURCES:\n{refs}"
        blocks.append(text)
    return "\n\n".join(blocks)


def page_assistant() -> None:
    conv = current_conversation()
    top = st.columns([5, 1.6])
    with top[0]:
        page_header("AI Knowledge Assistant", "Ask about policies, SOPs, guidelines and training material.")
    with top[1]:
        st.write("")
        if st.button("🗑  Clear chat", use_container_width=True, disabled=not conv["messages"]):
            conv["messages"] = []
            conv["title"] = "New conversation"
            save_history()
            st.rerun()

    docs = [d for d in cached_documents() if d.get("status") == "indexed"]
    with st.expander("🔎 Search scope (metadata filters)"):
        cats = sorted({d.get("category", "General") for d in docs})
        c1, c2 = st.columns(2)
        sel_cats = c1.multiselect("Categories", cats, key="filter_cats")
        doc_opts = {d["filename"]: d["doc_id"] for d in docs if not sel_cats or d.get("category") in sel_cats}
        sel_docs = c2.multiselect("Documents", list(doc_opts), key="filter_docs")

    if not conv["messages"]:
        if not docs:
            st.markdown("<div class='empty'>📭 The knowledge base is empty.<br>Go to <b>Upload Documents</b> to add "
                        "policies, SOPs or handbooks.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='empty'>Ask anything about the {len(docs)} indexed company documents.</div>",
                        unsafe_allow_html=True)
            st.write("")
            examples = [
                "Summarise the employee handbook in plain language",
                "How should a customer complaint be handled? Give me the steps.",
                "Compare what the documents say about customer communication",
                "What are the steps to onboard a new vendor?",
            ]
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                    st.session_state.pending_question = ex
                    st.rerun()

    for i, msg in enumerate(conv["messages"]):
        render_message(msg, i)

    question = st.chat_input("Ask a question about company documents…")
    question = question or st.session_state.pop("pending_question", None)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"] if not m.get("error")]
    user_msg = {"role": "user", "content": question}
    conv["messages"].append(user_msg)
    if conv["title"] == "New conversation":
        conv["title"] = question[:60]
    render_message(user_msg, len(conv["messages"]))

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Searching the knowledge base and drafting a grounded answer…"):
            try:
                result = api().chat(
                    question, history,
                    categories=sel_cats or None,
                    doc_ids=[doc_opts[n] for n in sel_docs] or None,
                    top_k=st.session_state.top_k,
                    relevance_threshold=st.session_state.threshold,
                )
                meta = f"{result.get('model') or 'retrieval only'} · {result.get('latency_ms', 0) / 1000:.1f}s"
                msg = {"role": "assistant", "content": result["answer"], "found": result["found"],
                       "sources": result["sources"], "retrieved": result["retrieved"],
                       "conflicts": result["conflicts"], "meta": meta}
            except APIError as exc:
                msg = {"role": "assistant", "content": f"⚠️ {exc}", "error": True}
    conv["messages"].append(msg)
    save_history()
    st.rerun()


# --------------------------------------------------------------------------- other pages
def page_history() -> None:
    page_header("Chat History", "Reopen, export or delete previous conversations.")
    convs = [c for c in st.session_state.conversations if c["messages"]]
    if not convs:
        st.markdown("<div class='empty'>No conversations yet.</div>", unsafe_allow_html=True)
        return
    query = st.text_input("Search conversations", placeholder="Filter by title…")
    for conv in convs:
        if query and query.lower() not in conv["title"].lower():
            continue
        n = sum(1 for m in conv["messages"] if m["role"] == "user")
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([5, 1.1, 1.1, 1.1])
            c1.markdown(f"**{escape(conv['title'])}**  \n<span class='muted'>{conv['created_at'].replace('T', ' ')} · "
                        f"{n} question{'s' if n != 1 else ''}</span>", unsafe_allow_html=True)
            c2.button("Open", key=f"open-{conv['id']}", use_container_width=True,
                      on_click=open_conversation, args=(conv["id"],))
            transcript = transcript_text(conv)
            c3.download_button("Export", transcript, file_name=f"chat-{conv['id']}.txt", key=f"dl-{conv['id']}",
                               use_container_width=True)
            if c4.button("Delete", key=f"del-{conv['id']}", use_container_width=True):
                st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != conv["id"]]
                if st.session_state.current_id == conv["id"]:
                    new_conversation()
                save_history()
                st.rerun()


STATUS_BADGE = {"indexed": "✅ Indexed", "failed": "❌ Failed", "processing": "⏳ Processing", "reindexing": "🔄 Re-indexing"}


def page_knowledge_base() -> None:
    page_header("Knowledge Base", "Manage indexed documents, inspect chunks and re-index.")
    try:
        stats = api().stats()
    except APIError as exc:
        st.error(str(exc))
        return
    m = st.columns(4)
    m[0].metric("Documents", stats["total_documents"])
    m[1].metric("Indexed", stats["indexed_documents"])
    m[2].metric("Indexed chunks", stats["total_chunks"])
    m[3].metric("Failed", stats["failed_documents"])

    docs = cached_documents()
    if not docs:
        st.markdown("<div class='empty'>No documents yet. Use <b>Upload Documents</b> to add some.</div>",
                    unsafe_allow_html=True)
        return

    st.write("")
    rows = [
        {
            "Document": d["filename"],
            "Category": d.get("category", ""),
            "Status": STATUS_BADGE.get(d.get("status"), d.get("status")),
            "Chunks": d.get("chunks", 0),
            "Pages": d.get("pages") or "—",
            "Version": d.get("version", 1),
            "Size (KB)": round(d.get("size_bytes", 0) / 1024, 1),
            "Uploaded": (d.get("uploaded_at") or "")[:16].replace("T", " "),
            "Updated": (d.get("updated_at") or "")[:16].replace("T", " "),
            "Notes": d.get("error") or "",
        }
        for d in docs
    ]
    # Plain HTML table so it follows the light/dark theme (st.dataframe draws on a canvas).
    head = "".join(f"<th>{escape(k)}</th>" for k in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in r.values()) + "</tr>" for r in rows)
    st.markdown(f"<div class='doc-table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
                unsafe_allow_html=True)

    a1, a2, _ = st.columns([1.8, 1.8, 3])
    if a1.button("🔄 Re-index all", use_container_width=True):
        with st.spinner("Re-indexing every document…"):
            try:
                results = api().reindex()["results"]
                st.success(f"Re-indexed {sum(r['status'] == 'reindexed' for r in results)} documents.")
            except APIError as exc:
                st.error(str(exc))
        refresh_data()
    if a2.button("📂 Sync folder", use_container_width=True, help="Index files copied directly into knowledge_base/"):
        with st.spinner("Scanning knowledge_base folder…"):
            try:
                results = api().sync()["results"]
                st.success(f"Sync complete: {len(results)} change(s).")
            except APIError as exc:
                st.error(str(exc))
        refresh_data()

    st.subheader("Document details")
    names = {d["filename"]: d for d in docs}
    selected = st.selectbox("Select a document", list(names))
    doc = names[selected]
    c1, c2, c3 = st.columns(3)
    if c1.button("Inspect chunks", use_container_width=True):
        st.session_state.inspect = doc["doc_id"]
    if c2.button("Re-index", use_container_width=True):
        with st.spinner(f"Re-indexing {selected}…"):
            try:
                res = api().reindex(doc["doc_id"])
                (st.success if res["status"] == "reindexed" else st.error)(res.get("message", res["status"]))
            except APIError as exc:
                st.error(str(exc))
        refresh_data()
    confirm = c3.checkbox("Confirm delete", key=f"confirm-{doc['doc_id']}")
    if c3.button("Delete document", type="primary", disabled=not confirm, use_container_width=True):
        try:
            api().delete(doc["doc_id"])
            st.success(f"Deleted {selected} and removed its chunks from the index.")
            refresh_data()
            st.rerun()
        except APIError as exc:
            st.error(str(exc))

    if st.session_state.get("inspect") == doc["doc_id"]:
        try:
            chunks = api().chunks(doc["doc_id"])["chunks"]
        except APIError as exc:
            st.error(str(exc))
            return
        st.caption(f"{len(chunks)} chunks")
        for ch in chunks:
            md = ch["metadata"]
            title = f"Chunk {md.get('chunk_index', 0) + 1}" + (f" · page {md['page']}" if md.get("page") else "")
            title += f" · {md.get('section', '')}"
            with st.expander(title):
                st.markdown(f"<div class='excerpt'>{escape(ch['text'])}</div>", unsafe_allow_html=True)


def page_upload() -> None:
    page_header("Upload Documents", "Add PDF, DOCX or TXT files. Text is extracted, chunked, embedded and indexed.")
    st.warning(DATA_POLICY_WARNING)
    settings = cached_settings() or {}
    categories = settings.get("categories", ["General"])
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Choose documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
                                 help=f"Max {settings.get('max_upload_mb', 25)} MB per file")
        category = st.selectbox("Category", categories, index=categories.index("General") if "General" in categories else 0)
        submitted = st.form_submit_button("Upload & index", type="primary")
    if not submitted:
        return
    if not files:
        st.error("Select at least one file.")
        return
    progress = st.progress(0.0, text="Uploading…")
    results = []
    for i, f in enumerate(files, start=1):
        progress.progress((i - 1) / len(files), text=f"Processing {f.name} ({i}/{len(files)})…")
        try:
            res = api().upload([(f.name, f.getvalue(), f.type)], category)
            results.extend(res["results"])
        except APIError as exc:
            results.append({"status": "rejected", "filename": f.name, "message": str(exc)})
    progress.progress(1.0, text="Done")
    refresh_data()
    icons = {"indexed": st.success, "updated": st.success, "duplicate": st.info, "failed": st.error, "rejected": st.error}
    for r in results:
        icons.get(r["status"], st.write)(f"**{r['filename']}** — {r['status'].upper()}: {r.get('message', '')}")


def page_settings() -> None:
    page_header("System Settings", "Backend configuration and per-session retrieval controls.")
    settings = cached_settings()
    if settings is None:
        st.error("The embedded backend failed to start - see the app logs." if EMBEDDED
                 else f"Cannot reach the backend at {BACKEND_URL}.")
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
        st.markdown("**Indexing**")
        st.write(f"Embeddings: `{settings['embedding_model']}`")
        st.write(f"Chunk size / overlap: `{settings['chunk_size']}` / `{settings['chunk_overlap']}` characters")
        st.write(f"Supported types: {', '.join(settings['supported_extensions'])} · max {settings['max_upload_mb']} MB")
        st.caption("Change these in `.env`, restart the API, then use Re-index all.")

    st.subheader("Retrieval (this session)")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 20, st.session_state.top_k or settings["top_k"])
    threshold = st.slider("Minimum relevance score", 0.0, 0.9, float(st.session_state.threshold or settings["relevance_threshold"]), 0.05,
                          help="Chunks scoring below this are ignored. Higher = stricter grounding, more fallbacks.")
    st.session_state.top_k = top_k if top_k != settings["top_k"] else None
    st.session_state.threshold = threshold if abs(threshold - settings["relevance_threshold"]) > 1e-9 else None

    st.subheader("Appearance & data")
    st.caption("Switch between light and dark theme with the **Dark mode** toggle in the sidebar.")
    if st.button("Delete all chat history"):
        st.session_state.conversations = []
        new_conversation()
        save_history()
        st.success("Chat history cleared.")


PAGES = {"assistant": page_assistant, "history": page_history, "kb": page_knowledge_base,
         "upload": page_upload, "settings": page_settings}


def main() -> None:
    st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚", layout="wide")
    init_state()
    inject_css("dark" if st.session_state.dark_mode else "light")
    sidebar()
    PAGES[st.session_state.page]()
