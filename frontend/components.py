"""Presentation pieces for the dashboard: theme + responsive CSS, small HTML components and charts.

Nothing here talks to the backend; pages pass in data they fetched through the API client.
"""
from __future__ import annotations

from html import escape
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

THEMES = {
    "light": {
        "bg": "#F6F7FB", "surface": "#FFFFFF", "surface2": "#F1F3F9", "text": "#141824", "muted": "#5D6475",
        "border": "#E2E6EF", "accent": "#4F46E5", "accent_soft": "#EEF0FF", "ok": "#0F7B45", "warn": "#9A5B00",
        "err": "#B42318", "chart": "#4F46E5",
    },
    "dark": {
        "bg": "#0E1117", "surface": "#161A23", "surface2": "#1D2230", "text": "#E7E9EF", "muted": "#9AA2B4",
        "border": "#2A3040", "accent": "#8B85FF", "accent_soft": "#23234A", "ok": "#3DDC97", "warn": "#F5B54A",
        "err": "#FF7A6E", "chart": "#827CFA",  # chart marks validated for the dark surface
    },
}

STATUS = {
    "indexed": ("ok", "✓", "Indexed"),
    "updated": ("ok", "✓", "Updated"),
    "reindexed": ("ok", "✓", "Re-indexed"),
    "processing": ("info", "◌", "Processing"),
    "reindexing": ("info", "◌", "Re-indexing"),
    "duplicate": ("info", "=", "Duplicate (skipped)"),
    "failed": ("err", "✕", "Failed"),
    "rejected": ("err", "✕", "Rejected"),
}


def theme(dark: bool) -> Dict[str, str]:
    return THEMES["dark" if dark else "light"]


def inject_css(dark: bool) -> None:
    t = theme(dark)
    st.markdown(
        f"""
<style>
:root {{
  --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --text:{t['text']}; --muted:{t['muted']};
  --border:{t['border']}; --accent:{t['accent']}; --accent-soft:{t['accent_soft']}; --ok:{t['ok']}; --warn:{t['warn']};
  --err:{t['err']}; --radius:14px; --shadow:0 1px 2px rgba(16,24,40,.05), 0 1px 3px rgba(16,24,40,.06);
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: var(--bg) !important; color: var(--text); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: var(--surface) !important; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] * {{ color: var(--text); }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 6rem; max-width: 1120px; }}
h1, h2, h3, h4, p, li, label, span, div {{ color: inherit; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color: var(--text); }}
[data-testid="stMarkdownContainer"] h3 {{ font-size: 1.05rem; margin: .9rem 0 .35rem; }}
[data-testid="stMarkdownContainer"] table {{ display: block; overflow-x: auto; max-width: 100%; border-collapse: collapse; font-size: .9rem; margin: .4rem 0 .8rem; }}
[data-testid="stMarkdownContainer"] th {{ background: var(--surface2); color: var(--text); text-align: left; padding: .45rem .7rem; border: 1px solid var(--border); }}
[data-testid="stMarkdownContainer"] td {{ padding: .45rem .7rem; border: 1px solid var(--border); vertical-align: top; }}
[data-testid="stCaptionContainer"], .muted {{ color: var(--muted) !important; }}
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.1rem; box-shadow: var(--shadow); }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--border) !important; border-radius: 10px; }}
[data-testid="stExpander"] summary {{ color: var(--text) !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--border) !important; border-radius: var(--radius) !important; background: var(--surface); }}
[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1rem; }}
textarea, input, [data-baseweb="select"] > div, [data-baseweb="input"] {{ background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }}
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div {{ background: var(--bg) !important; }}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{ background: var(--surface) !important; border-color: var(--border) !important; border-radius: 14px; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface) !important; border: 1.5px dashed var(--accent); border-radius: var(--radius); padding: 1.6rem; }}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] > button, [data-testid="stPopover"] button {{
  border-radius: 10px; border: 1px solid var(--border); background: var(--surface); color: var(--text);
}}
.stButton button:hover, .stDownloadButton button:hover {{ background: var(--surface2) !important; border-color: var(--accent) !important; color: var(--accent) !important; }}
.stButton button:disabled {{ opacity: .45; background: var(--surface) !important; color: var(--muted) !important; }}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind^="primary"] {{ background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }}
.stButton button[kind="primary"]:hover {{ filter: brightness(1.08); color: #fff !important; }}
/* suggestion buttons read as cards */
.st-key-suggestions .stButton button {{ text-align: left; justify-content: flex-start; min-height: 3.4rem; padding: .7rem .9rem; white-space: normal; line-height: 1.35; }}
.st-key-suggestions .stButton button p {{ text-align: left; }}

/* layout pieces */
.brand {{ display:flex; align-items:center; gap:.65rem; margin:.1rem 0 1.1rem; }}
.brand-logo {{ width:36px; height:36px; border-radius:11px; background:linear-gradient(135deg, var(--accent), #7C3AED); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; }}
.brand-title {{ font-weight:700; font-size:1.02rem; line-height:1.15; }}
.brand-sub {{ font-size:.76rem; color:var(--muted); }}
.eyebrow {{ font-size:.74rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--accent); }}
.page-title {{ font-size:1.65rem; font-weight:750; margin:.1rem 0 .15rem; letter-spacing:-.01em; }}
.page-sub {{ color:var(--muted); margin-bottom:1.1rem; }}
.section-title {{ font-weight:700; font-size:1.02rem; margin:1.3rem 0 .6rem; }}
.pill {{ display:inline-flex; align-items:center; gap:.25rem; padding:.12rem .55rem; border-radius:999px; font-size:.74rem; font-weight:600; background:var(--accent-soft); color:var(--accent); margin:0 .3rem .2rem 0; white-space:nowrap; }}
.pill.ok {{ color:var(--ok); background:color-mix(in srgb, var(--ok) 12%, transparent); }}
.pill.warn {{ color:var(--warn); background:color-mix(in srgb, var(--warn) 14%, transparent); }}
.pill.err {{ color:var(--err); background:color-mix(in srgb, var(--err) 12%, transparent); }}
.pill.info {{ color:var(--muted); background:var(--surface2); }}
.hero {{ border:1px solid var(--border); border-radius:18px; padding:1.8rem 1.8rem 1.5rem; margin-bottom:1rem;
  background: radial-gradient(1200px 200px at 0% 0%, var(--accent-soft), transparent 60%), var(--surface); box-shadow: var(--shadow); }}
.hero h2 {{ font-size:1.45rem; margin:.2rem 0 .4rem; font-weight:750; letter-spacing:-.01em; }}
.hero p {{ color:var(--muted) !important; margin:0 0 .8rem; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.75rem; margin:.2rem 0 1rem; }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.85rem 1rem; box-shadow: var(--shadow); }}
.kpi .label {{ font-size:.78rem; color:var(--muted); font-weight:600; }}
.kpi .value {{ font-size:1.6rem; font-weight:750; line-height:1.2; margin-top:.15rem; color:var(--text); }}
.kpi .hint {{ font-size:.74rem; color:var(--muted); margin-top:.1rem; }}
.empty {{ text-align:center; padding:2.4rem 1.2rem; color:var(--muted); border:1px dashed var(--border); border-radius:var(--radius); background:var(--surface); }}
.empty .icon {{ font-size:1.8rem; }}
.empty .title {{ font-weight:700; color:var(--text); margin:.35rem 0 .2rem; }}
.skeleton {{ height:3.4rem; border-radius:10px; margin-bottom:.6rem; background:linear-gradient(90deg, var(--surface2) 25%, var(--surface) 50%, var(--surface2) 75%); background-size:200% 100%; animation:shimmer 1.3s infinite; }}
@keyframes shimmer {{ 0% {{ background-position:200% 0; }} 100% {{ background-position:-200% 0; }} }}

/* citations */
sup.cite {{ color:var(--accent); font-weight:700; cursor:help; margin-left:1px; }}
.src-card {{ display:flex; gap:.7rem; align-items:flex-start; }}
.src-num {{ flex:none; width:1.7rem; height:1.7rem; border-radius:8px; background:var(--accent-soft); color:var(--accent); font-weight:800; font-size:.85rem; display:flex; align-items:center; justify-content:center; }}
.src-doc {{ font-weight:650; font-size:.92rem; word-break:break-word; }}
.src-where {{ font-size:.8rem; color:var(--muted); margin:.1rem 0 .3rem; word-break:break-word; }}
.meter {{ height:4px; border-radius:4px; background:var(--surface2); overflow:hidden; margin-top:.35rem; }}
.meter > span {{ display:block; height:100%; background:var(--accent); border-radius:4px; }}
.excerpt {{ background:var(--surface2); border-left:3px solid var(--accent); border-radius:6px; padding:.6rem .8rem; font-size:.87rem; white-space:pre-wrap; color:var(--text); }}
.excerpt mark {{ background:color-mix(in srgb, var(--accent) 22%, transparent); color:var(--text); padding:0 .1rem; border-radius:3px; font-weight:600; }}
.muted-excerpt {{ border-left-color:var(--border); color:var(--muted); margin-bottom:.5rem; }}
.src-meta {{ font-size:.8rem; color:var(--muted); margin-bottom:.35rem; }}
.src-table {{ border-collapse:collapse; font-size:.83rem; margin:.1rem 0 .8rem; width:100%; }}
.src-table th {{ text-align:left; color:var(--muted); font-weight:600; padding:.2rem .8rem .2rem 0; white-space:nowrap; vertical-align:top; width:6.5rem; }}
.src-table td {{ padding:.2rem 0; color:var(--text); word-break:break-word; }}
.callout {{ border:1px solid color-mix(in srgb, var(--warn) 45%, transparent); background:color-mix(in srgb, var(--warn) 9%, var(--surface)); border-radius:12px; padding:.8rem 1rem; margin:.6rem 0; }}
.callout .title {{ font-weight:700; color:var(--warn); margin-bottom:.25rem; }}
.answer-meta {{ font-size:.76rem; color:var(--muted); margin-top:.6rem; }}

/* tables */
.doc-table {{ overflow-x:auto; border:1px solid var(--border); border-radius:12px; background:var(--surface); margin-bottom:1rem; }}
.doc-table table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
.doc-table th {{ text-align:left; font-weight:600; color:var(--muted); background:var(--surface2); padding:.55rem .75rem; border-bottom:1px solid var(--border); white-space:nowrap; }}
.doc-table td {{ padding:.5rem .75rem; border-bottom:1px solid var(--border); color:var(--text); white-space:nowrap; }}
.doc-table tr:last-child td {{ border-bottom:none; }}
.conv-title {{ font-weight:650; }}
.conv-meta {{ font-size:.8rem; color:var(--muted); }}

/* responsive */
@media (max-width: 768px) {{
  .block-container {{ padding: 1rem .8rem 6rem; }}
  .page-title {{ font-size:1.3rem; }}
  .hero {{ padding:1.2rem; }}
  .hero h2 {{ font-size:1.2rem; }}
  .kpi-grid {{ grid-template-columns:repeat(2, 1fr); gap:.5rem; }}
  .kpi .value {{ font-size:1.3rem; }}
  [data-testid="stChatMessage"] {{ padding:.75rem; }}
  .doc-table td, .doc-table th {{ padding:.45rem .55rem; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    st.markdown(
        (f"<div class='eyebrow'>{escape(eyebrow)}</div>" if eyebrow else "")
        + f"<div class='page-title'>{escape(title)}</div>"
        + (f"<div class='page-sub'>{escape(subtitle)}</div>" if subtitle else ""),
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{escape(text)}</div>", unsafe_allow_html=True)


def empty_state(icon: str, title: str, body_html: str = "") -> None:
    st.markdown(f"<div class='empty'><div class='icon'>{icon}</div><div class='title'>{escape(title)}</div>"
                f"<div>{body_html}</div></div>", unsafe_allow_html=True)


def kpi_tiles(items: Sequence[Tuple[str, str, str]]) -> None:
    """(label, value, hint) tiles in a responsive grid."""
    tiles = "".join(f"<div class='kpi'><div class='label'>{escape(label)}</div><div class='value'>{escape(str(value))}</div>"
                    f"<div class='hint'>{escape(hint)}</div></div>" for label, value, hint in items)
    st.markdown(f"<div class='kpi-grid'>{tiles}</div>", unsafe_allow_html=True)


def status_pill(status: str) -> str:
    kind, icon, label = STATUS.get(status, ("info", "•", status.capitalize()))
    return f"<span class='pill {kind}'>{icon} {escape(label)}</span>"


def pill(text: str, kind: str = "") -> str:
    return f"<span class='pill {kind}'>{escape(text)}</span>"


def skeletons(n: int) -> str:
    return "".join("<div class='skeleton'></div>" for _ in range(n))


def html_table(rows: List[Dict], raw_html_cols: Iterable[str] = ()) -> None:
    """Theme-aware table (st.dataframe draws on a canvas that ignores the custom theme)."""
    if not rows:
        return
    raw = set(raw_html_cols)
    head = "".join(f"<th>{escape(k)}</th>" for k in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{v if k in raw else escape(str(v))}</td>" for k, v in r.items()) + "</tr>"
                   for r in rows)
    st.markdown(f"<div class='doc-table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
                unsafe_allow_html=True)


def bar_chart(data: List[Dict], label: str, value: str, value_title: str, dark: bool,
              tooltip_extra: Optional[List[str]] = None) -> None:
    """Sorted horizontal bars for one magnitude series: one hue, thin rounded bars, recessive axes, tooltips."""
    import altair as alt
    import pandas as pd

    t = theme(dark)
    df = pd.DataFrame(data)
    tooltip = [alt.Tooltip(f"{label}:N", title=label.capitalize()), alt.Tooltip(f"{value}:Q", title=value_title)]
    tooltip += [alt.Tooltip(f"{c}:N") for c in (tooltip_extra or [])]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.62}, color=t["chart"])
        .encode(
            x=alt.X(f"{value}:Q", title=value_title, axis=alt.Axis(tickMinStep=1, grid=True, gridColor=t["border"],
                                                                   domain=False, ticks=False, labelColor=t["muted"],
                                                                   titleColor=t["muted"])),
            y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelColor=t["text"], domain=False, ticks=False,
                                                                       labelLimit=230)),
            tooltip=tooltip,
        )
        .properties(height=max(120, 30 * len(df)), background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=12, titleFontSize=12)
    )
    st.altair_chart(chart, use_container_width=True)
