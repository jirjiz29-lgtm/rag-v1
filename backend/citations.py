"""Build citations strictly from retrieved chunk metadata.

Nothing here asks the model for a document name, page or section. Every field shown to the
user comes from the metadata stored with the chunk at indexing time, and the "relevant
excerpt" highlights are verbatim sentences copied out of the chunk text.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

_CITATION = re.compile(r"\[(S\d+)\]")
_LEADING_MARKERS = re.compile(r"^\s*((?:\[S\d+\]\s*)*)")
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Sentence / bullet boundaries inside a chunk.
_SEGMENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
# "(page 4)", "(p. 4)", "(pg 4)", "(pp. 4-5)" written by the model in its own prose.
_PAGE_MENTION = re.compile(r"\s*\((?:on\s+)?(?:page|pages|p\.|pp\.|pg\.?)\s*(\d+)(?:\s*[-–]\s*(\d+))?\)", re.I)

_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "have", "has", "was", "were", "will", "must",
    "should", "can", "not", "any", "all", "you", "your", "our", "their", "its", "into", "than", "then", "when",
    "who", "what", "which", "how", "also", "only", "each", "per", "may", "such", "these", "those", "there",
    "they", "them", "been", "being", "use", "used", "within", "about", "after", "before", "other", "more",
}

_PAGED_TYPES = {"pdf"}


def valid_page(metadata: Dict[str, Any]) -> Optional[int]:
    """Return the chunk's page number only if it is real: a paged format and within the page count."""
    if str(metadata.get("file_type", "")).lower() not in _PAGED_TYPES:
        return None
    try:
        page = int(metadata.get("page"))
    except (TypeError, ValueError):
        return None
    total = metadata.get("total_pages")
    if page < 1 or (isinstance(total, int) and total > 0 and page > total):
        return None
    return page


def valid_section(metadata: Dict[str, Any]) -> Optional[str]:
    section = str(metadata.get("section") or "").strip()
    # "General" was the loader's placeholder for text before any heading in older indexes.
    return section if section and section != "General" else None


def location_label(document: str, page: Optional[int], total_pages: Optional[int], section: Optional[str]) -> str:
    """Human-readable citation, e.g. `handbook.pdf · p. 2 of 3 · 4. Remote Work`."""
    parts = [document]
    if page:
        parts.append(f"p. {page} of {total_pages}" if total_pages else f"p. {page}")
    if section:
        parts.append(section)
    return " · ".join(parts)


def page_note(file_type: str, page: Optional[int]) -> str:
    """Explain a missing page number instead of leaving the reader guessing."""
    if page:
        return ""
    if file_type in _PAGED_TYPES:
        return "Page number unavailable for this excerpt"
    return f"{file_type.upper() or 'This'} files have no fixed page numbers"


def _tokens(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _heading_key(text: str) -> str:
    """Normalise a heading for comparison: drop '#', numbering like '4.' / '2.3' / 'IV.', case and ':'."""
    text = text.strip().lstrip("#").strip()
    text = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+", "", text)
    return text.rstrip(":").strip().lower()


def _segments(text: str, heading: Optional[str] = None) -> List[str]:
    """Sentences and bullet lines of a chunk, skipping heading lines (they restate the section, not evidence)."""
    heading_key = _heading_key((heading or "").split(" > ")[-1])
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or (heading_key and _heading_key(stripped) == heading_key):
            continue
        out.extend(s.strip() for s in _SEGMENT.split(line) if len(s.strip()) >= 12 and not s.strip().endswith(":"))
    return out


def claims_for(answer: str, source_id: str) -> List[str]:
    """Answer sentences/lines that cite `source_id` (whole answer if none cite it explicitly)."""
    pieces: List[str] = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if not piece.strip():
            continue
        # "...every 60 minutes. [S1] Next step." - markers right after a full stop belong to the sentence before.
        lead = _LEADING_MARKERS.match(piece)
        if pieces and lead.group(1):
            pieces[-1] += " " + lead.group(1).strip()
            piece = piece[lead.end(1):]
        if piece.strip(" .,;"):
            pieces.append(piece.strip())
    tagged = [p for p in pieces if f"[{source_id}]" in p]
    return tagged or [answer]


def highlight_sentences(chunk_text: str, claims: List[str], max_sentences: int = 6,
                        heading: Optional[str] = None) -> List[str]:
    """Pick the chunk sentences that best support the claims. Returned strings are verbatim substrings."""
    claim_tokens = [_tokens(_CITATION.sub("", c)) for c in claims]
    scored = []
    for order, segment in enumerate(_segments(chunk_text, heading)):
        seg_tokens = _tokens(segment)
        if len(seg_tokens) < 2:
            continue
        best = max((len(seg_tokens & ct) for ct in claim_tokens), default=0)
        ratio = best / len(seg_tokens)
        if best >= 2 and ratio >= 0.3:
            scored.append((best + ratio, order, segment))
    top = sorted(scored, reverse=True)[:max_sentences]
    return [segment for _, _, segment in sorted(top, key=lambda t: t[1])]


def scrub_page_mentions(answer: str, real_pages: Set[int]) -> str:
    """Remove parenthetical page references in the model's prose that no retrieved chunk supports."""

    def keep(match: re.Match) -> str:
        pages = {int(g) for g in match.groups() if g}
        return match.group(0) if pages and pages <= real_pages else ""

    return _PAGE_MENTION.sub(keep, answer)
