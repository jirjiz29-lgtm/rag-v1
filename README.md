# Company Knowledge Assistant

An internal RAG chatbot that answers employee questions from company documents (policies, SOPs, handbooks, training material), with source citations on every answer. When the documents don't contain the answer, it says so instead of guessing.

It answers from a **fixed knowledge base**: the 10 GypsyCircle documents in `knowledge_base/`. They're committed with the code and indexed automatically when the app starts. There's no upload option, and the API refuses uploads, deletes and re-indexing (`KB_READ_ONLY=true`).

- **LLM:** Google Gemini (`gemini-3.8-flash`, falling back to `gemini-3.5-flash-lite` when overloaded) through the official `google-genai` SDK. Anthropic Claude (`claude-opus-5-5`, official `anthropic` SDK) is still supported: set `LLM_PROVIDER=anthropic`.
- **RAG framework:** LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-chroma`, `langchain-huggingface`)
- **Vector DB:** ChromaDB, persisted locally in `vectorstore/`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally
- **Backend:** FastAPI · **Frontend:** Streamlit · **Parsing:** pypdf, python-docx, plain text

---

## 1. Architecture

```
┌──────────────────────────┐   HTTP/JSON   ┌───────────────────────────────────────────────┐
│  Streamlit dashboard     │ ────────────▶ │  FastAPI backend (main.py)                     │
│  frontend/dashboard.py   │               │                                                │
│  • Chat + history        │               │  KnowledgeBaseManager  ──▶ document_loader     │
│  • Knowledge base mgmt   │               │   (dedupe, update,          (PDF/DOCX/TXT/MD,  │
│  • Upload                │               │    delete, re-index)         pages, headings)  │
│  • Settings              │               │        │                         │             │
│  (no API keys here)      │               │        ▼                         ▼             │
└──────────────────────────┘               │  registry.json            text_splitter        │
                                           │                                  │             │
                                           │                     embeddings (MiniLM, local) │
                                           │                                  ▼             │
                                           │  Retriever ◀──────────── ChromaDB (vectorstore/)│
                                           │      │                                         │
                                           │      ▼                                         │
                                           │  RAGPipeline ──▶ Gemini / Claude (JSON output) │
                                           │      └─▶ citation validation ─▶ answer + sources│
                                           └───────────────────────────────────────────────┘
```

The frontend never sees the Anthropic API key. It only calls the backend's REST API.

## 2. How RAG works here

**Ingestion** (`backend/knowledge_base.py`)
1. **Validate**: extension (`.pdf/.docx/.txt/.md`), size limit, non-empty, PDF/DOCX file signature, sanitised filename (no path traversal).
2. **Deduplicate**: a SHA-256 of the content is checked against the registry. Identical content is skipped, even under another filename. Uploading a *changed* file under the same name replaces the old version (old chunks are deleted and the version number goes up).
3. **Extract** (`document_loader.py`): text is extracted per page (PDF) and split into sections using headings: DOCX heading styles, numbered headings (`3. Paid Time Off`), ALL-CAPS lines, and short Title Case lines. Markdown files are split only on `#` headings (outside code fences), and sub-sections get a path such as `4. Refunds > Partial refunds`. DOCX tables are kept in document order. Each piece carries `source`, `page`, `section`, `category` and `file_type` metadata.
4. **Chunk** (`text_splitter.py`): sections are chunked separately (a chunk never mixes two headings), then capped at `CHUNK_SIZE` tokens with `CHUNK_OVERLAP`, counted with the embedding model's own tokenizer. The cap stays at or below MiniLM's 256-token input limit, because text past that limit is never embedded.
5. **Embed and store** (`embeddings.py`, `vector_database.py`): normalised MiniLM vectors go into a cosine-space Chroma collection with stable chunk ids (`<doc_id>:<n>`). Each document's registry entry records the chunking/embedding settings it was indexed with. Changing them re-indexes unchanged files on the next start.

**Answering** (`backend/retriever.py`, `backend/rag_pipeline.py`)
1. **Query processing**: whitespace is normalised. Short follow-ups ("and after 4 hours?") are expanded with the previous question so retrieval has context.
2. **Hybrid search** (`retriever.py`, `hybrid.py`): semantic search (embeddings) and BM25 keyword search (over each chunk prefixed with its document title and section heading) each return `CANDIDATE_POOL` chunks, fused by reciprocal rank. Optional **metadata filters** (category, specific documents) apply to both.
3. **Reranking** (`reranker.py`): a cross-encoder (`RERANKER_MODEL`) reads the question with each candidate and re-orders them. Chunks scoring below `RERANK_KEEP_RATIO` × the best are dropped, and at most `TOP_K` are sent to the LLM.
4. **Hard fallback**: if no chunk reaches `RELEVANCE_THRESHOLD` cosine similarity, the bot returns the fallback message **without calling the LLM**.
4. **Grounded generation**: chunks are sent to the LLM as `<source id="S1" document=… page=… section=…>` blocks under a strict system prompt (answer only from sources, cite `[S#]`, flag conflicts, never invent policies). Structured outputs force a JSON reply: `answer`, `found_in_documents`, `cited_sources`, `conflicts`.
5. **Citation validation**: every cited id is checked against what was actually retrieved. Unknown ids (`[S99]`) are stripped. An answer with no valid citation, or one the model marks as not found, is replaced with the exact fallback message. **Fake citations can't reach the user.**
6. **Citation building** (`citations.py`): the model only ever cites `[S#]`. Document name, page, section and excerpt are then copied from the chunk's stored metadata, never from the model's text:
   - **Page** is shown only for PDFs and only if it lies within the document's page count. DOCX, TXT and Markdown have no fixed pages, so the citation says so instead of guessing.
   - **Section** is the nearest heading found at indexing time. Text before the first heading is labelled as such rather than given an invented heading.
   - **Highlighted excerpt**: the sentences in the passage that best match the answer's claims are marked. They are verbatim substrings of the passage.
   - **Page references in the model's prose** such as "(page 7)" are removed unless a retrieved passage is on that page.
6. **Response**: the answer, cited sources (document, page, section, excerpt, relevance), all retrieved chunks for inspection, and any detected conflicts.

Fallback message (exact):
> I couldn't find reliable information about this in the company's knowledge base. Please consult the relevant department or upload the required documentation.

**Note on the relevance threshold.** No retrieval score separates "answerable" from "not covered" on this knowledge base: the weakest answerable question scores 0.248 cosine and the strongest unanswerable one 0.396. The reranker scores correct paraphrase matches as low as off-topic ones. So the threshold (default 0.20) only filters clearly unrelated questions. For in-domain questions the documents can't answer, the model decides `found_in_documents=false`, and the live check returned the exact fallback for 10 of 10 such questions. Details are in [docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md).

## 3. Project structure

```
company-knowledge-bot/
├── app.py                      # Streamlit entry point (streamlit run app.py)
├── main.py                     # FastAPI app (uvicorn main:app)
├── config.py                   # Settings from environment / .env
├── start.sh                    # Starts backend + frontend together
├── requirements.txt  .env.example  .gitignore  pytest.ini
├── .streamlit/config.toml
├── knowledge_base/             # The fixed GypsyCircle documents + categories.json (indexed on startup)
├── vectorstore/                # ChromaDB data + document_registry.json (created at runtime)
├── data/chat_history.json      # Saved conversations (created at runtime)
├── sample_documents/           # Fictional sample docs for demos/tests
├── scripts/generate_sample_docs.py
├── backend/
│   ├── document_loader.py      # PDF/DOCX/TXT/MD extraction, page + section metadata
│   ├── text_splitter.py        # Chunking with stable ids
│   ├── embeddings.py           # HuggingFace sentence-transformers
│   ├── vector_database.py      # Chroma wrapper (add/delete/search/count)
│   ├── registry.py             # JSON registry: status, hash, dates, versions
│   ├── knowledge_base.py       # Validation, dedupe, update, delete, re-index, folder sync
│   ├── retriever.py            # Semantic search + metadata filters + threshold
│   ├── prompts.py              # System prompt, fallback text, JSON schema
│   ├── rag_pipeline.py         # LLM interface, Claude client, grounding/citation validation
│   ├── gemini_client.py        # Gemini client with model fallback
│   └── services.py             # Dependency wiring
├── frontend/
│   ├── dashboard.py            # Streamlit UI
│   └── api_client.py           # HTTP client for the backend
└── tests/                      # pytest suite (offline + optional live LLM tests)
```

## 4. Installation

Requires **Python 3.10+** (developed on 3.12). macOS's system Python is 3.9, which is too old.

```bash
cd company-knowledge-bot
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

No Python 3.10+ installed? [uv](https://docs.astral.sh/uv/) can provide one:

```bash
pip install --user uv && uv python install 3.12 && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

The first run downloads the MiniLM embedding model (~90 MB) from Hugging Face. After that it's cached.

## 5. Configuration

```bash
cp .env.example .env
```

Then set the key for your provider: `GEMINI_API_KEY` (default) or `ANTHROPIC_API_KEY` together with `LLM_PROVIDER=anthropic`. All options:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `GEMINI_API_KEY` | – | Gemini key from <https://aistudio.google.com/apikey>. Read only by the backend. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Main Gemini model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite` | Comma-separated models tried in order when the main one is overloaded (429/5xx) or times out |
| `LLM_MAX_TOKENS` / `LLM_TIMEOUT_SECONDS` | `16000` / `60` | Gemini output cap / per-model request timeout |
| `ANTHROPIC_API_KEY` | – | Used when `LLM_PROVIDER=anthropic`. Must be an Anthropic Console key (starts with `sk-ant-`); the dashboard warns if it doesn't look like one. |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model id |
| `CLAUDE_EFFORT` | `medium` | `low`/`medium`/`high`/`xhigh`/`max`: answer thoroughness vs. latency/cost |
| `CLAUDE_ENABLE_FALLBACKS` | `true` | If a request is declined by a safety classifier, the API retries on Anthropic's recommended fallback model (Claude API only; set `false` on Bedrock/Vertex/Foundry) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `30` | Max tokens per chunk / overlap (embedding-model tokens; keep ≤ 256) |
| `TOP_K` | `6` | Max chunks sent to the LLM per question |
| `HYBRID_SEARCH` | `true` | Combine semantic and BM25 keyword search |
| `CANDIDATE_POOL` | `20` | Candidates from each search before reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker (`none` disables; about 90 MB, downloaded on first use) |
| `RERANK_KEEP_RATIO` | `0.001` | Drop reranked chunks scoring below this fraction of the best |
| `RELEVANCE_THRESHOLD` | `0.20` | Fallback without calling the LLM when no chunk reaches this cosine |
| `HISTORY_TURNS` | `6` | Previous Q&A turns sent for follow-up context |
| `KB_READ_ONLY` | `true` | Fixed knowledge base: no uploads, deletes or re-indexing (API returns 403; UI hides them) |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit (only used when `KB_READ_ONLY=false`) |
| `KNOWLEDGE_BASE_DIR` / `VECTORSTORE_DIR` | `knowledge_base` / `vectorstore` | Storage locations |
| `API_HOST` / `API_PORT` / `BACKEND_URL` | `127.0.0.1` / `8000` / `http://127.0.0.1:8000` | Server addresses |

After changing chunk settings, restart the API and click **Re-index all** on the Knowledge Base page.

**Which key goes where:** Gemini keys from Google AI Studio (they start with `AQ.` or `AIza`) go in `GEMINI_API_KEY`. Anthropic keys (`sk-ant-…`) go in `ANTHROPIC_API_KEY`. A Gemini key cannot call Claude and vice versa. Restart the API after changing `.env`.

**"The AI service is overloaded or timed out"**: Google's servers were busy for the main and all fallback models. Retry, or add another model to `GEMINI_FALLBACK_MODELS`.

## 6. Running

Both servers together:

```bash
./start.sh
```

Or separately, in two terminals:

```bash
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

```bash
.venv/bin/streamlit run app.py
```

Open http://localhost:8501. API docs are at http://127.0.0.1:8000/docs.

The knowledge base is indexed on startup, so there's nothing to load. The fictional "Brightpath Solutions" files in `sample_documents/` are only test fixtures; don't copy them into `knowledge_base/`.

### Deploying to Streamlit Community Cloud

Community Cloud runs only `streamlit run app.py`; there is no separate API server. When `BACKEND_URL` is unset (or `embedded`), the dashboard runs the FastAPI app inside the Streamlit process through FastAPI's in-memory client: same endpoints, validation and errors, one process. Locally, `.env` keeps `BACKEND_URL=http://127.0.0.1:8000`, so `start.sh` works as before.

1. Push the repo to GitHub (done: `jirjiz29-lgtm/rag-v1`).
2. Go to <https://share.streamlit.io>, choose **Create app → Deploy a public app from GitHub**, and set:
   - Repository `jirjiz29-lgtm/rag-v1`, branch `main`, main file path `app.py`
   - **Python version:** any of 3.11–3.15 (the default is fine). On Linux `requirements.txt` installs the CPU-only PyTorch build for these versions, avoiding ~3 GB of NVIDIA GPU libraries Streamlit Cloud can't use.
   - **Advanced settings → Secrets** (TOML). Top-level secrets become environment variables, which `config.py` reads:
     ```toml
     LLM_PROVIDER = "gemini"
     GEMINI_API_KEY = "your-gemini-key"
     GEMINI_MODEL = "gemini-3.8-flash"
     GEMINI_FALLBACK_MODELS = "gemini-3.5-flash-lite"
     LLM_TIMEOUT_SECONDS = "20"
     ```
     Do **not** set `BACKEND_URL` there.
3. Deploy. The first start downloads the embedding model and indexes `knowledge_base/`, so it takes a minute or two.

Things to know before putting company documents on Community Cloud:
- **The documents ship with the code.** `knowledge_base/` is committed, so every start (including after the app sleeps or restarts) re-indexes the same 10 documents automatically. Nothing needs uploading.
- **The repo must be private to keep the documents private.** On a public GitHub repo, anyone can read `knowledge_base/` directly.
- **Apps are public by default.** Anyone with the URL could ask questions about your documents. In the app's **Settings → Sharing**, restrict who can view it, and follow your company's data policy.

**"The app's code is not connected to a remote GitHub repository"** (from the local **Deploy** button): Streamlit checks the git repo of the folder it was started from, so it only sees a repo and remote that already existed when the server started. Restart `streamlit run app.py` from the repo folder after pushing, or skip the button and create the app directly at share.streamlit.io as above.

## 7. Using the dashboard

- **AI Knowledge Assistant**: ask questions. Inline citations appear as numbers `[1]`, `[2]`; hover one to see its document, page and section. Under each answer, **Sources** lists one expandable entry per cited passage. Each entry has a details table (document, page or why there is none, section, category, passage position, relevance, upload date), the passage with its supporting sentences highlighted, and a **Show surrounding text** toggle that loads the passages before and after it. **Inspect all retrieved passages** shows everything the search returned, with cited ones marked. You also get a **Conflicting information** warning when documents disagree. **Search scope** limits a question to chosen categories or documents. **Clear chat** resets the conversation.
- **Chat History**: reopen, export (.txt, including each answer's sources) or delete past conversations. They're saved in `data/chat_history.json`.
- **Knowledge Base**: totals (documents, chunks, failures), a table of the fixed documents with category, status, chunks and size, and a viewer for each document's indexed passages.
- **Upload Documents** (only when `KB_READ_ONLY=false`): pick multiple PDF/DOCX/TXT/MD files and a category. You get a result per file: Indexed, Updated, Duplicate (skipped), Failed (e.g. a scanned PDF with no text), or Rejected (wrong type, too large, empty). A data-handling notice reminds users to follow company data policy.
- **System Settings**: model/effort/key status, indexing configuration, per-session top-k and relevance threshold. The dark/light theme switch is in the sidebar.

Questions that work well with the samples:
- "How many PTO days do I get and how many can I carry over?"
- "Explain the vendor onboarding procedure in simple terms."
- "Give me step-by-step instructions to submit an expense claim."
- "Compare the remote work rules in the handbook and the remote work policy." (a conflict: 3 days vs 2 days)
- "What is the daily meal allowance when travelling?" (a conflict: $50 vs $75)
- "What is our parental leave policy?" (not in the documents, so you get the fallback)

### REST API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, whether the LLM is configured, counts |
| GET | `/settings` | Public configuration (never secrets) |
| POST | `/chat` | `{question, history?, categories?, doc_ids?, top_k?, relevance_threshold?}` |
| GET | `/documents` · `/stats` | List documents · totals |
| POST | `/documents/upload` | multipart `files[]` + `category` |
| GET | `/documents/{id}` · `/documents/{id}/chunks` | Metadata · indexed chunks |
| DELETE | `/documents/{id}` | Remove file, chunks and registry entry |
| POST | `/documents/{id}/reindex` · `/documents/reindex` · `/documents/sync` | Re-index one · all · scan folder |

### The fixed knowledge base

`knowledge_base/` holds the 10 GypsyCircle Markdown documents. `knowledge_base/categories.json` assigns each one a category (Employee Handbook, Customer Support, Sales and so on), which the dashboard's **Search scope** filter uses.

On every start the app compares the folder with the index. New or changed files are indexed, with a changed file getting a new version. Files that were removed are dropped from the index. To change what the assistant knows:

1. Add, edit or delete files in `knowledge_base/` (PDF, DOCX, TXT or MD) and update `categories.json`.
2. Restart the app. On Streamlit Community Cloud, commit and push; the app redeploys and re-indexes.

While `KB_READ_ONLY=true` (the default), the write endpoints (`/documents/upload`, `DELETE /documents/{id}`, `/reindex`, `/sync`) return **403**, and the dashboard hides the Upload page and the management buttons. Set `KB_READ_ONLY=false` to bring back upload, delete and re-index for local administration.

## 8. Testing

```bash
.venv/bin/python -m pytest -v
```

The offline suite (93 tests) uses real document parsing, real embeddings, the real reranker and a real Chroma store. A deterministic `FakeLLM` stands in for the model, so no API key or network is needed once the embedding and reranker models are cached. It covers the required scenarios:

| Requirement | Tests |
|---|---|
| Correct answers from documents | `test_retrieves_correct_document_for_question`, `test_answer_is_grounded_in_top_chunk` |
| Valid source citations | `test_sources_include_document_page_section_and_excerpt`, `test_fabricated_citations_are_removed`, `test_answer_without_valid_citations_falls_back` |
| Fallback for unsupported questions | `test_unrelated_question_returns_fallback_without_calling_llm`, `test_llm_reporting_not_found_returns_exact_fallback`, `test_chat_on_empty_kb_returns_fallback` |
| Multi-document search | `test_multi_document_retrieval`, `test_multi_document_answer_cites_both`, `test_metadata_filter_restricts_documents` |
| Duplicate uploads | `test_duplicate_upload_is_skipped`, `test_changed_document_with_same_name_replaces_old_version`, `test_duplicate_upload_via_api` |
| Deleted docs leave retrieval | `test_deleted_document_is_not_retrieved`, `test_upload_list_chat_delete_flow` |
| Conflicting information | `test_conflicting_sources_are_both_retrieved`, `test_conflicts_are_passed_through_with_documents` |
| Markdown support | `test_markdown_sections_and_code_fences` (nested heading paths, code fences ignored, indexed and retrieved) |
| Validation & security | unsupported/empty/fake-PDF uploads, path traversal, key value never returned by `/health` or `/settings`, missing-key error |
| Gemini client | request shape (system instruction, JSON schema, role mapping), fallback to the next model on 429/503/504, no fallback on auth/404 errors, error mapping, blocked/truncated/unreadable responses |
| Embedded mode | `tests/test_embedded.py`: the dashboard's client drives upload, chat with citations, chunk inspection, delete and error reporting in-process |
| Fixed knowledge base | `tests/test_fixed_kb.py`: all 10 documents present and categorised, indexed at startup with their categories, every write endpoint returns 403, editing or removing a file updates the index on restart |
| Retrieval quality | `tests/test_retrieval_eval.py`: before vs after on a labelled 57-question set (recall, full and multi-document coverage, paraphrase recall, MRR, context size and precision, no answerable question withheld), plus unit tests for BM25, rank fusion, chunk selection, reranker ordering and fallback, and the off-topic gate. See [docs/RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md) |
| Citations | `tests/test_citations.py`: every citation field equals the stored chunk metadata (PDF, DOCX, TXT), exact PDF page and section, no page numbers for non-paged formats, out-of-range pages rejected, invented `(page N)` and `[S#]` references removed, highlights verbatim, heading lines never highlighted |
| Claude client | exact request shape (model, adaptive thinking, effort, JSON schema, prompt caching, fallbacks beta) and error mapping, via the real SDK against a mocked HTTP transport |

**Live LLM tests** (`tests/test_live_llm.py`) run against the configured provider, only when its key is set. They check a cited answer, step-by-step instructions, the fallback for an uncovered question, and conflict detection on the meal-allowance discrepancy:

```bash
.venv/bin/python -m pytest tests/test_live_llm.py -v
```

**Citation audit** (`scripts/audit_citations.py`) asks real cross-document questions through the configured LLM. It then checks each citation against the vector store independently: the passage exists, and its document name, page, section, excerpt and highlights match what is stored. It prints every citation with ✓ or the problem found, and exits non-zero if any check fails:

```bash
.venv/bin/python scripts/audit_citations.py            # your knowledge base
.venv/bin/python scripts/audit_citations.py --samples  # sample PDF/DOCX/TXT files (exercises page numbers)
```

## 9. Security & reliability

- The API key lives only in the backend's environment (`.env` is git-ignored). `/settings` and `/health` expose only whether a key is configured.
- Uploads are checked for type, size, emptiness and file signature. Filenames are sanitised, and per-file results are reported without failing the whole batch.
- Logs record file names, statuses and error types, **never document contents**.
- LLM API errors (auth, overload, rate limit, network, server, refusal, truncation) become clear HTTP errors (429/502/503) that the UI shows inline.
- Retrieved text is labelled as data in the prompt, and the model is told to ignore instructions inside documents (prompt-injection hygiene).
- Embeddings run locally. Only the retrieved excerpts and the question are sent to Anthropic.
- **Data policy:** only upload documents you're authorised to share, and handle them under your company's data classification and retention policies.

## 10. Limitations & next steps

- Scanned/image-only PDFs need OCR before upload (they're reported as "no extractable text").
- Heading detection is heuristic for PDFs and TXT. Well-structured DOCX gives the best section metadata.
- Single-node design: the registry is a JSON file and indexing runs inside the upload request. For many users or large files, move ingestion to a background worker and add authentication/SSO plus per-department access control.
