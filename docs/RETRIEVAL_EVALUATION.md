# Retrieval evaluation and optimisation

This document records how the Company Knowledge Assistant's retrieval was measured and improved, which changes helped, which didn't, and why.

## Before → after

Measured on the real GypsyCircle knowledge base with the app's own ingestion and retriever (`python scripts/eval_retrieval.py --production`):

| Metric | Before | After | |
|---|---|---|---|
| Evidence recall | 0.865 | **0.961** | +11% |
| Full coverage (all evidence found) | 0.830 | **0.936** | +13% |
| Multi-document coverage | 0.600 | **0.800** | +33% |
| Paraphrase recall | 0.875 | **1.000** | |
| MRR (rank of first evidence) | 0.787 | **0.878** | |
| Chunks sent to the LLM | 5.5 | **4.6** | −16% |
| Tokens sent to the LLM | 296 | **253** | −15% |
| Context precision | 0.200 | **0.266** | +33% |
| Answerable questions wrongly rejected | 1 | **0** | |
| Unanswerable questions rejected before the LLM | 4/10 | 2/10 | see [Thresholds](#5-similarity-thresholds) |
| Retrieval time per question (CPU) | ~5 ms | ~170 ms | reranker |

**Before:** heading-based chunks capped at 1000 characters, semantic search only, top-6, chunks dropped below cosine 0.25.
**After:** heading-based chunks capped at 200 tokens (overlap 30), hybrid semantic + BM25 search (20 candidates each, fused by RRF), cross-encoder reranking, chunks below 0.001 × the best reranker score dropped, top-6, fallback without the LLM if no chunk reaches cosine 0.20.

Grounding was checked live with Gemini after the change:
- **Unanswerable questions:** all 10 returned the exact fallback message. 2 were stopped before the LLM; the model declined the other 8.
- **Answerable questions:** every one tested was answered with citations. That includes "Can I offer an unhappy customer a free room upgrade on my own?", which the old retriever rejected outright.

## How it's measured

**Evaluation set:** `tests/data/retrieval_eval.json`, 57 questions over the 10 GypsyCircle documents:

| Type | Count | Example |
|---|---|---|
| Single document | 26 | "When is a booking considered confirmed?" |
| Paraphrase (different wording from the document) | 8 | "Am I allowed to paste customer details into ChatGPT?" → "Do not upload confidential data to public AI tools" |
| Keyword / identifier | 3 | "What is the gc_pax field for?" |
| Multi-document | 10 | "How should complaints be handled by support, and what should trip staff record about complaints?" |
| Unanswerable | 10 | "What is the price of the Manali trip?", "Who won the 2011 cricket world cup?" |

Each answerable question lists **evidence**: verbatim phrases from the documents that the retrieved context must contain. Some evidence items accept alternatives. Because matching is by phrase rather than by chunk id, the same set scores any chunking strategy fairly. A test checks that every phrase exists in its document.

**Metrics** (over answerable questions unless noted):
- **Recall:** share of evidence items found in the retrieved context.
- **Full coverage:** share of questions where *all* evidence was found. Multi-document coverage is the same, restricted to multi-document questions.
- **MRR:** 1 / rank of the first chunk containing evidence.
- **Chunks / tokens sent:** the context the LLM would receive. Tokens are counted with the embedding model's tokenizer.
- **Context precision:** share of sent chunks that contain evidence. This is a lower bound, since a chunk can be useful without containing an exact evidence phrase.
- **Answerable rejected:** answerable questions that got no context. These are the errors that matter most, because the user gets "I couldn't find…" for something the documents answer.
- **Unanswerable rejected:** unanswerable questions stopped before the LLM, which saves cost.

## The corpus shapes the results

Two facts explain several results below:
- **The documents are short.** Each GypsyCircle document is 305–438 tokens in total. Split at headings, they become 78 sections of 19–124 tokens (median 42).
- **The embedding model reads at most 256 tokens.** `all-MiniLM-L6-v2` truncates longer input, so text past token 256 is never embedded.

## Experiments

### 1–2. Chunk size (400 / 600 / 800 tokens) and overlap

| Strategy | Recall | Full cov. | Multi cov. | Paraphrase | MRR | Tokens sent | Answerable rejected |
|---|---|---|---|---|---|---|---|
| Heading sections, 1000 chars (before) | 0.865 | 0.830 | 0.600 | 0.875 | 0.787 | 296 | 1 |
| Fixed windows 400 tok, overlap 0 / 40 | 0.823 | 0.809 | 0.900 | 0.500 | 0.694 | 1283 | 6 |
| Fixed windows 400 tok, overlap 80 | 0.812 | 0.787 | 0.800 | 0.500 | 0.729 | 1263 | 5 |
| Fixed windows 600 tok (any overlap) | 0.780 | 0.766 | 0.900 | 0.250 | 0.668 | 1379 | 8 |
| Fixed windows 800 tok (any overlap) | 0.780 | 0.766 | 0.900 | 0.250 | 0.668 | 1379 | 8 |
| Heading sections, 128 / 200 / 256 tok | 0.865 | 0.830 | 0.600 | 0.875 | 0.787 | 296 | 1 |

**400–800-token windows are worse on every metric that matters.** Three things cause this:
- **Each window holds almost a whole document.** With documents of 305–438 tokens, a question's answer shares one vector with up to 10 unrelated sections, so the match is diluted. Paraphrase recall drops from 0.875 to 0.25–0.50.
- **Text past token 256 is never embedded.** Answers near the end of a document can't be found, which is why 6–8 answerable questions got no context.
- **They send 4× more text to the LLM**, about 1,300 tokens instead of about 300.

Their higher multi-document coverage is brute force: whole documents are sent, so coverage of *other* evidence comes along by accident. Overlap makes no difference, because a document rarely spans more than one window. 600 and 800 give identical results because every document fits in one window.

**Heading-based chunks are the right unit here.** Capping them at 128, 200 or 256 tokens gives identical results, since no section exceeds 124 tokens. The production cap is **200 tokens with 30 overlap**: it never splits today's sections, and it guarantees a future long section is split into pieces the model can fully embed. Chunk size is now counted in the embedding model's own tokens, not characters.

### 3. Better semantic retrieval: contextual headers

Prefixing each chunk with "document title > section heading" before embedding **lowered** recall from 0.865 to 0.805 (paraphrase 0.875 → 0.500). With chunks this short, a ~15-token header is a large share of the vector, so all chunks of a document start to look alike and the passage's own meaning is diluted.

The same prefix **helps keyword search** (see 6), so it's used there only. Embeddings use the plain chunk text.

### 4. Configurable top-k

With the final pipeline:

| top-k | Recall | Multi cov. | Tokens sent |
|---|---|---|---|
| 4 | 0.887 | 0.500 | 157 |
| 5 | 0.908 | 0.500 | 187 |
| 6 | 0.918 | 0.600 | 210 |
| 8 | 0.918 | 0.600 | 250 |

*(These rows use a keep-ratio of 0.01. The shipped 0.001 gives 0.961 / 0.800 at top-6.)*

Below 6, multi-document questions lose evidence; above 6 nothing improves. `TOP_K=6` remains the default. It is configurable via `.env`, per request via the API's `top_k`, and per session on the System Settings page.

### 5. Similarity thresholds

Score distributions (best candidate per question):

| | Answerable (47) | Unanswerable (10) |
|---|---|---|
| Best cosine | min **0.248**, median 0.62 | max **0.396**, median 0.32 |
| Best reranker logit | min **−11.1**, median 3.2 | max −4.9, median −10.3 |

**No threshold separates the two groups.** The old 0.25 cut-off was exactly why "free room upgrade" (best cosine 0.248) was wrongly rejected. The reranker is worse still as an absolute gate. It is trained on web search passages and gives a correct but differently worded bullet point ("Do not select solely on the lowest price" for "Should we always pick the cheapest hotel?") −11, the same as an off-topic question. Any absolute reranker threshold from p ≥ 0.05 up rejected 8–9 answerable questions.

**What ships:**
- **A conservative cosine floor of 0.20 per question.** It rejects 2 clearly off-topic questions (Ladakh weather, cricket) and no answerable ones.
- **A *relative* cut-off on reranked chunks** (keep chunks ≥ 0.001 × the best). This trims the tail without depending on the reranker's poorly calibrated absolute scores.
- **Everything else is left to the LLM.** It must return `found_in_documents: false` and the exact fallback when the context doesn't support an answer. The live check confirmed 10/10 fallbacks.

Fewer questions are now stopped before the LLM (2 instead of 4). That's the deliberate trade: the old 0.25 threshold achieved its 4 rejections only by also wrongly rejecting an answerable question.

### 6. Hybrid search (keyword + semantic)

| | Recall | Full cov. | Multi cov. | MRR |
|---|---|---|---|---|
| Semantic only | 0.865 | 0.830 | 0.600 | 0.787 |
| BM25 only | 0.918 | 0.872 | 0.600 | 0.878 |
| Hybrid, no headers | 0.908 | 0.872 | 0.700 | 0.795 |
| **Hybrid, headers for BM25** | **0.950** | **0.936** | **0.900** | 0.864 |

BM25 catches what embeddings blur: identifiers (`gc_pax`), exact policy terms ("screenshot", "interns") and, through the header prefix, **document and section names**. A question such as "Compare what the marketing guidelines and the sales SOP say about…" now matches those documents' chunks by name. That raises multi-document coverage from 0.6 to 0.9. Reciprocal-rank fusion combines the two rankings without calibrating BM25 scores against cosine similarities.

### 7. Reranking

| On top of hybrid | Recall | Full cov. | Multi cov. | Paraphrase | MRR |
|---|---|---|---|---|---|
| No reranker | 0.950 | 0.936 | 0.900 | 0.875 | 0.864 |
| **Cross-encoder, plain chunk text** | **0.982** | **0.957** | 0.800 | **1.000** | **0.882** |
| Cross-encoder, text with headers | 0.972 | 0.936 | 0.700 | 1.000 | 0.972 |

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the question and each candidate *together*, so it judges "does this answer the question" rather than "is this about the same topic". It fixed the last paraphrase misses and lifted overall recall. Its cost is about 150 ms per question on CPU and a ~90 MB model download on first start.

It lowers multi-document coverage (0.9 → 0.8). In m07 ("How do the sales and customer support procedures each use the lead tracker?"), it ranks the sales SOP's list of CRM field names low, because a list of identifiers doesn't read like an answer. Blending the reranker's order with the first-stage order by rank fusion was tested at weights 0.3–1.0. It didn't recover this case and it disabled context trimming, so it was dropped. The reranker stays because its gains on everything else outweigh this one regression.

### 8. Multi-document questions

| | Multi cov. |
|---|---|
| Hybrid + rerank | 0.800 |
| + sub-query decomposition (search each part of "X, and Y?" separately) | 0.800 |
| + at most 2 chunks per document | 0.700 |

What actually improved multi-document retrieval was **keyword search over document titles** (0.6 → 0.9, see 6). Splitting multi-part questions into sub-queries changed nothing: only 3 of the 10 multi-document questions have that shape, and those were already covered. Capping chunks per document hurt, because some questions need two sections of the same document. Neither ships.

Remaining multi-document misses:
- **m02:** the support SOP line "Do not request sensitive identity or payment information" isn't found for a question about "personal data". It shares no keywords with the question and is a weak semantic match.
- **m07:** the CRM field list, see 7.

### 9. Less irrelevant context

The relative reranker cut-off removes chunks that are far below the best one:

| Keep chunks ≥ x × best | Recall | Multi cov. | Chunks | Tokens |
|---|---|---|---|---|
| keep all | 0.982 | 0.800 | 6.0 | 321 |
| **0.001** | 0.961 | 0.800 | **4.6** | **253** |
| 0.01 | 0.918 | 0.600 | 3.7 | 210 |
| 0.05 | 0.918 | 0.600 | 3.1 | 177 |

0.001 cuts context by about a fifth for a small recall cost and no loss of multi-document coverage. Stronger cut-offs start dropping the second document of multi-document answers.

Compared with before, the LLM now receives 15% fewer tokens, of which a third more are evidence. This also comes from no longer sending padding for off-topic questions.

## What improved accuracy, ranked

1. **Keyword search with document titles and headings (hybrid).** It finds exact terms and identifiers that embeddings blur, and matches documents named in the question. This is the largest gain in multi-document coverage.
2. **Cross-encoder reranking.** It judges "does this passage answer the question" directly, which fixed paraphrased questions and raised recall to 0.96–0.98.
3. **A lower, per-question floor instead of a per-chunk 0.25 cut-off.** This stopped answerable questions being rejected. The LLM, which must cite or return the fallback, is the safer judge for borderline cases.
4. **A relative reranker cut-off.** It sends less irrelevant text without depending on the reranker's uncalibrated absolute scores.
5. **Token-based chunk caps at heading boundaries.** These don't change today's numbers, but they prevent silent truncation (past token 256) when longer documents are added. Changing chunk settings now re-indexes automatically on restart.

## What did not help on this corpus

| Tried | Result | Why |
|---|---|---|
| 400–800-token chunks | worse everywhere, 4× more tokens | Windows ≈ whole documents; embeddings truncated at 256 tokens |
| More overlap | no effect | Documents rarely span two windows; headings already bound sections |
| Contextual headers in embeddings | −6 pts recall | Header dominates short chunk vectors |
| Absolute reranker threshold | 8–9 answerable rejected | Reranker gives correct short or paraphrased passages near-zero probability |
| Sub-query decomposition | no change | Few questions have that shape; already covered |
| Per-document chunk cap | −10 pts multi coverage | Some answers need two sections of one document |
| Rank fusion of reranker and first stage | no gain | Didn't recover the CRM field list case and disabled trimming |

These results reflect a small corpus (10 short documents). With much longer documents, larger chunks and overlap matter more, which is why chunk size, overlap, top-k, pool size, reranker and thresholds are all configurable. The harness reproduces every number here.

## Reproduce

```bash
.venv/bin/python scripts/eval_retrieval.py --production            # the app's retriever with current settings
.venv/bin/python scripts/eval_retrieval.py                         # full experiment grid
.venv/bin/python scripts/eval_retrieval.py --only "R2"             # a subset of configurations
.venv/bin/python -m pytest tests/test_retrieval_eval.py -v         # before/after assertions + unit tests
```

Raw numbers: `docs/eval_before.json`, `docs/eval_after.json`, `docs/eval_experiments.json`.
