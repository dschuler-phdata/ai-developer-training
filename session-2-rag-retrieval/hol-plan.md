# HOL Plan: RAG & Retrieval — building a grounded Q&A pipeline over synthetic underwriting documents

## Lab Overview
| Field | Value |
|---|---|
| Client / Event | Specific customer engagement — Great American Insurance Group (GAIG) |
| Industry | Insurance |
| Audience | Software engineers with programming experience, no prior AI-app-building experience (same cohort as Session 1) |
| Session Length | 90 minutes, instructor-led |
| Platform | Local machine (JupyterLab) |
| Lab Format | notebook |
| Constraints | Concepts over frameworks — no hidden abstractions, every pipeline stage must be visible in notebook code. All LLM text calls go through `shared.llm.get_client().generate(...)`; all embedding calls go through `shared.llm.get_client().embed(...)` — never a raw `boto3`/`openai` SDK call. Vector store is Chroma, used directly (`chromadb.Client()`), not wrapped by LangChain. If LangChain appears at all, it's only for minor glue, never for the chunk/embed/retrieve/augment/generate flow itself. Sample underwriting documents are provided as ready-made data in the notebook (matching Session 1's `SUBMISSIONS` pattern) — no external file loading or dataset download required. Session builds toward Session 3 (tools/agents combined with retrieved context). |

## Learning Objectives
By the end of this lab, participants will be able to:
- Explain the purpose of Retrieval-Augmented Generation (RAG) and why it grounds LLM output in real data
- Distinguish the offline indexing pipeline from the online query pipeline
- Chunk unstructured documents and reason about how chunk size affects retrieval
- Generate embeddings and explain what a vector represents
- Build and inspect a local vector index using Chroma
- Retrieve the most relevant chunks for a user query and compare different Top-K values
- Describe where reranking fits in a production RAG pipeline (conceptually, not implemented)
- Augment a prompt with retrieved context and generate a grounded answer
- Recognize common RAG failure modes: irrelevant retrieval, poor chunking, missing context, hallucination despite retrieval

## Lab Environment
- Same environment as Session 1: local machine, JupyterLab, repo cloned, `.venv` active, `pip install -e .` run from repo root, `.env` populated.
- New dependency this session: `chromadb` (added to `pyproject.toml` — picked up by the same `pip install -e .` participants already ran for Session 1, no separate install step).
- New shared capability this session: `shared.llm.get_client().embed(texts: list[str]) -> list[list[float]]`, implemented for both providers (Bedrock: Titan Embed v2; Azure OpenAI: `text-embedding-3-small` via a new `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` env var) — already built and smoke-tested against live Bedrock (2 texts embedded → 1024-dim vectors; Chroma add/query round-trip confirmed correct nearest-neighbor match).
- No external datasets or accounts required — all underwriting documents are synthetic and defined directly in the notebook, same as Session 1's `SUBMISSIONS`.
- Facilitator: confirm the active `PROVIDER` (bedrock or azure_openai) and, if Azure, that the embedding deployment exists in the target resource before the session.

## Lab Format Notes
Notebook-driven, same structure as Session 1: each of the 7 sections below is a run-through of markdown (goal/key takeaway) + code cells + an exercise cell. No presenter-led-only phases — this session is fully hands-on end to end, building one pipeline incrementally across all 7 sections.

## Lab Sections

### Section 1 — Setup & RAG overview
**Goal:** Understand what RAG is for and see the shape of the full pipeline before building any of it.
**Duration:** 10 min
**Format:** notebook
**Key Activities:**
- Recap: LLMs only know what's in their training data + what's in the prompt — RAG closes that gap with retrieval
- Walk the pipeline diagram: offline (documents → chunk → embed → store) vs. online (query → embed → retrieve → augment → generate)
- Inspect the provided synthetic underwriting documents (3-5 short docs: general liability guidelines, property risk appetite, cyber coverage exclusions, workers' comp classification rules)
**Participant Action:** Run the provided cell that prints each document's title and first few lines; skim the full text of at least one.
**Takeaway:** RAG has two distinct pipelines (offline indexing, online query) and today's lab builds both.

### Section 2 — Chunking documents
**Goal:** Understand why documents must be split into chunks before embedding, and that chunk boundaries matter.
**Duration:** 15 min
**Format:** notebook
**Key Activities:**
- Implement a simple chunking function (fixed-size character or word window, with overlap)
- Run it against the sample documents at 2-3 different chunk sizes
- Inspect chunk boundaries — spot a chunk that awkwardly splits a sentence or loses context
**Participant Action:** Write/modify the chunking function; compare chunk counts and boundary quality across chunk sizes for the same document.
**Takeaway:** Chunk size is a trade-off between retrieval precision, context completeness, and token cost — there's no universally "correct" size.

### Section 3 — Embeddings & vector indexing
**Goal:** Understand what an embedding represents and build a searchable index from the chunks.
**Duration:** 15 min
**Format:** notebook
**Key Activities:**
- Call `get_client().embed([...])` on all chunks from Section 2
- Create a Chroma collection and add the chunks with their embeddings and metadata (source document, chunk index)
- Inspect the collection (count, a sample record) — note that embeddings themselves aren't human-readable, but similarity between them is what matters
**Participant Action:** Embed the full chunk set; build the Chroma collection; print the collection's item count and one raw embedding vector's length.
**Takeaway:** An embedding is a fixed-length numeric vector positioned so that semantically similar text ends up close in vector space — that's what makes semantic search possible.

### Section 4 — Retrieval
**Goal:** Retrieve the most relevant chunks for a real question and see how K affects what comes back.
**Duration:** 15 min
**Format:** notebook
**Key Activities:**
- Embed a user query with the same `embed()` call
- Query the Chroma collection for Top-K nearest chunks
- Compare K=1 vs. K=3 vs. K=5 for the same query — inspect which chunks come back and whether extras help or add noise
**Participant Action:** Try at least 2 different queries against the index; vary K and observe the retrieved chunk text change.
**Takeaway:** Retrieval quality depends on both the embedding model and K — too small misses context, too large dilutes it with irrelevant chunks.

### Section 5 — Build the complete RAG pipeline
**Goal:** Chain every prior step into one working pipeline and see retrieval's actual effect on answer quality.
**Duration:** 20 min
**Format:** notebook
**Key Activities:**
- Assemble the full flow: query → embed → retrieve top-K → build an augmented prompt (retrieved chunks + question) → `get_client().generate(...)`
- Run the same question through the LLM with no retrieved context vs. with retrieved context, side by side
- Discuss (not implement) where a reranking step would slot in between retrieval and prompt-augmentation
**Participant Action:** Implement the augmented-prompt assembly step; run both the no-retrieval and with-retrieval versions on 2+ questions about the underwriting documents.
**Takeaway:** Retrieved context measurably grounds the answer — the LLM answers correctly and specifically instead of guessing or refusing.

### Section 6 — Failure modes & tuning
**Goal:** Recognize that RAG isn't automatically correct — it fails in specific, recognizable ways.
**Duration:** 10 min
**Format:** notebook
**Key Activities:**
- Trigger and observe: a query that retrieves irrelevant chunks (topic not covered by any document), a too-small chunk size that loses needed context, and a case where the LLM hallucinates a plausible-sounding answer despite having relevant context
**Participant Action:** Run the provided failure-mode examples; for each, identify which pipeline stage (chunking, retrieval, or generation) is the likely cause.
**Takeaway:** Debugging a RAG system means checking each stage independently — bad answers can originate from chunking, retrieval, or generation, and the fix is different for each.

### Section 7 — Wrap-up
**Goal:** Consolidate the full pipeline mentally and connect it to what's next.
**Duration:** 5 min
**Format:** notebook
**Key Activities:**
- Recap the full chunk → embed → store → retrieve → augment → generate flow built across the session
- Preview Session 3: the retrieved context from today becomes an input to tool-calling/agent workflows
**Participant Action:** Read the wrap-up cell; no new code.
**Takeaway:** Today's retrieval pipeline is a building block — Session 3 combines it with tools and agent orchestration.

## Facilitator Notes
- If short on time: Section 6 (failure modes) can become a facilitator-led discussion using pre-run output instead of a full hands-on exercise — do not cut Section 5, it's the lab's synthesis moment.
- Common pitfall: participants embedding the raw documents instead of chunks in Section 3 — watch for this, it defeats the purpose of chunking.
- Common pitfall: participants importing `boto3`/`openai` directly for embeddings instead of `get_client().embed()` — same steering as Session 1's rule for `generate()`.
- Watch for: chunk size too small in Section 2 carrying forward and producing poor retrieval in Section 4 — a good moment to have participants go back and re-chunk.
- What success looks like by Section 5: participants can point to a specific answer that's clearly better/more specific with retrieval than without.
