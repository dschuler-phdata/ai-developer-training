# Facilitator Guide: Session 2 — RAG & Retrieval

Companion to `hol-lab.ipynb`. Use `hol-plan.md` for the full rationale behind each section; this doc is timing/talking-point focused for running the room.

**Total time: 90 minutes.** Confirm the active provider (Bedrock or Azure OpenAI) and do a full run-through before the session — see `setup.md`.

## Section 1 — Setup & RAG Overview (10 min)
- Talking point: an LLM only knows two things — training data and whatever's in the prompt. RAG closes the gap with retrieval instead of retraining.
- Walk the two-pipeline diagram (offline indexing vs. online query) before anyone writes code — participants should be able to name both pipelines before Section 2 starts.
- What success looks like: everyone sees all 4 document titles print, and has skimmed at least one full document.

## Section 2 — Chunking Documents (15 min)
- Talking point: we can't embed/retrieve a whole document as one unit — chunking is what makes retrieval granular enough to be useful.
- The 📌 note after the Key Takeaway lists other chunking strategies (word-based, sentence-based, recursive, semantic) — mention these exist but aren't implemented today; this session sticks to fixed-size character chunking to keep the concept clear.
- What success looks like: participants can point to a specific chunk where a small chunk size cut off a sentence or number mid-way.

## Section 3 — Embeddings & Vector Indexing (15 min)
- Talking point: an embedding is a fixed-length vector positioned so similar meaning = close distance. You can't read it, but distance between vectors is the whole mechanism behind semantic search.
- Watch for: participants embedding `UNDERWRITING_DOCS` (the raw documents) instead of `ALL_CHUNKS` — this is the single most common mistake in this lab and defeats the point of Section 2. Catch it early.
- What success looks like: `collection.count()` matches `len(ALL_CHUNKS)`, and participants can state the embedding vector's length.

## Section 4 — Retrieval (15 min)
- Talking point: retrieval quality depends on both the embedding model and K — there's no universal "right" K.
- Exercise 4.1's first query is pre-filled (cyber exclusions); have participants also run their own second query — this is where they should start noticing which of the 4 documents "wins" for different phrasings of similar questions.
- What success looks like: participants can describe a case where k=5 pulled in an irrelevant chunk that k=1 didn't.

## Section 5 — Build the Complete RAG Pipeline (20 min)
- **Do not cut this section even if running behind — it's the lab's synthesis moment.**
- Talking point: this is where every prior piece (chunk, embed, retrieve) chains into one pipeline that actually answers a question.
- The no-retrieval / with-retrieval cells are split apart deliberately — have participants read both outputs before moving on, the contrast is the point.
- Reranking discussion (conceptual only): a production system retrieves a larger candidate set and reorders it with a slower, more accurate model before prompting. With only 4 documents there's nothing to rerank here — flag it as "the next lever to pull at scale," not something missing from this lab.
- What success looks like: participants can point to a specific answer that's clearly more specific/accurate with retrieval than without, for both Exercise 5.1 questions.

## Section 6 — Failure Modes & Tuning (10 min)
- Talking point: a bad RAG answer can come from chunking, retrieval, or generation — debugging means checking each stage independently, not just rewriting the final prompt.
- Three pre-built failure demos: irrelevant retrieval (parking policy query), too-small chunks (40-char), hallucination despite context (cyber deductible). Use them as-is; don't over-explain before running them, let participants see the failure first.
- If short on time: this section can become a facilitator-led discussion using the pre-run output instead of a full hands-on exercise.

## Section 7 — Wrap-Up (5 min)
- Recap the full chunk → embed → store → retrieve → augment → generate flow built across the session.
- Preview Session 3: today's retrieval pipeline becomes an input to tool-calling/agent workflows.

## Common Pitfalls
- Participants embedding raw documents instead of chunks in Section 3 (see above) — the single most common mistake in this lab.
- Participants importing `boto3`/`openai` directly for embeddings instead of `get_client().embed()` — same steering as Session 1's rule for `generate()`.
- Chunk size too small in Section 2 carrying forward and producing poor retrieval in Section 4 — a good moment to have participants go back and re-chunk with a larger size.
- Kernel restarts clear the in-memory Chroma collection (`chromadb.Client()`) — participants need to re-run from Section 3 onward, not just the cell that errored.

## Time Overrun Options
- Shorten Section 6 to a discussion instead of hands-on (saves ~5-7 min).
- Section 4's second query exercise can be shortened to one query instead of two if needed.
- Never cut Section 5.
