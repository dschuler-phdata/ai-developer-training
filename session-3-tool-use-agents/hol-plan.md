# HOL Plan: Tool Use, Agents & Orchestration — turning the RAG pipeline into an underwriting agent

## Lab Overview
| Field | Value |
|---|---|
| Client / Event | Specific customer engagement — Great American Insurance Group (GAIG) |
| Industry | Insurance |
| Audience | Software engineers with programming experience, no prior AI-app-building experience (same cohort as Sessions 1–2) |
| Session Length | 90 minutes, instructor-led |
| Platform | Local machine (JupyterLab) |
| Lab Format | notebook |
| Constraints | Concepts over frameworks — no hidden abstractions, every step of the tool-calling loop must be visible in notebook code. All LLM calls (including tool-calling) go through `shared.llm.get_client()` — never a raw `boto3`/`openai` SDK call. This session adds exactly one new capability to that client: `generate_with_tools(...)`, implemented for both providers, normalized to one return shape so notebook code never branches on provider:
  - **Bedrock:** extends the Converse `toolConfig`/`toolUse` pattern `generate_structured` already uses internally for a single forced tool, switched to `toolChoice={"auto": {}}` over multiple `toolSpec` entries, reading `toolUse` blocks out of `response["output"]["message"]["content"]`.
  - **Azure OpenAI:** the same Chat Completions call `generate()` already uses (`client.chat.completions.create`), extended with `tools=[{"type": "function", "function": {"name", "description", "parameters": <JSON schema>}}]` and `tool_choice="auto"`. The response's `message.tool_calls` is a list of `ChatCompletionMessageToolCall` objects (`.id`, `.function.name`, `.function.arguments` as a JSON string to `json.loads`); continuing the loop means appending the assistant's message (with its `tool_calls`) plus one `{"role": "tool", "tool_call_id": ..., "content": ...}` message per call before calling again. This is distinct from `generate_structured`'s `beta.chat.completions.parse(...)`, which forces a single structured output and can't itself choose whether to call a tool — so it isn't reusable as-is for this method. No agent framework (LangChain, LangGraph, AutoGen, CrewAI) is introduced — the tool-calling loop is hand-rolled in the notebook so every decision point stays visible. Toolset is intentionally capped at 3 tools. MCP is a presentation-only topic this session — no MCP demo in the notebook. Chroma stays in-memory (`chromadb.Client()`), matching Session 2 — the same 4 underwriting PDFs are copied into this session's `docs/` folder and the index is rebuilt in a provided Section 1 cell (kernel-restart-wipes-the-collection caveat carries over from Session 2). Session builds toward Session 4 (evaluation — tool selection, retrieval quality, decision quality, groundedness, reliability). |

## Learning Objectives
By the end of this lab, participants will be able to:
- Explain what a tool/function is from an LLM's perspective and why a schema + description matters as much as the implementation
- Expose an existing Python function as an LLM tool (name, description, input schema, structured output)
- Explain how a model decides which tool to call and with what arguments
- Distinguish a deterministic workflow from an agent that chooses its own next action
- Build and run a minimal tool-calling loop, inspecting every tool request and tool result as it happens
- Orchestrate multiple tools across a single reasoning loop to gather the information needed for a decision
- Produce a structured, grounded underwriting appetite decision from the accumulated tool outputs
- Recognize common agent failure modes: wrong tool selected, bad arguments, repeated/uncapped tool calls, hallucinated decisions, ignored evidence

## Lab Environment
- Same environment as Sessions 1–2: local machine, JupyterLab, repo cloned, `.venv` active, `pip install -e .` run from repo root, `.env` populated.
- No new third-party dependencies — `openai>=1.40` and `boto3>=1.34` (already in `pyproject.toml`, and the installed versions are well above these floors) already support tool-calling on both providers; nothing new to install.
- New shared capability this session: `shared.llm.get_client().generate_with_tools(user_message, tools, system_prompt="", max_tokens=1024, messages=None)`, added to `LLMProvider` in `shared/llm/base.py` and implemented in both `BedrockProvider` and `AzureOpenAIProvider` — this must be built and smoke-tested against live Bedrock and Azure **before** the session, not live-coded in the room.
- Underwriting documents: the same 4 synthetic PDFs from Session 2 (`gl-guidelines.pdf`, `property-appetite.pdf`, `cyber-exclusions.pdf`, `workcomp-classification.pdf`), copied into `session-3-tool-use-agents/docs/` so this session's folder is self-contained like Sessions 1 and 2.
- Facilitator: confirm the active `PROVIDER` (bedrock or azure_openai) and re-confirm credentials/model access; additionally dry-run the full agent loop at least once against a fresh kernel to confirm the configured model reliably resolves within the iteration cap.

## Lab Format Notes
Notebook-driven, same structure as Sessions 1–2: each section below is markdown (goal/key takeaway) + code cells + an exercise cell. No presenter-led-only phases — fully hands-on end to end. Section 1 is a fast, mostly-provided rebuild of the Session 2 index (not a new teaching moment) so the rest of the session can focus on tools and the agent loop.

Notebook authoring conventions for Phase 3 (carried forward from Session 1, where they're already established in Section 2 and Section 6):
- Use `### ` subsection headers throughout every section — not just one `## Section N` header per section — so each distinct activity within a section is visually separated (e.g. within Section 2, a subsection per tool; within Section 3, a subsection for the deterministic pipeline and a separate one for the agent loop).
- Whenever a step compares multiple variants of something (Section 2's three tool descriptions; Section 5's before/after tool description or system-instruction rewrite), give each variant its own `### Variant N — ...` subsection header and its own code cell, with the call and its printed output immediately below it in that same cell — mirroring Session 1's `### Variant 1 — No system prompt` / `### Variant 2 — Vague system prompt` / `### Variant 3 — Well-structured system prompt` pattern exactly, rather than combining variants into a single cell or loop.

## Lab Sections

### Section 1 — Recap & rebuild the pipeline
**Goal:** Reconnect to where Session 2 left off and have a working retrieval index ready to wrap in tools.
**Duration:** 8 min
**Format:** notebook
**Key Activities:**
- Recap the Session 2 pipeline (chunk → embed → store → retrieve → augment → generate) and Session 1's structured-output pattern
- Run a provided cell that rebuilds the Chroma collection from this session's `docs/` (same `semantic_chunk` + embedding code from Session 2's Section 2, copied in, not re-derived — the semantic-chunking variant is used here rather than fixed-size, since its meaning-based boundaries produce better retrieval)
- Discuss: of everything built in Session 2, which pieces are things an LLM should be allowed to *decide* to call, vs. things that should just run?
**Participant Action:** Run the index-rebuild cell; confirm collection count matches Session 2's; discuss the framing question as a group.
**Takeaway:** An agent's tools are just the application's existing functions — the RAG pipeline participants already built is 90% of the toolbox.

### Section 2 — Creating tools
**Goal:** Understand what makes a Python function an LLM tool, and test each one in isolation before any agent exists.
**Duration:** 25 min
**Format:** notebook
**Key Activities:**
- Wrap `retrieve()` into `search_documents(query, k=3)`, reshaping raw Chroma output into a clean `list[{source, chunk_index, text, distance}]`
- Wrap Session 1's `SubmissionExtraction` + `generate_structured()` into `extract_submission_information(submission_text)`
- Build `check_underwriting_rules(...)` as a small deterministic function (no LLM call) encoding explicit thresholds from the 4 PDFs — the one tool that isn't RAG-backed, to make clear a "tool" just means a described function, not necessarily another prompt
- For each tool: define name, description, input schema (as a Pydantic model, same mechanism Session 1 already used for structured outputs — `model.model_json_schema()`), and call it directly, unwrapped by any agent, to see its raw return value
**Participant Action:** Implement the `search_documents` output reshaping; implement at least one rule category in `check_underwriting_rules` (e.g. the property flood-zone cap); call all 3 tools directly against sample inputs and inspect the outputs.
**Takeaway:** A good tool description is the interface contract between your code and the model's reasoning — vague descriptions produce vague tool selection later.

### Section 3 — Building the agent
**Goal:** See the same 3 tools used two ways — a fixed pipeline vs. a model deciding — and build the loop that lets the model decide.
**Duration:** 20 min
**Format:** notebook
**Key Activities:**
- Show a 4-line deterministic pipeline calling the 3 tools in fixed order (Extract → Search → Check Rules → Generate) — this *is* the "workflow" half of the workflow-vs-agent objective, shown as code, not just a diagram
- Contrast with the agent loop: call `generate_with_tools()` with all 3 tool specs, inspect the raw tool-call request the model returns, execute the matching Python function, print the raw tool result, append it to the conversation, and loop until the model stops requesting tools or a max-iteration guard trips
- Discuss: when would you want the fixed pipeline instead of the agent? (predictability, cost, latency vs. flexibility)
**Participant Action:** Run the deterministic pipeline version first; then implement the loop's stopping condition (max-iteration guard) and run the agent version on the same submission, comparing tool call order/count to the fixed version.
**Takeaway:** A workflow calls tools in an order *you* chose; an agent calls tools in an order *it* chose — same tools, different control flow, different guarantees.

### Section 4 — Process a complete submission
**Goal:** Run the agent end-to-end on a brand-new, unseen submission and produce a structured underwriting decision.
**Duration:** 20 min
**Format:** notebook
**Key Activities:**
- Feed the agent a new synthetic submission it hasn't seen before
- Watch it autonomously decide to extract facts, search documents, and check rules (in whatever order it chooses) via the loop built in Section 3
- Once the loop stops, assemble the accumulated tool outputs and call `generate_structured()` with a provided `UnderwritingDecision` model (`decision`, `confidence`, `risk_factors`, `missing_information`, `evidence`, `rationale`)
**Participant Action:** Run the full loop on 2+ new submissions; for each, inspect every tool call/result in sequence and the final structured decision; identify which tool output(s) informed which `evidence`/`risk_factors` entries.
**Takeaway:** The final decision is only as grounded as the tool outputs it's built from — every field in the structured decision should trace back to a specific tool call, not a guess.

### Section 5 — Improve the agent & failure modes
**Goal:** See how small changes to tool descriptions, schemas, or instructions change agent behavior, and recognize when the agent gets it wrong.
**Duration:** 10 min
**Format:** notebook
**Key Activities:**
- Trigger and observe at least one failure mode live: a submission with information not covered by any document (hallucination-despite-tools, echoing Session 2's cyber-exclusions demo through the tool lens), or an uncapped loop that keeps calling the same tool
- Rewrite one tool's description or the agent's system instructions and re-run the same submission to see the behavior change
**Participant Action:** Diagnose the triggered failure mode (wrong tool? bad arguments? ignored evidence? missing info?) and identify which pipeline stage is responsible; make one description/instruction edit and confirm it changes the outcome.
**Takeaway:** Agent behavior is steered by descriptions, schemas, and guardrails, not just the underlying model — and every failure mode has a specific, diagnosable cause.

### Section 6 — Wrap-up
**Goal:** Consolidate the tool → agent → decision flow and connect it to what's next.
**Duration:** 7 min
**Format:** notebook
**Key Activities:**
- Recap: tools are existing functions with schemas; an agent is a loop that lets the model choose which to call and when to stop; the final decision is only as good as the tool outputs behind it
- Preview Session 4: evaluating tool selection, retrieval quality, decision quality, groundedness, and reliability — i.e., how do we know today's agent is actually good?
**Participant Action:** Read the wrap-up cell; no new code.
**Takeaway:** Building an agent was the easy part — Session 4 is about proving it works.

## Facilitator Notes
- If short on time: Section 5's description/instruction-rewrite exercise can become a facilitator-led demo instead of a hands-on exercise — do not cut Section 4, it's the lab's synthesis moment (the first full, unseen, end-to-end agent run).
- Common pitfall: participants build `check_underwriting_rules` as another LLM call instead of plain deterministic logic — steer them back; the point of that tool is contrast with the RAG-backed ones.
- Common pitfall: forgetting the max-iteration guard in Section 3, leading to a runaway loop in Section 4 — have the fix ready to share if a group gets stuck.
- Watch for: tool descriptions copy-pasted from docstrings without editing for the model's audience — a good moment to compare a vague vs. specific description live and show the difference in tool selection.
- What success looks like by Section 4: participants can point to a specific tool call/result pair and explain why it appears in the final decision's `evidence` field.
- Pre-session requirement: `generate_with_tools()` must be smoke-tested against live Bedrock and Azure before the room sees it — this is new code, not a reuse of already-proven Session 1/2 paths.
