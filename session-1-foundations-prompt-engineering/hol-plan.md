# HOL Plan: Prompt Engineering Foundations for Underwriting Data Extraction

## Lab Overview
| Field | Value |
|---|---|
| Client / Event | Great American Insurance Group (specific customer engagement) |
| Industry | Insurance |
| Audience | Software engineers with programming experience, but no prior AI-app-building experience |
| Session Length | 90 minutes, instructor-led |
| Platform | Local machine (JupyterLab) |
| Lab Format | notebook |
| Constraints | Not a Python training — minimize Python boilerplate, emphasize AI/prompt-engineering concepts over syntax. All LLM calls must go through the repo's existing `shared.llm.get_client().generate(user_message, system_prompt, max_tokens) -> GenerateResult(text, usage, model)` helper — no direct `boto3`/`openai`/`anthropic` SDK calls in notebook cells. Attendees run `pip install -e .` from the repo root first, so `import shared` works from inside this session folder. Interactive throughout: attendees write/test/compare/improve prompts themselves, not just run pre-built cells. Session builds toward later sessions (RAG, tools/agents, evals), so extraction output format should stay reusable. |

## Learning Objectives
By the end of this lab, participants will be able to:
- Make a basic LLM API call using provided starter code
- Understand system prompts vs. user prompts
- Apply prompt-engineering best practices
- Extract data from unstructured insurance text
- Compare zero-shot and few-shot prompting
- Generate and validate structured JSON output
- Inspect token usage
- Reduce unnecessary token usage and cost

## Lab Environment
- Platform: local machine, JupyterLab, notebook-driven
- Required tools: Python 3.10+, a virtual environment, `pip install -e .` from the repo root (installs the `shared` helper package and JupyterLab)
- Required services: Amazon Bedrock access (Claude) for the live training; `.env` populated per `.env.example` with `PROVIDER` and provider credentials
- Pre-provisioned resources: none — all insurance submission text used in the lab is synthetic, generated within the notebook's setup section
- Facilitator setup: confirm Bedrock model access is enabled in the target AWS account/region before the session; do a full run-through of the notebook beforehand
- Participant setup: clone/pull the repo, run the setup steps in this folder's `setup.md`, confirm their first API call succeeds before the session starts

## Lab Format Notes
The notebook is the primary participant artifact for the entire 90 minutes. No presenter-led-only phases — the facilitator narrates and demos briefly at the start of each section, then participants work in their own copy of the notebook. Section 8 is unstructured buffer/wrap-up time, not a notebook section.

## Lab Sections

### Section 1 — Setup & First Model Call
**Goal:** Verify the local environment and configuration; make a basic model call; inspect the request, response, and token metadata.
**Duration:** 8 minutes
**Format:** notebook
**Key Activities:**
- Confirm `shared` imports correctly and `.env` is loaded
- Call `get_client().generate(...)` with a simple message and print the response text
- Inspect `GenerateResult.usage` (`input_tokens`, `output_tokens`, `total_tokens`) and `model`
**Participant Action:** Run the provided call, then modify the user message and compare the response.
**Takeaway:** Every LLM call in this training returns the same shape (text + usage + model) regardless of provider — this is the foundation for everything else in the lab.

### Section 2 — Prompt Anatomy: System vs. User
**Goal:** Understand the separation between stable instructions (system) and task-specific input (user); apply the core components of a good prompt (role, clear task, relevant context, constraints/guardrails, examples, expected output format).
**Duration:** 12 minutes
**Format:** notebook
**Key Activities:**
- Run the same user message with no system prompt, a vague system prompt, and a well-structured system prompt
- Observe how behavior changes across the three
**Participant Action:** Compare a vague prompt with a structured prompt; compare with/without a system prompt; change one prompt component at a time and inspect the result.
**Takeaway:** System prompts set stable behavior/role; user prompts carry the task-specific data. Precision in the system prompt reduces variance in output.

### Section 3 — Zero-Shot Extraction from Unstructured Text
**Goal:** Extract underwriting information from a synthetic insurance submission (business name, industry, locations, requested limits, notable risks, missing information); distinguish extraction from summarization; handle missing information without inventing values.
**Duration:** 15 minutes
**Format:** notebook
**Key Activities:**
- Provide 1 synthetic underwriting submission (unstructured text)
- Write a zero-shot extraction prompt for the fields above
- Deliberately include a submission with at least one missing field to surface hallucination risk
**Participant Action:** Write and refine a zero-shot extraction prompt; inspect incorrect, missing, or unsupported values in the output.
**Takeaway:** Zero-shot extraction is a reasonable starting point but is inconsistent on formatting and prone to inventing values for missing fields unless explicitly instructed otherwise.

### Section 4 — Few-Shot Prompting
**Goal:** Understand how examples improve consistency; evaluate whether the improvement justifies the additional tokens.
**Duration:** 10 minutes
**Format:** notebook
**Key Activities:**
- Add 1-2 worked examples (input submission → correctly extracted output, including a missing-field example) to the Section 3 prompt
- Re-run against the same submissions used in Section 3
**Participant Action:** Add example(s) to the extraction prompt; compare zero-shot vs. few-shot outputs on consistency, quality, and token usage.
**Takeaway:** Examples materially improve consistency and formatting, but each example adds input tokens — few-shot is a deliberate cost/quality trade-off, not a free upgrade.

### Section 5 — Structured Outputs
**Goal:** Generate predictable JSON for downstream application code; handle missing values explicitly; parse and validate the response.
**Duration:** 15 minutes
**Format:** notebook
**Key Activities:**
- Define a target JSON schema for the extracted fields, with explicit null-handling for missing information
- Prompt for pure JSON output and parse it with `json.loads`
- Handle/observe malformed or incomplete JSON output
**Participant Action:** Convert the extraction output into the defined JSON structure; validate the returned structure with lightweight Python; improve the prompt after observing malformed or incomplete output.
**Takeaway:** Structured output turns free-text extraction into something application code can reliably consume — but the prompt still has to explicitly define the schema and missing-value behavior; the model won't infer it.

### Section 6 — Token Cost & Prompting Best Practices
**Goal:** Connect prompt quality with token efficiency and cost; understand that longer prompts and more examples are not always better.
**Duration:** 12 minutes
**Format:** notebook
**Key Activities:**
- Take an intentionally verbose/redundant version of the Section 5 prompt
- Shorten it while preserving output quality
- Compare input tokens, output tokens, response quality, and consistency before/after
**Participant Action:** Shorten a verbose prompt; compare trade-offs across token counts and output quality.
**Takeaway:** Best practices reinforced: define the task clearly, provide only relevant context, separate instructions from source data, specify the expected output format, state how to handle missing information, prevent unsupported assumptions, remove duplicated/conflicting instructions, request only the output the application needs, and use few-shot examples only when they materially improve results.

### Section 7 — Final Challenge
**Goal:** Apply everything from the lab to a new, unseen synthetic insurance submission.
**Duration:** 10 minutes
**Format:** notebook
**Key Activities:**
- Provide one new synthetic submission not used earlier in the lab
- Participants build/refine their own prompt from scratch
**Participant Action:** Create or refine a prompt that uses system and user messages appropriately, extracts the required fields, returns valid JSON, identifies missing information, avoids unsupported assumptions, minimizes unnecessary tokens, and follows the best practices taught in the lab.
**Takeaway:** A working checklist for evaluating any extraction prompt going forward, before writing one from scratch.

## Facilitator Notes
- Section 8 (Buffer & Wrap-Up, 8 minutes) is unstructured: use it for local troubleshooting, participant questions, comparing different prompt approaches, reviewing key takeaways, and previewing Session 2 (RAG/retrieval) — no notebook cells needed for this part.
- Confirm Bedrock model access and a successful test call *before* the session — Section 1 depends on this working immediately for every participant.
- If running short on time, Section 6 (token cost) can be shortened to a quick discussion rather than a full hands-on exercise; Section 7 (final challenge) should not be cut, since it's the lab's synthesis moment.
- Common pitfall: participants inventing values for missing fields in Sections 3–5 — use this as a live teaching moment about explicit missing-value handling rather than treating it as a bug.
- Watch for participants importing SDKs directly instead of using `shared.llm.get_client()` — steer them back early since later sessions depend on this pattern.
