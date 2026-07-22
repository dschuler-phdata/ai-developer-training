# HOL Plan: Evals and Reliability — proving the underwriting agent actually works

## Lab Overview
| Field | Value |
|---|---|
| Client / Event | Specific customer engagement — Great American Insurance Group (GAIG) |
| Industry | Insurance |
| Audience | Software engineers with programming experience, no prior AI-app-building experience (same cohort as Sessions 1–3) |
| Session Length | 90 minutes, instructor-led |
| Platform | Local machine (JupyterLab) |
| Lab Format | notebook |
| Constraints | Reuse the underwriting system built across Sessions 1–3 (prompt-based extraction, RAG/retrieval, structured outputs, tool calling, agent orchestration, appetite decisions) — this session does not rebuild the RAG pipeline or the agent. All LLM calls, including the judge's, go through `shared.llm.get_client()` — no raw SDK calls, and no new provider capability is added this session (unlike Session 3, which added `generate_with_tools`). No external observability framework (LangSmith, Arize, etc.) is introduced or is the focus — observability here means a compact, in-notebook report built from data the agent loop already produces (`tool_call_log`) plus lightweight timing/token capture added around existing calls. No eval framework (Ragas, DeepEval, promptfoo) is introduced — evaluators are plain Python functions and one `generate_structured()` call, consistent with every prior session's "concepts over frameworks" rule. Keep the dataset small (4–6 cases) so the full suite reruns quickly and repeatedly during the 90 minutes. At least one case must be an intentionally failing example so Section 6 has something real to diagnose — do not rely on chance the way Session 3's original Section 5 did (see Session 3 facilitator notes on why an assumed-guaranteed failure is fragile against a capable model — this dataset should hand-pick a case that is judged/checked to fail, not hope one does). |

## Learning Objectives
By the end of this lab, participants will be able to:
- Explain why a fluent LLM answer is not necessarily a correct or reliable one, and why systematic evaluation is required once an agent is in the loop
- Create a small evaluation dataset with explicit expected behavior per case
- Write deterministic evaluation checks (schema validity, field presence/types, decision match, risk-factor/missing-information coverage, tool-usage bounds)
- Explain the trade-off between deterministic checks (reproducible, cheap, fast, easy to debug) and their limits (can't judge subjective qualities like rationale quality or groundedness)
- Implement an LLM-as-a-judge evaluator with a structured-output rubric (correctness, groundedness, completeness, decision quality, missing-information handling)
- Explain the specific limitations of LLM-as-a-judge (prompt sensitivity, model bias, inconsistent scoring, verbosity preference, added latency/cost, same-model-grades-itself risk) rather than treating judge scores as ground truth
- Capture and interpret basic observability data for a run (tokens, latency, tool calls, retrieved sources, errors/retries)
- Read a combined evaluation report to identify a specific failing case and its likely root cause (prompt, retrieval, tool use, agent logic, or output formatting)
- Make one targeted improvement, rerun the suite, and check for regressions elsewhere in the dataset — not just confirm the one target case now passes

## Lab Environment
- Same environment as Sessions 1–3: local machine, JupyterLab, repo cloned, `.venv` active, `pip install -e .` run from repo root, `.env` populated.
- No new third-party dependencies and no new `shared.llm` capability — this session only calls `generate()` / `generate_structured()`, both already implemented and proven in Sessions 1–3. Latency capture uses `time.perf_counter()`; nothing new to install.
- Underwriting system reused as-is: `search_documents`, `extract_submission_information`, `check_underwriting_rules`, `TOOLS`, `run_agent_loop`, `run_full_agent`, and `UnderwritingDecision` are copied in from Session 3 (not re-derived) so this session's folder is self-contained like Sessions 1–3.
- Underwriting documents: the same 4 synthetic PDFs from Session 2 (`gl-guidelines.pdf`, `property-appetite.pdf`, `cyber-exclusions.pdf`, `workcomp-classification.pdf`) — reused via the same relative-path pattern Session 3 used (`../session-2-rag-retrieval/docs/`), no new copies.
- Facilitator: confirm the active `PROVIDER` and credentials as in prior sessions; additionally dry-run the full evaluation suite at least once end-to-end beforehand to confirm the intentionally-failing case actually fails under the live provider/model, and that the judge's scores are reasonably stable across a couple of runs.

## Lab Format Notes
Notebook-driven, same structure as Sessions 1–3: each section below is markdown (goal/key takeaway) + code cells + an exercise cell. No presenter-led-only phases. Section 1 is a fast, mostly-provided rebuild of the Session 3 agent/tools (not a new teaching moment) so the rest of the session can focus on evaluation itself.

Notebook authoring conventions (carried forward from Sessions 1 and 3):
- Use `### ` subsection headers throughout every section, not just one `## Section N` header per section.
- Where a step compares before/after (Section 6's improve-and-rerun), give each state its own `### Variant N — ...` subsection with its own code cell and printed output immediately below, mirroring the Session 1/3 pattern rather than combining into one cell or loop.

## Lab Sections

### Section 1 — Recap & reload the agent
**Goal:** Reconnect to where Session 3 left off with a working agent ready to evaluate.
**Duration:** 5 min
**Format:** notebook
**Key Activities:**
- Recap Sessions 1–3 in one paragraph: extraction → structured output → RAG → tools → agent loop → underwriting decision
- Run a provided cell that reloads the Session 3 system: rebuild the Chroma index, and copy in `search_documents`, `extract_submission_information`, `check_underwriting_rules`, `TOOLS`, `run_agent_loop`, `run_full_agent`, `UnderwritingDecision`
- Reflection: run the agent once on a familiar submission from Session 3 and ask — how do you currently know if this output is *actually* good, beyond "it reads well"?
**Participant Action:** Run the reload cell; confirm the agent still produces a decision on a known submission.
**Takeaway:** A fluent, well-formatted answer is not the same as a correct one — everything built so far has been judged by eye; this session replaces that with something repeatable.

### Section 2 — Define what good looks like
**Goal:** Introduce evaluation as making expected system behavior explicit, and identify what dimensions of an underwriting output are actually worth checking.
**Duration:** 10 min
**Format:** notebook
**Key Activities:**
- Review one sample underwriting output (structured decision + its `tool_call_log`) from Section 1
- As a group, identify what should be evaluated about it: correct decision, valid structured output, correct risk factors, missing information identified, grounded rationale, appropriate use of retrieved evidence, appropriate tool usage, response completeness
- Introduce the 4-layer evaluation model used for the rest of the lab: structural validation → deterministic correctness → model-based quality → operational reliability
**Participant Action:** For the sample output, individually list 3 things that look right and 1 thing that can't be verified just by reading it; compare answers as a group.
**Takeaway:** A fluent answer is not necessarily a correct or reliable one — evaluation makes "good" an explicit, checkable definition instead of a gut feeling.

### Section 3 — Build a small evaluation dataset
**Goal:** Create a small, labeled set of underwriting submissions with explicit expected behavior, small enough to rerun repeatedly during the lab.
**Duration:** 15 min
**Format:** notebook
**Key Activities:**
- Provide 4 pre-built evaluation cases (clear accept, clear decline, refer due to missing information, retrieval-dependent case) with `case_id`, `submission`, `expected_decision`, `expected_risk_factors`, `expected_missing_information`, `notes`
- Leave 2 cases for participants: one ambiguous/conflicting case, and one intentionally-failing compound-rule case (a new submission, not Session 3's crime/theft question — that one was tested twice in Session 3 and the agent handled it correctly both times, so it can't be relied on to fail here; instead, stack two borderline rule conditions together, e.g. flood zone A/V *and* over the $2,000,000 cap *and* no mitigation on file, combined with a second borderline condition like a 21-year-old roof, so the failure mode under test is "did the agent's `risk_factors`/`evidence` catch the full compound rule, not just one half of it" — a mechanical gap a deterministic check can catch reliably regardless of model) — both provided as raw submission text only, with the expected fields left blank
**Participant Action:** Fill in `expected_decision`, `expected_risk_factors`, and `expected_missing_information` for the 2 blank cases by reasoning through the 4 PDFs' rules directly (not by running the agent first).
**Takeaway:** Writing down expected behavior *before* running the system is what makes an eval dataset useful — deciding what's "right" after seeing the output just rationalizes whatever came out.

### Section 4 — Deterministic evaluations
**Goal:** Implement evaluator functions that check objective, rule-based correctness without another LLM call.
**Duration:** 20 min
**Format:** notebook
**Key Activities:**
- Provide `validate_schema(output)` (valid `UnderwritingDecision`, required fields present, types correct, `decision` is one of the allowed values) fully implemented, since it's mechanical Pydantic validation, not the teaching point
- Provide `check_decision(output, expected)` and `check_tool_usage(trace, expected)` as worked examples
- Leave `check_risk_factors(output, expected)` and `check_missing_information(output, expected)` as participant exercises — reasonable fuzzy-matching against the expected lists (e.g. keyword/substring containment), not exact string equality
- Run all 6 evaluation cases through the deterministic checks and inspect a pass/fail table
**Participant Action:** Implement `check_risk_factors` and `check_missing_information`; run the full deterministic suite across all 6 cases and identify which case(s) fail which check(s).
**Takeaway:** Deterministic checks are reproducible, inexpensive, fast, and easy to debug — but they can only judge what you can write a rule for; they say nothing about whether the rationale is actually grounded or well-reasoned.

### Section 5 — LLM as a judge
**Goal:** Build a structured-output LLM judge for qualities that can't be captured with exact-match rules.
**Duration:** 20 min
**Format:** notebook
**Key Activities:**
- Provide an initial, deliberately weak judge prompt/rubric (single vague "rate this 1-5" instruction, no defined dimensions) and a `JudgeScore` Pydantic model (`correctness`, `groundedness`, `completeness`, `decision_quality`, `passed`, `feedback`)
- As a group, run it once and critique it: does the feedback cite actual evidence? does it reward verbosity? would two runs agree?
- Rewrite the rubric with explicit per-dimension criteria and a requirement that `feedback` cite specific tool outputs (mirroring the "every field must trace back to a tool output" discipline from Session 3's decision prompt)
- Run the improved judge across all 6 cases and compare judge results to Section 4's deterministic results side by side
**Participant Action:** Rewrite the rubric/system prompt for the judge so it requires evidence-based feedback and defined scoring criteria per dimension; run it across all 6 cases; note at least one case where the judge and the deterministic checks disagree.
**Takeaway:** LLM-as-a-judge is useful for subjective quality but is not objective ground truth — it's sensitive to prompt wording, can be biased toward verbose answers, may score inconsistently across runs, costs extra tokens/latency, and is weakest when the same model both generates and judges the answer.

### Section 6 — Observability, then improve and rerun
**Goal:** Capture lightweight run metadata, assemble one combined report, diagnose a real failure, fix it, and confirm no regression.
**Duration:** 15 min
**Format:** notebook
**Key Activities:**
- Wrap the agent call with simple timing (`time.perf_counter()`) and pull token counts already returned on `GenerateResult`/`ToolUseResult.usage`; assemble one row per case: case ID, tokens, latency, tool call count, retrieved sources, errors/retries, final decision, deterministic result, judge score, final pass/fail
- Build the combined report as a list of dicts (or a `pandas` DataFrame if participants prefer)
- Run the full suite (Sections 4 + 5 + this section's metadata) across all 6 cases and inspect the report
- Identify the one case built to fail in Section 3; diagnose whether the failure traces to the prompt, retrieval, tool use, agent logic, or output formatting
- Make one targeted change (a prompt tweak, a tool description edit, or an expected-value correction) and rerun the full suite
**Participant Action:** Diagnose the failing case using the report; make one targeted fix; rerun the suite; confirm the target case now passes AND check the other 5 cases still pass (no regression).
**Takeaway:** An evaluation report turns "I think it's working" into "here's the specific case that isn't, and here's what fixed it" — and re-running the whole suite (not just the fixed case) is what catches a fix that broke something else.

### Section 7 — Wrap-up
**Goal:** Consolidate the evaluation flow and connect it to ongoing use.
**Duration:** 5 min
**Format:** notebook
**Key Activities:**
- Recap the flow: dataset → run system → deterministic checks → LLM judge → observability data → combined report → diagnose → fix → rerun
- Discuss: which of these 4 layers (structural, deterministic, model-based, operational) would you run on every commit vs. only occasionally, and why?
**Participant Action:** Read the wrap-up cell; no new code.
**Takeaway:** Building the agent was the easy part (Session 3) — this suite is what lets you know, going forward, whether a change made it better or worse.

## Proposed Timing
| Section | Time |
|---|---|
| 1. Recap & reload the agent | 5 min |
| 2. Define what good looks like | 10 min |
| 3. Build a small evaluation dataset | 15 min |
| 4. Deterministic evaluations | 20 min |
| 5. LLM as a judge | 20 min |
| 6. Observability, then improve and rerun | 15 min |
| 7. Wrap-up | 5 min |
| **Total** | **90 min** |

## Facilitator Notes
- If short on time: Section 5's rubric rewrite can become a facilitator-led demo (show the weak version, show the improved version) rather than fully hands-on — do not cut Section 6, it's the lab's synthesis moment (the only point where participants diagnose and fix a real failure).
- Common pitfall: participants writing `check_risk_factors`/`check_missing_information` as exact list equality — steer them toward substring/keyword containment, since the agent's exact wording will never match the expected list verbatim.
- Common pitfall: treating the judge's `passed` field as authoritative over the deterministic checks — use any case where they disagree (Section 5) as the live teaching moment for "judge scores are signals, not ground truth."
- Watch for: participants using the same weak judge rubric from the start of Section 5 for the Section 6 rerun — make sure the improved rubric from Section 5 carries forward.
- Pre-session requirement: confirm the intentionally-failing case in the dataset actually fails reliably under the live provider/model before the session — same lesson learned in Session 3 about not assuming a failure mode without verifying it against the model actually in use.
- What success looks like by Section 6: participants can point to one specific report row, say why it failed, name the fix they made, and show the rerun report confirming the fix without a new failure appearing elsewhere.

## Assumptions / Open Questions
- Assumes the same PROVIDER/model configuration as Session 3 is still in place; if the model changes between sessions, re-verify the intentionally-failing case still fails.
- Confirmed: participants are running this on a different day than Session 3, with no prior notebook/kernel state carried over. Section 1 is not a fallback for this — it's the only path: the Chroma index rebuild and all reused functions (`search_documents`, `extract_submission_information`, `check_underwriting_rules`, `run_agent_loop`, `run_full_agent`) are copied into *this* session's notebook as fresh, runnable code, exactly as Session 3 did with Session 2's chunking code rather than assuming Session 2's notebook was still open.
- Resolved: the intentionally-failing 6th case uses a new compound-rule submission, not Session 3's crime/theft question — see Section 3's Key Activities for the reasoning (that case didn't reliably fail in Session 3 testing, so it can't be trusted to fail here either).
