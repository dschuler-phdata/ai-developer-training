# Facilitator Guide: Session 1 — Foundations & Prompt Engineering

Companion to `hol-lab.ipynb`. Use `hol-plan.md` for the full rationale behind each section; this doc is timing/talking-point focused for running the room.

**Total time: 90 minutes.** Confirm the active provider (Bedrock or Azure OpenAI — see `hol-plan.md`'s open question) and do a full run-through before the session.

## Section 1 — Setup & First Model Call (8 min)
- Talking point: every call in this notebook goes through one function, `get_client().generate(...)` — this is deliberate, so participants focus on prompts, not SDKs.
- Watch for: import errors mean `pip install -e .` wasn't run from the repo root, or the venv isn't activated — this is the most common failure point, fix it before moving on.
- What success looks like: everyone sees a response printed with non-zero token counts.

## Section 2 — Prompt Anatomy: System vs. User (12 min)
- Talking point: system = stable role/instructions, user = task-specific input that changes every call.
- Highlight the three outputs live — the difference between "no system prompt" and "structured system prompt" is usually dramatic and a good discussion moment.
- Prompt injection reflection: don't need a deep security tangent, just plant the idea that user input can try to override instructions.
- What success looks like: participants' custom system prompt actually refuses the off-topic message.

## Section 3 — Zero-Shot Extraction (15 min)
- Talking point: extraction ≠ summarization — we want structured facts, not a paraphrase.
- Pause on `SUB-004` specifically — if someone's zero-shot prompt invents a limit for the missing field, use it as a live teaching moment (see Common Pitfalls below), not a bug to apologize for.
- What success looks like: participants notice inconsistent formatting and/or invented values, motivating Section 4.

## Section 4 — Few-Shot Prompting (10 min)
- Talking point: few-shot is a deliberate trade — better consistency, more input tokens. Ask the room whether the token increase seems "worth it" before revealing the actual numbers.
- What success looks like: participants can point to specific consistency improvements between Section 3 and 4 outputs, and state the input_tokens difference.

## Section 5 — Structured Outputs (15 min)
- Talking point: this is what makes extraction usable by downstream code — a human can read messy text, but application code needs a schema.
- Watch for: `json.JSONDecodeError` — some participants' prompts will produce malformed JSON (extra prose before/after, trailing commas). This is expected and is the point of Exercise 5.1.
- What success looks like: all 5 submissions parse cleanly with all required keys present.

## Section 6 — Token Cost & Best Practices (12 min)
- Talking point: bigger ≠ better. The `verbose_system` example is intentionally bad — let participants feel how little the extra verbosity buys.
- If short on time: this section can become a quick discussion (show the before/after numbers, skip having everyone edit) rather than a full hands-on exercise.
- What success looks like: participants can shorten the prompt and get equal or better output quality with fewer input tokens.

## Section 7 — Final Challenge (10 min)
- Do not cut this section even if running behind — it's the lab's synthesis moment.
- Talking point: this is the checklist they should carry forward: system/user separation, explicit schema, explicit missing-value handling, no unsupported assumptions, token discipline.
- What success looks like: participant's prompt on `SUB_006` returns valid JSON and correctly flags at least one missing field (submission is missing a requested coverage limit).

## Section 8 — Buffer & Wrap-Up (8 min)
- No notebook cells for this — open floor for troubleshooting, comparing approaches, Q&A.
- Close by previewing Session 2 (RAG/retrieval) using the notebook's final wrap-up cell as a jumping-off point.

## Common Pitfalls
- Participants inventing values for missing fields (Sections 3–5) — treat as a teaching moment about explicit missing-value instructions, not a failure.
- Participants importing `boto3`/`openai`/`anthropic` directly instead of using `shared.llm.get_client()` — steer them back early; later sessions depend on this pattern holding.
- Malformed JSON in Section 5 — expected, don't over-explain it upfront; let Exercise 5.1 surface it naturally.

## Time Overrun Options
- Shorten Section 6 to a discussion instead of hands-on (saves ~5-7 min).
- Section 2's custom-system-prompt exercise can be shortened to one test case instead of on-topic + off-topic if needed.
- Never cut Section 7.
