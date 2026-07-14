# Setup — Session 1: Foundations & Prompt Engineering

## Participants

Complete these steps before the session starts:

1. Clone/pull the repo and open a terminal at its root.
2. Set up the environment (see the root `README.md` for full details):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
   cp .env.example .env   # fill in the credentials provided to you
   ```
3. Launch JupyterLab from the repo root:
   ```bash
   jupyter lab
   ```
4. Open `session-1-foundations-prompt-engineering/hol-lab.ipynb` and run the first cell (Section 1) to confirm you get a response back with token usage printed. If this fails, flag it before the session starts — don't wait until the room is live.

No other datasets or accounts are required — all insurance submission text used in this lab is synthetic and defined directly in the notebook.

## Facilitator

In addition to the participant steps above:

- Confirm which provider is active for this session (`PROVIDER` in `.env` — `bedrock` or `azure_openai`) and that credentials/model access are valid for it. See `hol-plan.md` for the open question on which provider is actually used live.
- If using Bedrock: confirm model access is enabled for the configured model in the target AWS account/region (AWS Bedrock console → Model access), and that `AWS_PROFILE`/SSO login is valid.
- If using Azure OpenAI: confirm the `AZURE_OPENAI_*` values in `.env` are correct and the deployment exists in the target Azure resource.
- Do a full run-through of `hol-lab.ipynb` end-to-end before the session, including the exercise cells, to confirm nothing has broken (e.g., model deprecation, changed pricing/limits).
 
