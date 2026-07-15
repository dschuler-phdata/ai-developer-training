# Setup — Session 2: RAG & Retrieval

## Participants

Complete these steps before the session starts:

1. Clone/pull the repo and open a terminal at its root (if you already did this for Session 1, just `git pull` to pick up this session's changes).
2. Set up the environment (see the root `README.md` for full details):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
   cp .env.example .env   # fill in the credentials provided to you
   ```
   This session adds two new dependencies, `chromadb` and `pypdf` — both are already in `pyproject.toml`, so the same `pip install -e .` picks them up. No separate install step.
3. Launch JupyterLab from the repo root:
   ```bash
   jupyter lab
   ```
4. Open `session-2-rag-retrieval/hol-lab.ipynb` and run the first two cells of Section 1 to confirm you get a response back and the 4 underwriting documents print out. If this fails, flag it before the session starts — don't wait until the room is live.

No external datasets or accounts are required — the underwriting documents used in this lab are synthetic PDFs checked into `session-2-rag-retrieval/docs/`, loaded directly by the notebook.

## Facilitator

In addition to the participant steps above:

- Confirm which provider is active for this session (`PROVIDER` in `.env` — `bedrock` or `azure_openai`) and that credentials/model access are valid for it.
- If using Bedrock: confirm model access is enabled for the embedding model (Titan Embed v2) in addition to the text-generation model, in the target AWS account/region (AWS Bedrock console → Model access), and that `AWS_PROFILE`/SSO login is valid.
- If using Azure OpenAI: confirm `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (new this session) is set and the deployment exists in the target Azure resource, in addition to the existing `AZURE_OPENAI_*` values.
- Do a full run-through of `hol-lab.ipynb` end-to-end before the session, including the exercise cells — this session chains more steps together than Session 1 (chunk → embed → index → retrieve → generate), so a broken step upstream (e.g. embedding dimension mismatch) can cause confusing failures several cells later.
- Chroma's `chroma_client = chromadb.Client()` creates an in-memory collection — restarting the kernel clears it, so if a participant's kernel dies mid-session they'll need to re-run from Section 3 onward, not just the cell that failed.
