# AI Developer Training

Welcome! This repository contains the hands-on labs for our four-session AI
Developer training. Each session is a 90-minute, instructor-led lab you'll
work through in a Jupyter notebook on your own machine.

| Session | Date | Topic |
|---|---|---|
| 1 | 7/21 | Foundations & prompt engineering |
| 2 | 7/23 | RAG / retrieval |
| 3 | 7/28 | Tool use, agents, orchestration |
| 4 | 7/30 | Evals / reliability |

The sessions build on each other, so hang on to your work from each lab —
later sessions reuse it.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
cp .env.example .env   # fill in the API credentials provided to you
jupyter lab
```

This opens JupyterLab in your browser. Navigate into the folder for the
current session (e.g. `session-1-foundations-prompt-engineering/`) and open
`hol-lab.ipynb` to begin.

Each session folder also includes `setup.md` with any session-specific setup
steps.

## Repository layout

- `shared/` — helper code used across all four labs (for calling the LLM,
  formatting output, etc.), so each notebook can stay focused on the
  concepts being taught rather than boilerplate.
- `session-*/` — one folder per lab, containing that session's notebook and
  setup instructions.
