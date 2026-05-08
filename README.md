# AANA Contract Gate + GPT-4.1-mini τ²-Bench Submission Artifact

This artifact contains the official `tau2 submit prepare` output for a custom
AANA pre-tool-call contract gate wrapped around the τ²-Bench text agent.

Evaluation date: 2026-05-07

Configuration:

- Agent model: `openai/gpt-4.1-mini`
- User simulator: `openai/gpt-4.1-mini`
- Trials: 1 per task
- Domains: `airline`, `retail`, `telecom`, `banking_knowledge`
- Banking retrieval: `bm25`
- Submission type: `custom`

Important limitation: this is an architecture validation run, not a 4-trial
statistical reliability leaderboard claim. The custom AANA scaffold gates tool
calls before execution with accept/ask/defer/refuse decisions; no tasks were
omitted.

Prepared and validated with:

```bash
tau2 submit prepare ...
tau2 submit validate ...
```

Public references:

- AANA demo Space: https://huggingface.co/spaces/mindbomber/aana-demo
- AANA model card: https://huggingface.co/mindbomber/aana
- AANA docs: https://mindbomber.github.io/Alignment-Aware-Neural-Architecture--AANA-/
