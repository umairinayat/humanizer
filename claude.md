# PEFT via LoRA / QLoRA (recommended default for most teams)

This document is a **team-friendly playbook** for adapting a large language model (LLM) to your domain or writing style using **parameter‑efficient fine‑tuning (PEFT)** — specifically **LoRA** and **QLoRA**.

It’s written so a junior engineer (or an LLM “coding agent”) can follow it end‑to‑end and make good decisions.

> **Allowed use only:** This is for legitimate model adaptation (domain tone, task quality, brand voice, formatting).  
> It is **not** intended to help “bypass” AI detectors or misrepresent authorship.

---

## Executive summary

**Choose QLoRA by default** when:
- You want **good quality** without training full weights.
- You want to train on **one GPU** (or a small cluster).
- You want a deployable result as a small **adapter**.

**Choose LoRA (non‑quantized base)** when:
- You already have enough VRAM (or want slightly simpler serving).
- You need maximum stability for certain kernels/hardware.

**Choose full fine‑tuning** only when:
- You have very large, high‑quality data and budget.
- You need deep behavior changes that PEFT can’t reach.

---

## What you need (non‑negotiables)

### Data
For **text** adaptation (not code):
- Prefer **high‑quality human‑written** text (best default).
- If you include model‑generated text, keep it **small** and **curated** (avoid “training on your own outputs” loops).
- Ensure licensing/permissions are compatible with your use case.

### Evaluation
Define success with at least **one** objective measure:
- Task accuracy (QA, extraction, classification)
- Preference or rubric scoring (human eval)
- Style adherence (brand guidelines checklist)
- Safety/quality checks (toxicity, hallucinations, policy)

### Compute
QLoRA lets you train 3B–8B models on a single consumer GPU in many cases, but:
- Batch size, sequence length, and optimizer settings determine actual VRAM needs.
- Longer context ≈ more VRAM.

---

## When to use LoRA vs QLoRA vs RAG (decision guide)

### If you want…
**A stable “voice” / writing style shift**
- ✅ **QLoRA/LoRA**
- Optional: light RAG for facts, examples, references

**Up‑to‑date knowledge from private docs**
- ✅ **RAG first** (no training needed)
- Add LoRA only if you also need consistent tone/format

**Domain skill improvement (legal drafting, support replies, medical summaries)**
- ✅ **QLoRA/LoRA** on domain examples
- Optional: RAG for current policies and product updates

**Frequent content updates**
- ✅ **RAG** (update index, not model)
- Optionally combine with LoRA for tone

---

## Data recipe for text fine‑tuning (recommended)

### Target format: instruction (SFT) examples
Use **prompt → response** pairs where the response is the *ideal* assistant output.

Examples you want:
- Real requests from your domain
- Clear, correct, complete answers
- Good tone & formatting

Avoid:
- Duplicates / near‑duplicates
- Very short “OK thanks” type responses (unless needed)
- Unverified factual content
- Private data you can’t ship

### Recommended dataset structure (JSONL)
Each line is one training example:

- `system`: high‑level behavior rules (optional)
- `user`: the prompt
- `assistant`: the ideal response

Example fields:
- `{"system":"…","user":"…","assistant":"…"}`

**Tip:** Keep “system” consistent for a run; variation belongs in the user prompt.

---

## QLoRA configuration defaults (good starting point)

These are typical starting values for 3B models (adjust as needed):

### Base settings
- Quantization: **4‑bit** (NF4), double quantization on
- Compute dtype: BF16 if available, otherwise FP16
- Optimizer: paged AdamW 8‑bit (or similar memory‑efficient optimizer)

### LoRA adapter settings
- Rank (**r**): 16–64
- Alpha: 16–32
- Dropout: 0.0–0.1
- Target modules: attention projections + MLP projections (varies by architecture)

### Training hyperparameters
- LR: 1e‑4 to 2e‑4 for many SFT runs
- Warmup: 1–5% of steps
- Epochs: 1–3 (start with 1 and evaluate)
- Max seq length: 512–2048 depending on your data and VRAM
- Gradient accumulation: increase if you need effective batch size without VRAM blowups

---

## Practical workflow (step‑by‑step)

### Step 1 — Clarify the goal
Write down:
- What *exactly* should improve? (tone, formatting, domain correctness, refusal style)
- What should **not** change? (general ability, safety boundaries)
- What evaluation proves you succeeded?

### Step 2 — Assemble & clean data
- Remove duplicates
- Remove low‑quality, ambiguous, or incorrect samples
- Normalize formatting (e.g., consistent bullet style)
- Split into train / validation (e.g., 95/5)

### Step 3 — Train with QLoRA
- Start small (e.g., 10k–50k samples) to validate the pipeline
- Evaluate on held‑out prompts
- Scale up only after you see real gains

### Step 4 — Save outputs
Prefer saving:
- **Adapter only** (small, easy to share)
Optionally:
- **Merged** weights (larger, easier deployment but heavier storage)

### Step 5 — Evaluate & iterate
- Compare before/after on your rubric
- Identify failure modes and add targeted training examples
- Avoid over‑training (repetition, verbosity, loss of diversity)

---

## Common failure modes and fixes

### “Outputs became generic / repetitive”
- Data too homogeneous
- LR too high or too many epochs
- Not enough varied prompts
**Fix:** diversify prompts, reduce epochs, lower LR, add style variety.

### “Model follows style but hallucinated facts”
- Style tuning doesn’t teach truth
**Fix:** add RAG for factual grounding; include more fact‑checked examples; add refusal patterns.

### “It memorizes training text”
- Data too small or too unique
**Fix:** more data, stronger dedup, reduce epochs, add regularization (dropout).

---

## Recommended best practice: LoRA + light RAG

A common production setup:
- **QLoRA adapter** for tone/format/domain behavior
- **RAG** for current facts/policies/product data
- **Guardrails** (policy, refusal rules, eval gates)

This balances quality, cost, and maintainability.

---

## Deliverables checklist (share with your team)

- [ ] Clear objective + rubric
- [ ] Data license verified
- [ ] Train/val split and dedup report
- [ ] Baseline eval scores
- [ ] QLoRA run config captured
- [ ] Adapter artifacts saved
- [ ] After‑tune eval scores
- [ ] Deployment plan (adapter vs merged)
- [ ] Monitoring + rollback plan

---

## Glossary (short)

- **PEFT**: training only a small number of parameters instead of all weights
- **LoRA**: low‑rank adapter matrices added to a frozen base model
- **QLoRA**: LoRA on a 4‑bit quantized base model
- **SFT**: supervised fine‑tuning on prompt/response examples
- **RAG**: retrieve documents at inference time to ground responses

