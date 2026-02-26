# Context — Humanizer Project

## Purpose

This project fine-tunes a language model on **human-written text** so it learns to produce natural, fluent, human-like output. The approach uses QLoRA (4-bit quantized LoRA) for parameter-efficient training on a single GPU.

## Why QLoRA Over Full Fine-Tuning

| Factor | QLoRA | Full Fine-Tuning |
|--------|-------|-----------------|
| VRAM needed | ~10–16 GB | 40–80+ GB |
| Trainable params | ~0.5–2% | 100% |
| Training time | Hours | Days |
| Output artifact | Small adapter (~50–200 MB) | Full model (6–14 GB) |
| Quality | Very good for style/tone | Marginally better |
| Risk of catastrophic forgetting | Low | Higher |

**Decision:** QLoRA is the right default. It trains on 1 GPU, produces a small deployable adapter, and gives strong style adaptation without destroying the base model's general capabilities.

## Why Llama 3.2 3B

- **Size:** 3B parameters fits comfortably in 24GB VRAM when 4-bit quantized
- **Quality:** Strong instruction-following baseline for its size
- **License:** Permissive (Llama 3.2 Community License)
- **Ecosystem:** Wide support in transformers, peft, trl, vLLM, Ollama
- **Alternatives considered:**
  - Mistral 7B — better quality but 2x VRAM, slower training
  - Qwen 2.5 3B — comparable, can be swapped in `config.py`
  - Phi-3 Mini — good but less tested with QLoRA pipelines

## Data Strategy

### Source Selection

Five datasets were chosen to maximize **diversity** and **volume** of human-written text:

1. **RAID** (`liamdugan/raid`) — Academic/research human text alongside AI-generated, well-labeled
2. **AI-Human Detection v1** (`silentone0725/ai-human-text-detection-v1`) — Binary labeled detection dataset
3. **Human-AI Generated Text** (`dmitva/human_ai_generated_text`) — Mixed source, string-labeled
4. **AI and Human Text** (`NabeelShar/ai_and_human_text`) — Binary labeled (0=human, 1=AI)
5. **Wikipedia Human-Written** (`badhanr/wikipedia_human_written_text`) — Pure human, encyclopedic style

### Why Only Human Text

The model trains exclusively on human-written text because:
- The goal is to learn **human writing patterns** (varied sentence length, natural transitions, authentic voice)
- Including AI text would teach the model AI patterns — the opposite of what we want
- Human text provides the ground truth for the "rewrite naturally" task

### Cleaning Rationale

| Rule | Why |
|------|-----|
| Min 50 chars | Removes fragments that teach nothing |
| Max 8000 chars | Prevents memory issues; most training value is in shorter texts |
| Min 10 words | Ensures actual sentences, not metadata |
| Remove URLs/emails | Noise that doesn't help style learning |
| Exact dedup (MD5) | Identical texts waste training compute |
| Near-dedup (Jaccard > 0.85) | Paraphrases add marginal value but risk overfitting |

## SFT Format Design

Each training example:
```
system: "You are a skilled writer. Rewrite the given text so it sounds natural..."
user:   "Rewrite the following text in a natural, human-written style:\n\n{text}"
assistant: {text}
```

**Why this format:**
- The human text IS the ideal output — the model learns that "rewriting naturally" means producing text like this
- The system prompt establishes the task context
- At inference time, you feed AI-generated text as input, and the model rewrites it in the human style it learned

## Hyperparameter Reasoning

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| LoRA r=32 | Medium rank | Balances capacity vs. efficiency; r=16 is too low for style, r=64 is overkill |
| Alpha=16 | Half of r | Standard ratio; effective learning rate scaling = alpha/r = 0.5 |
| Dropout=0.05 | Light | Prevents overfitting without hurting convergence |
| LR=1.5e-4 | Moderate | Sweet spot for QLoRA SFT; 1e-4 is safe, 2e-4 is aggressive |
| Cosine scheduler | Standard | Smooth decay, works well with 1–2 epoch runs |
| Warmup=3% | Brief | Enough to stabilize early training without wasting steps |
| Epochs=2 | Conservative | 1 epoch may underfit, 3+ risks overfitting on style |
| Effective batch=16 | 4 × 4 grad accum | Good balance of gradient quality and VRAM |
| Seq length=1024 | Moderate | Covers most training texts; longer wastes VRAM |

## Failure Modes

### Overfitting Indicators
- **Eval loss increases** while train loss decreases (classic)
- **Outputs become repetitive** — same phrases, sentence starters
- **Verbosity explosion** — model adds unnecessary filler
- **Memorization** — outputs copy training data verbatim

### Mitigations
- Early stopping via `load_best_model_at_end`
- Checkpoint saving every 500 steps (rollback possible)
- Dropout in LoRA layers
- Diverse training data from multiple sources
- Limited epochs (2 max initially)

### Underfitting Indicators
- Eval loss doesn't decrease meaningfully
- Outputs are indistinguishable from base model

### Mitigations
- Increase epochs to 3
- Increase LoRA rank to 64
- Increase learning rate to 2e-4
- Add more training data

## Deployment Options

### Option A: Adapter-Only (Recommended)
- Save: ~50–200 MB adapter files
- Load: Base model + adapter at inference time
- Pros: Small, versioned, can swap adapters
- Cons: Requires base model download at runtime

### Option B: Merged Model
- Save: Full merged weights (~6 GB for 3B)
- Load: Single model, no adapter loading
- Pros: Simpler deployment, works with any inference server
- Cons: Larger artifact, harder to version/rollback

### Rollback Strategy
1. Keep all checkpoint adapters (`save_total_limit=3`)
2. Tag adapter versions with eval metrics
3. If new adapter regresses, load previous checkpoint
4. Base model is never modified — worst case, remove adapter entirely

## Monitoring Plan

| What | Tool | When |
|------|------|------|
| Training loss | Tensorboard / WandB | Every 25 steps |
| Eval loss | Tensorboard / WandB | Every 200 steps |
| GPU utilization | `nvidia-smi` | During training |
| Output quality | Manual review | After training |
| Repetition rate | Eval script | After training |
| Content preservation | Eval script (word overlap) | After training |

## Experiment Checklist

- [ ] Objective defined (style adaptation, not factual improvement)
- [ ] Data licenses verified (all HF datasets are open)
- [ ] Train/val split created (95/5)
- [ ] Dedup report reviewed
- [ ] Baseline eval scores captured
- [ ] QLoRA config documented (`run_config.json`)
- [ ] Adapter saved and verified
- [ ] Post-training eval completed
- [ ] Output samples manually reviewed
- [ ] Deployment decision made (adapter vs merged)
- [ ] Rollback plan tested
