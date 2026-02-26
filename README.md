# Humanizer — QLoRA Fine-Tuning Pipeline

A production-ready pipeline for fine-tuning a language model on **human-written text** using **QLoRA (4-bit NF4)**. The trained model learns to produce natural, fluent, human-like writing.

## Architecture

- **Method:** QLoRA (Parameter-Efficient Fine-Tuning)
- **Base Model:** `unsloth/Llama-3.2-3B-Instruct`
- **Quantization:** 4-bit NF4, double quantization, BF16 compute
- **LoRA:** r=32, alpha=16, dropout=0.05
- **Targets:** All attention + MLP projections
- **Format:** Instruction-style SFT (system → user → assistant)

## Datasets

Human-written text is extracted from 5 HuggingFace datasets:

| Dataset | Filter |
|---------|--------|
| [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid) | `model == "human"` |
| [silentone0725/ai-human-text-detection-v1](https://huggingface.co/datasets/silentone0725/ai-human-text-detection-v1) | `label == 0` |
| [dmitva/human_ai_generated_text](https://huggingface.co/datasets/dmitva/human_ai_generated_text) | `source == "human"` |
| [NabeelShar/ai_and_human_text](https://huggingface.co/datasets/NabeelShar/ai_and_human_text) | `Label == 0` |
| [badhanr/wikipedia_human_written_text](https://huggingface.co/datasets/badhanr/wikipedia_human_written_text) | All rows (pure human) |

## Project Structure

```
humanizer/
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
├── download_datasets.py      # Step 1: Download human text from HF
├── prepare_data.py           # Step 2: Clean, dedup, create SFT splits
├── train.py                  # Step 3: QLoRA fine-tuning
├── evaluate.py               # Step 4: Evaluate fine-tuned model
├── inference.py              # Step 5: Humanize text (CLI/interactive)
├── merge_adapter.py          # Step 6: Merge adapter into base model
├── run_pipeline.py           # Run all steps end-to-end
├── CLAUDE.md                 # Agent instructions
├── context.md                # Project context & design decisions
├── data/
│   ├── raw/                  # Downloaded human texts (per dataset)
│   ├── processed/            # Cleaned unified corpus
│   └── splits/               # train.jsonl + val.jsonl
├── outputs/
│   ├── checkpoints/          # Training checkpoints
│   ├── adapter/              # Saved LoRA adapter
│   └── merged/               # Merged full model (optional)
└── logs/                     # Training logs + tensorboard
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python run_pipeline.py
```

Or run steps individually:

```bash
python download_datasets.py        # Download human text
python prepare_data.py             # Clean + dedup + split
python train.py --no_wandb         # Train (disable wandb logging)
python evaluate.py                 # Evaluate
```

### 3. Humanize Text

```bash
# Single text
python inference.py --text "Your text to humanize here"

# From file
python inference.py --file input.txt --output humanized.txt

# Interactive mode
python inference.py --interactive
```

### 4. Merge Adapter (optional)

```bash
python merge_adapter.py
```

## Training Configuration

| Parameter | Default |
|-----------|---------|
| Epochs | 2 |
| Batch size | 4 (effective 16 with grad accum) |
| Learning rate | 1.5e-4 |
| LR scheduler | Cosine |
| Warmup | 3% of steps |
| Max seq length | 1024 |
| Optimizer | Paged AdamW 8-bit |
| Gradient checkpointing | Enabled |
| Eval strategy | Every 200 steps |

Override via CLI:

```bash
python train.py --epochs 3 --lr 2e-4 --batch_size 2 --lora_r 64
```

## Hardware Requirements

- **Training:** 1x GPU with 24GB VRAM (RTX 3090/4090, A5000, etc.)
- **Inference:** 1x GPU with 8GB+ VRAM (4-bit quantized)
- **CPU-only:** Not supported for training; inference possible but slow

## Data Pipeline

1. **Download** — Pulls human text from 5 HF datasets
2. **Clean** — Removes URLs, emails, normalizes whitespace, filters by length (50–8000 chars)
3. **Deduplicate** — Exact (MD5 hash) + near-duplicate (Jaccard similarity > 0.85)
4. **Format** — Creates SFT pairs: `user: "Rewrite naturally: {text}"` → `assistant: {text}`
5. **Split** — 95% train / 5% validation

## Evaluation Metrics

- **Length ratio** — Generated vs reference word count
- **Word overlap** — Content preservation (Jaccard on word sets)
- **Vocabulary diversity** — Unique words / total words
- **Sentence length std** — Natural writing has varied sentence lengths

## License

This project is for legitimate model adaptation (domain tone, task quality, style consistency). It is **not** intended for bypassing AI detectors or misrepresenting authorship.
