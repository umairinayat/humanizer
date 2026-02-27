"""
Unified Pipeline: Preprocessing -> Training (single script)

Performs ALL steps in one go:
  1. Download human text from HuggingFace datasets
  2. Clean, deduplicate, create SFT splits
  3. QLoRA fine-tuning

Usage:
    python full_pipeline.py                          # Full pipeline
    python full_pipeline.py --skip_download          # Skip download (use existing raw data)
    python full_pipeline.py --skip_prepare           # Skip prepare (use existing splits)
    python full_pipeline.py --no_wandb               # Disable wandb
    python full_pipeline.py --epochs 3 --lr 2e-4     # Override training params
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    CHECKPOINT_DIR,
    DATA,
    DATASETS,
    LOG_DIR,
    LORA,
    PROCESSED_DIR,
    QLORA,
    RAW_DIR,
    SPLITS_DIR,
    SYSTEM_PROMPT,
    TRAINING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "full_pipeline.log"),
    ],
)
log = logging.getLogger("full_pipeline")

# ── HF Token Login ─────────────────────────────────────────────────────
def setup_hf_token():
    """Load HF token from .env file and login."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

    token = os.environ.get("HG_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            log.info("Logged in to HuggingFace Hub successfully.")
        except Exception as e:
            log.warning("HF login failed: %s", e)
    else:
        log.warning("No HF token found in .env. Some datasets may require authentication.")

setup_hf_token()

SEP = "=" * 60


def banner(text: str):
    log.info("\n" + SEP)
    log.info(text)
    log.info(SEP + "\n")


# ════════════════════════════════════════════════════════════════════════
#  STEP 1: DOWNLOAD
# ════════════════════════════════════════════════════════════════════════

def extract_human_texts(ds_config: dict) -> list[str]:
    repo = ds_config["repo"]
    text_col = ds_config["text_col"]
    label_col = ds_config["label_col"]
    human_value = ds_config["human_value"]
    subset = ds_config["subset"]
    split = ds_config["split"]

    log.info("Loading %s (split=%s, subset=%s)...", repo, split, subset)

    try:
        if subset:
            ds = load_dataset(repo, subset, split=split)
        else:
            ds = load_dataset(repo, split=split)
    except Exception as e:
        log.warning("Failed to load split='%s' for %s: %s", split, repo, e)
        try:
            ds = load_dataset(repo)
            available_splits = list(ds.keys())
            log.info("Available splits: %s", available_splits)
            ds = ds[available_splits[0]]
        except Exception as e2:
            log.error("Failed to load %s: %s", repo, e2)
            return []

    log.info("  Loaded %d rows. Columns: %s", len(ds), ds.column_names)

    if text_col not in ds.column_names:
        candidates = ["text", "content", "generation", "Text", "article", "document"]
        found = [c for c in candidates if c in ds.column_names]
        if found:
            text_col = found[0]
            log.warning("  Column '%s' not found. Using '%s'.", ds_config["text_col"], text_col)
        else:
            log.error("  No text column found in %s. Skipping.", ds.column_names)
            return []

    texts = []
    if label_col is None:
        for row in tqdm(ds, desc="  Extracting [" + ds_config["name"] + "]"):
            t = row.get(text_col, "")
            if t and isinstance(t, str) and len(t.strip()) > 0:
                texts.append(t.strip())
    else:
        if label_col not in ds.column_names:
            col_map = {c.lower(): c for c in ds.column_names}
            if label_col.lower() in col_map:
                label_col = col_map[label_col.lower()]
            else:
                log.error("  Label column '%s' not found. Skipping.", label_col)
                return []

        for row in tqdm(ds, desc="  Extracting [" + ds_config["name"] + "]"):
            label = row.get(label_col)
            if isinstance(human_value, str):
                is_human = str(label).lower().strip() == human_value.lower().strip()
            else:
                try:
                    is_human = int(label) == human_value
                except (ValueError, TypeError):
                    is_human = label == human_value
            if is_human:
                t = row.get(text_col, "")
                if t and isinstance(t, str) and len(t.strip()) > 0:
                    texts.append(t.strip())

    log.info("  Extracted %d human texts from %s", len(texts), ds_config["name"])
    return texts


def save_raw(name: str, texts: list[str], output_dir: Path) -> Path:
    out_path = output_dir / (name + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text, "source": name}, f, ensure_ascii=False)
            f.write("\n")
    log.info("  Saved %d texts -> %s", len(texts), out_path)
    return out_path


def step_download():
    banner("STEP 1/3: Download & extract human text from HF datasets")

    total = 0
    for ds_config in DATASETS:
        log.info("\n--- Dataset: %s (%s) ---", ds_config["name"], ds_config["repo"])
        texts = extract_human_texts(ds_config)
        if texts:
            save_raw(ds_config["name"], texts, RAW_DIR)
            total += len(texts)
        else:
            log.warning("  No human texts extracted from %s", ds_config["name"])

    log.info("\nTOTAL HUMAN TEXTS EXTRACTED: %s", f"{total:,}")
    return total


# ════════════════════════════════════════════════════════════════════════
#  STEP 2: CLEAN, DEDUP, PREPARE
# ════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^\w\s.,!?;:'\"\\()\[\]{}/\\@#$%&*+=<>~`\n-]", "", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def is_valid_text(text: str) -> bool:
    if not text or len(text) < DATA["min_text_length"] or len(text) > DATA["max_text_length"]:
        return False
    if len(text.split()) < 10:
        return False
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return alnum / len(text) >= 0.5


def compute_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def ngram_set(text: str, n: int = 5) -> set:
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate(texts: list[dict], threshold: float) -> list[dict]:
    log.info("  Deduplicating %d texts (threshold=%s)...", len(texts), threshold)

    seen_hashes = set()
    exact_deduped = []
    for item in texts:
        h = compute_hash(item["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            exact_deduped.append(item)
    removed = len(texts) - len(exact_deduped)
    log.info("  After exact dedup: %d (removed %d)", len(exact_deduped), removed)

    if len(exact_deduped) > 50000:
        log.info("  Dataset > 50k: using hash-based dedup only (near-dedup too slow)")
        return exact_deduped

    log.info("  Running near-duplicate detection...")
    ngrams_cache = []
    kept = []
    for item in tqdm(exact_deduped, desc="  Near-dedup"):
        item_ngrams = ngram_set(item["text"])
        is_dup = False
        for existing_ngrams in ngrams_cache[-500:]:
            if jaccard_similarity(item_ngrams, existing_ngrams) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            ngrams_cache.append(item_ngrams)

    removed2 = len(exact_deduped) - len(kept)
    log.info("  After near-dedup: %d (removed %d)", len(kept), removed2)
    return kept


def create_sft_pairs(texts: list[dict]) -> list[dict]:
    sft_data = []
    for item in texts:
        text = item["text"]
        sft_data.append({
            "system": SYSTEM_PROMPT,
            "user": "Rewrite the following text in a natural, human-written style:\n\n" + text,
            "assistant": text,
            "source": item.get("source", "unknown"),
        })
    return sft_data


def step_prepare():
    banner("STEP 2/3: Clean, deduplicate, and prepare SFT dataset")

    all_texts = []
    raw_files = list(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        log.error("No raw JSONL files found in %s. Run without --skip_download first.", RAW_DIR)
        sys.exit(1)

    for raw_file in raw_files:
        count = 0
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    all_texts.append(row)
                    count += 1
                except json.JSONDecodeError:
                    continue
        log.info("  Loaded %s texts from %s", f"{count:,}", raw_file.name)

    log.info("\nTotal raw texts: %s", f"{len(all_texts):,}")

    # Clean
    log.info("\n-- Cleaning --")
    cleaned = []
    for item in tqdm(all_texts, desc="Cleaning"):
        text = clean_text(item["text"])
        if is_valid_text(text):
            cleaned.append({"text": text, "source": item.get("source", "unknown")})
    removed = len(all_texts) - len(cleaned)
    log.info("After cleaning: %s (removed %s)", f"{len(cleaned):,}", f"{removed:,}")

    # Deduplicate
    log.info("\n-- Deduplication --")
    deduped = deduplicate(cleaned, DATA["dedup_threshold"])
    log.info("After dedup: %s", f"{len(deduped):,}")

    # Shuffle
    random.seed(DATA["seed"])
    random.shuffle(deduped)

    # Save cleaned corpus
    cleaned_path = PROCESSED_DIR / "human_texts_clean.jsonl"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for item in deduped:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    log.info("Cleaned corpus saved -> %s", cleaned_path)

    # Create SFT pairs
    log.info("\n-- Creating SFT Pairs --")
    sft_data = create_sft_pairs(deduped)
    log.info("Created %s SFT training examples", f"{len(sft_data):,}")

    # Train/val split
    val_size = int(len(sft_data) * DATA["val_split"])
    train_data = sft_data[val_size:]
    val_data = sft_data[:val_size]
    log.info("Train: %s  |  Val: %s", f"{len(train_data):,}", f"{len(val_data):,}")

    # Save splits
    for path, data in [(SPLITS_DIR / "train.jsonl", train_data), (SPLITS_DIR / "val.jsonl", val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        log.info("Saved -> %s", path)

    log.info("\nDATA PREPARATION COMPLETE -- Train: %s | Val: %s",
             f"{len(train_data):,}", f"{len(val_data):,}")


# ════════════════════════════════════════════════════════════════════════
#  STEP 3: TRAIN (QLoRA)
# ════════════════════════════════════════════════════════════════════════

def load_sft_dataset(path: Path) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": example.get("system", SYSTEM_PROMPT)},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: simple concat
        sys_c = messages[0]["content"]
        usr_c = messages[1]["content"]
        ast_c = messages[2]["content"]
        tag_sys = "<" + "|system|" + ">"
        tag_usr = "<" + "|user|" + ">"
        tag_ast = "<" + "|assistant|" + ">"
        return tag_sys + "\n" + sys_c + "\n" + tag_usr + "\n" + usr_c + "\n" + tag_ast + "\n" + ast_c


def step_train(args):
    banner("STEP 3/3: QLoRA Fine-Tuning")

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"
    else:
        report_to = TRAINING["report_to"]

    log.info("  Base model:     %s", args.base_model)
    log.info("  Epochs:         %d", args.epochs)
    log.info("  LR:             %s", args.lr)
    log.info("  Batch size:     %d", args.batch_size)
    log.info("  Max seq length: %d", args.max_seq_length)
    log.info("  LoRA rank:      %d", args.lora_r)
    log.info("  Report to:      %s", report_to)

    # Load datasets
    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"

    if not train_path.exists():
        log.error("Train file not found: %s. Run without --skip_prepare first.", train_path)
        sys.exit(1)

    train_ds = load_sft_dataset(train_path)
    val_ds = load_sft_dataset(val_path) if val_path.exists() else None

    # Subsample if requested
    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        log.info("  Subsampling train: %s -> %s", f"{len(train_ds):,}", f"{args.max_train_samples:,}")
        train_ds = train_ds.shuffle(seed=TRAINING["seed"]).select(range(args.max_train_samples))

    log.info("  Train samples:  %s", f"{len(train_ds):,}")
    if val_ds:
        log.info("  Val samples:    %s", f"{len(val_ds):,}")

    # Quantization config
    compute_dtype = getattr(torch, QLORA["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QLORA["load_in_4bit"],
        bnb_4bit_quant_type=QLORA["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=QLORA["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load model
    log.info("\nLoading model: %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare model for QLoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=LORA["lora_alpha"],
        lora_dropout=LORA["lora_dropout"],
        target_modules=LORA["target_modules"],
        bias=LORA["bias"],
        task_type=LORA["task_type"],
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    log.info("  Trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", pct)

    # Format dataset
    def formatting_func(example):
        chat_example = {
            "system": example.get("system", SYSTEM_PROMPT),
            "user": example["user"],
            "assistant": example["assistant"],
        }
        return format_chat(chat_example, tokenizer)

    # Training arguments

    training_args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=TRAINING["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING["gradient_accumulation_steps"],
        learning_rate=args.lr,
        lr_scheduler_type=TRAINING["lr_scheduler_type"],
        warmup_ratio=TRAINING["warmup_ratio"],
        weight_decay=TRAINING["weight_decay"],
        max_grad_norm=TRAINING["max_grad_norm"],
        fp16=TRAINING["fp16"],
        bf16=TRAINING["bf16"],
        logging_steps=TRAINING["logging_steps"],
        eval_strategy=TRAINING["eval_strategy"] if val_ds else "no",
        eval_steps=TRAINING["eval_steps"] if val_ds else None,
        save_strategy=TRAINING["save_strategy"],
        save_steps=TRAINING["eval_steps"],  # must be multiple of eval_steps
        save_total_limit=TRAINING["save_total_limit"],
        load_best_model_at_end=TRAINING["load_best_model_at_end"] if val_ds else False,
        metric_for_best_model=TRAINING["metric_for_best_model"] if val_ds else None,
        greater_is_better=TRAINING["greater_is_better"] if val_ds else None,
        report_to=report_to,
        seed=TRAINING["seed"],
        dataloader_num_workers=TRAINING["dataloader_num_workers"],
        gradient_checkpointing=TRAINING["gradient_checkpointing"],
        optim=TRAINING["optim"],
        remove_unused_columns=False,
        # SFT-specific
        max_length=args.max_seq_length,
        packing=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Train
    log.info("\n-- Starting Training --")
    start_time = time.time()
    if args.resume_from:
        log.info("  Resuming from checkpoint: %s", args.resume_from)
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    elapsed = time.time() - start_time
    log.info("Training completed in %.1f minutes", elapsed / 60)

    # Save adapter
    log.info("\nSaving adapter -> %s", ADAPTER_DIR)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # Save training config
    run_config = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": LORA["lora_alpha"],
        "lora_dropout": LORA["lora_dropout"],
        "target_modules": LORA["target_modules"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "qlora_config": QLORA,
        "training_time_minutes": round(elapsed / 60, 1),
    }
    config_path = ADAPTER_DIR / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    log.info("Run config saved -> %s", config_path)

    # Final eval
    if val_ds:
        log.info("\n-- Final Evaluation --")
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss", "N/A")
        log.info("  Final eval loss: %.4f", eval_loss if isinstance(eval_loss, float) else 0.0)
        metrics_path = ADAPTER_DIR / "final_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    banner("FULL PIPELINE COMPLETE")
    log.info("  Adapter saved:  %s", ADAPTER_DIR)
    log.info("  Checkpoints:    %s", CHECKPOINT_DIR)
    log.info("  Logs:           %s", LOG_DIR)


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Humanizer: full pipeline (download -> prepare -> train)")
    parser.add_argument("--skip_download", action="store_true", help="Skip dataset download step")
    parser.add_argument("--skip_prepare", action="store_true", help="Skip data preparation step")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=TRAINING["num_train_epochs"])
    parser.add_argument("--lr", type=float, default=TRAINING["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=TRAINING["per_device_train_batch_size"])
    parser.add_argument("--max_seq_length", type=int, default=TRAINING["max_seq_length"])
    parser.add_argument("--lora_r", type=int, default=LORA["r"])
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (subsample)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # Automatically create all required directories
    for directory in [DATA_DIR, RAW_DIR, SPLITS_DIR, CHECKPOINT_DIR, ADAPTER_DIR, LOG_DIR, LOG_DIR / "tensorboard"]:
        directory.mkdir(parents=True, exist_ok=True)

    banner("HUMANIZER UNIFIED PIPELINE")
    steps = []
    if not args.skip_download:
        steps.append("download")
    if not args.skip_prepare:
        steps.append("prepare")
    steps.append("train")
    log.info("Steps to run: %s", " -> ".join(steps))

    pipeline_start = time.time()

    # Step 1: Download
    if not args.skip_download:
        step_download()
    else:
        log.info("Skipping download step (--skip_download)")

    # Step 2: Prepare
    if not args.skip_prepare:
        step_prepare()
    else:
        log.info("Skipping prepare step (--skip_prepare)")

    # Step 3: Train
    step_train(args)

    total_time = time.time() - pipeline_start
    log.info("\nTotal pipeline time: %.1f minutes", total_time / 60)


if __name__ == "__main__":
    main()
