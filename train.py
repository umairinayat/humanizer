"""
Step 3: QLoRA fine-tuning with PEFT on the prepared SFT dataset.

Usage:
    python train.py
    python train.py --base_model "unsloth/Llama-3.2-3B-Instruct" --epochs 2
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    CHECKPOINT_DIR,
    LOG_DIR,
    LORA,
    QLORA,
    SPLITS_DIR,
    SYSTEM_PROMPT,
    TRAINING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train.log"),
    ],
)
log = logging.getLogger(__name__)


def load_sft_dataset(path: Path) -> Dataset:
    """Load JSONL SFT dataset into HF Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            records.append(row)
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> str:
    """Format a single example into the chat template."""
    messages = [
        {"role": "system", "content": example.get("system", SYSTEM_PROMPT)},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    # Use the model's chat template if available
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback: simple concat
        return (
            f"<|system|>\n{messages[0]['content']}\n"
            f"<|user|>\n{messages[1]['content']}\n"
            f"<|assistant|>\n{messages[2]['content']}"
        )


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for humanizer")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=TRAINING["num_train_epochs"])
    parser.add_argument("--lr", type=float, default=TRAINING["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=TRAINING["per_device_train_batch_size"])
    parser.add_argument("--max_seq_length", type=int, default=TRAINING["max_seq_length"])
    parser.add_argument("--lora_r", type=int, default=LORA["r"])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"
    else:
        report_to = TRAINING["report_to"]

    log.info("=" * 60)
    log.info("STEP 3: QLoRA Fine-Tuning")
    log.info("=" * 60)
    log.info(f"  Base model:     {args.base_model}")
    log.info(f"  Epochs:         {args.epochs}")
    log.info(f"  LR:             {args.lr}")
    log.info(f"  Batch size:     {args.batch_size}")
    log.info(f"  Max seq length: {args.max_seq_length}")
    log.info(f"  LoRA rank:      {args.lora_r}")
    log.info(f"  Report to:      {report_to}")

    # ── Load datasets ─────────────────────────────────────────────────
    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"

    if not train_path.exists():
        log.error(f"Train file not found: {train_path}. Run prepare_data.py first.")
        return

    train_ds = load_sft_dataset(train_path)
    val_ds = load_sft_dataset(val_path) if val_path.exists() else None

    log.info(f"  Train samples:  {len(train_ds):,}")
    if val_ds:
        log.info(f"  Val samples:    {len(val_ds):,}")

    # ── Quantization config ───────────────────────────────────────────
    compute_dtype = getattr(torch, QLORA["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QLORA["load_in_4bit"],
        bnb_4bit_quant_type=QLORA["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=QLORA["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # ── Load model ────────────────────────────────────────────────────
    log.info(f"\nLoading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # ── Load tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Prepare model for QLoRA ───────────────────────────────────────
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
    log.info(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Format dataset ────────────────────────────────────────────────
    def formatting_func(examples):
        output = []
        for i in range(len(examples["user"])):
            example = {
                "system": examples["system"][i] if "system" in examples else SYSTEM_PROMPT,
                "user": examples["user"][i],
                "assistant": examples["assistant"][i],
            }
            output.append(format_chat(example, tokenizer))
        return output

    # ── Training arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
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
        logging_dir=str(LOG_DIR / "tensorboard"),
        logging_steps=TRAINING["logging_steps"],
        eval_strategy=TRAINING["eval_strategy"] if val_ds else "no",
        eval_steps=TRAINING["eval_steps"] if val_ds else None,
        save_strategy=TRAINING["save_strategy"],
        save_steps=TRAINING["save_steps"],
        save_total_limit=TRAINING["save_total_limit"],
        load_best_model_at_end=TRAINING["load_best_model_at_end"] if val_ds else False,
        metric_for_best_model=TRAINING["metric_for_best_model"] if val_ds else None,
        greater_is_better=TRAINING["greater_is_better"] if val_ds else None,
        report_to=report_to,
        seed=TRAINING["seed"],
        dataloader_num_workers=TRAINING["dataloader_num_workers"],
        group_by_length=TRAINING["group_by_length"],
        gradient_checkpointing=TRAINING["gradient_checkpointing"],
        optim=TRAINING["optim"],
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # ── Train ─────────────────────────────────────────────────────────
    log.info("\n── Starting Training ──")
    if args.resume_from:
        log.info(f"  Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────
    log.info(f"\nSaving adapter → {ADAPTER_DIR}")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # Save training config for reproducibility
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
    }
    config_path = ADAPTER_DIR / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    log.info(f"Run config saved → {config_path}")

    # ── Final eval ────────────────────────────────────────────────────
    if val_ds:
        log.info("\n── Final Evaluation ──")
        metrics = trainer.evaluate()
        log.info(f"  Final eval loss: {metrics.get('eval_loss', 'N/A'):.4f}")
        metrics_path = ADAPTER_DIR / "final_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    log.info(f"\n{'=' * 60}")
    log.info("TRAINING COMPLETE")
    log.info(f"  Adapter saved:  {ADAPTER_DIR}")
    log.info(f"  Checkpoints:    {CHECKPOINT_DIR}")
    log.info(f"  Logs:           {LOG_DIR}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
