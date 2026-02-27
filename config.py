"""
Central configuration for the humanizer QLoRA fine-tuning pipeline.
All paths, hyperparameters, and dataset specs in one place.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
ADAPTER_DIR = OUTPUT_DIR / "adapter"
MERGED_DIR = OUTPUT_DIR / "merged"
LOG_DIR = BASE_DIR / "logs"

# Create dirs
for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, CHECKPOINT_DIR, ADAPTER_DIR, MERGED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"  # Good 3B model for QLoRA
# Alternatives: "mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-3B-Instruct"

# ── QLoRA Settings ─────────────────────────────────────────────────────
QLORA = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}

# ── LoRA Adapter Settings ─────────────────────────────────────────────
LORA = {
    "r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# ── Training Hyperparameters ──────────────────────────────────────────
TRAINING = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,  # effective batch = 16
    "learning_rate": 1.5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_seq_length": 1024,
    "fp16": False,
    "bf16": True,
    "logging_steps": 25,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "wandb",
    "seed": 42,
    "dataloader_num_workers": 2,
    "group_by_length": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
}

# ── Data Settings ─────────────────────────────────────────────────────
DATA = {
    "val_split": 0.05,
    "min_text_length": 50,      # chars — skip very short texts
    "max_text_length": 8000,    # chars — skip excessively long texts
    "dedup_threshold": 0.85,    # Jaccard similarity for near-dedup
    "seed": 42,
}

# ── Dataset Sources (HuggingFace) ─────────────────────────────────────
DATASETS = [
    {
        "name": "raid",
        "repo": "liamdugan/raid",
        "text_col": "generation",
        "label_col": "model",
        "human_value": "human",
        "subset": None,
        "split": "train",
    },
    {
        "name": "ai_human_detection_v1",
        "repo": "silentone0725/ai-human-text-detection-v1",
        "text_col": "text",
        "label_col": "label",
        "human_value": "human",  # labels are strings: "human" / "ai"
        "subset": None,
        "split": "train",
    },
    {
        "name": "human_ai_generated",
        "repo": "dmitva/human_ai_generated_text",
        "text_col": "human_text",  # actual column name is human_text
        "label_col": None,  # all rows contain human text in human_text column
        "human_value": None,
        "subset": None,
        "split": "train",
    },
    {
        "name": "ai_and_human_text",
        "repo": "NabeelShar/ai_and_human_text",
        "text_col": "text",  # lowercase in actual dataset
        "label_col": "generated",  # actual column: 0 = human, 1 = AI
        "human_value": 0,
        "subset": None,
        "split": "train",
    },
    {
        "name": "wikipedia_human",
        "repo": "badhanr/wikipedia_human_written_text",
        "text_col": "text",
        "label_col": None,  # all human
        "human_value": None,
        "subset": None,
        "split": "train",
    },
]

# ── System Prompt for SFT ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a skilled writer. Rewrite the given text so it sounds natural, "
    "fluent, and human-written. Preserve the original meaning, key details, "
    "and structure. Use varied sentence lengths, natural transitions, and "
    "an authentic voice. Avoid robotic phrasing, excessive formality, or "
    "repetitive patterns."
)

# ── Inference Defaults ────────────────────────────────────────────────
GENERATION = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}
