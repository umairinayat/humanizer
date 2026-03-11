"""
Quick sanity-check: does the checkpoint actually humanize text?

Tests three cases and prints a comparison table:
  1. Base model only (no adapter) — baseline behavior
  2. Old checkpoint (outputs_old_identity_bug) — was trained with identity bug
  3. New checkpoint (outputs/checkpoints/latest) — trained with fixed data
     (only run if a checkpoint exists there)

Usage:
    conda run -n humanizer python test_checkpoint.py
    conda run -n humanizer python test_checkpoint.py --checkpoint outputs_old_identity_bug/checkpoints/checkpoint-5400
"""

import argparse
import os
import textwrap
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import BASE_MODEL, GENERATION, QLORA, SYSTEM_PROMPT

# ── Test inputs (typical AI-generated sounding texts) ───────────────────────
TEST_CASES = [
    (
        "Short sentence",
        "The implementation of advanced technological solutions facilitates the "
        "optimization of operational efficiency across all organizational units.",
    ),
    (
        "Medium paragraph",
        "In conclusion, it is evident that the utilization of renewable energy "
        "sources presents numerous advantages. Firstly, it significantly reduces "
        "carbon emissions. Secondly, it ensures energy sustainability for future "
        "generations. Furthermore, it promotes economic growth through job creation "
        "in the green energy sector.",
    ),
    (
        "Technical text",
        "The algorithm operates by iteratively processing each data point within "
        "the dataset, applying a series of transformations to normalize the input "
        "values. Subsequently, the processed data is fed into the neural network "
        "architecture, which generates probabilistic outputs based on the learned "
        "weight parameters.",
    ),
]


def word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets — 1.0 = identical, 0.0 = no overlap."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def load_model(adapter_path: str | None):
    compute_dtype = getattr(torch, QLORA["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print(f"\nLoading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("No adapter — using base model only.")

    model.eval()
    return model, tokenizer


def humanize(text: str, model, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite the following text in a natural, human-written style:\n\n{text}"},
    ]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\nRewrite the following text in a natural, human-written style:\n\n{text}\n"
            f"<|assistant|>\n"
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_tests(adapter_path: str | None, label: str):
    print("\n" + "=" * 70)
    print(f"  TEST: {label}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")
    else:
        print("  Adapter: None (base model)")
    print("=" * 70)

    model, tokenizer = load_model(adapter_path)

    for name, input_text in TEST_CASES:
        output = humanize(input_text, model, tokenizer)
        similarity = word_overlap(input_text, output)
        changed = similarity < 0.70

        print(f"\n{'─'*70}")
        print(f"[{name}]  word-overlap={similarity:.2f}  "
              f"{'✓ CHANGED' if changed else '✗ IDENTITY (no change)'}")
        print(f"\nINPUT:\n{textwrap.fill(input_text, 70)}")
        print(f"\nOUTPUT:\n{textwrap.fill(output, 70)}")

    # Free GPU memory before potentially loading next model
    del model
    torch.cuda.empty_cache()


def find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific adapter/checkpoint path to test. If omitted, runs all three cases.",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Test base model only (no adapter)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.checkpoint:
        run_tests(args.checkpoint, f"Custom checkpoint: {args.checkpoint}")
        return

    if args.base_only:
        run_tests(None, "Base model (no adapter)")
        return

    # ── Case 1: Base model baseline ──────────────────────────────────────────
    run_tests(None, "Base model (no adapter) — baseline")

    # ── Case 2: Old checkpoint (trained with identity bug) ───────────────────
    old_ckpt = find_latest_checkpoint(base_dir / "outputs_old_identity_bug" / "checkpoints")
    if old_ckpt:
        run_tests(old_ckpt, f"OLD checkpoint (identity bug era): {Path(old_ckpt).name}")
    else:
        print("\n[SKIP] No old checkpoints found in outputs_old_identity_bug/")

    # ── Case 3: New checkpoint (fixed data) ──────────────────────────────────
    new_ckpt = find_latest_checkpoint(base_dir / "outputs" / "checkpoints")
    if new_ckpt:
        run_tests(new_ckpt, f"NEW checkpoint (fixed data): {Path(new_ckpt).name}")
    else:
        print("\n[SKIP] No new checkpoints yet in outputs/checkpoints/ — training still running.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("  word-overlap < 0.70  → model is actively transforming the text  ✓")
    print("  word-overlap ≥ 0.70  → model is acting as identity function     ✗")
    print("=" * 70)


if __name__ == "__main__":
    main()
