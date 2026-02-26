"""
Step 4: Evaluate the fine-tuned model against the base model.
Compares outputs on held-out validation prompts.

Usage:
    python evaluate.py
    python evaluate.py --num_samples 50
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    GENERATION,
    LOG_DIR,
    OUTPUT_DIR,
    QLORA,
    SPLITS_DIR,
    SYSTEM_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "evaluate.log"),
    ],
)
log = logging.getLogger(__name__)


def load_model_and_tokenizer(base_model: str, adapter_path: str | None = None):
    """Load base model with optional LoRA adapter."""
    compute_dtype = getattr(torch, QLORA["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and Path(adapter_path).exists():
        log.info(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a single response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION["max_new_tokens"],
            temperature=GENERATION["temperature"],
            top_p=GENERATION["top_p"],
            top_k=GENERATION["top_k"],
            repetition_penalty=GENERATION["repetition_penalty"],
            do_sample=GENERATION["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_text_metrics(reference: str, generated: str) -> dict:
    """Simple text quality metrics."""
    ref_words = reference.split()
    gen_words = generated.split()

    # Length ratio
    length_ratio = len(gen_words) / max(len(ref_words), 1)

    # Word overlap (rough content preservation)
    ref_set = set(w.lower() for w in ref_words)
    gen_set = set(w.lower() for w in gen_words)
    overlap = len(ref_set & gen_set) / max(len(ref_set | gen_set), 1)

    # Vocabulary diversity
    vocab_diversity = len(gen_set) / max(len(gen_words), 1)

    # Average sentence length variance (natural writing varies)
    import re
    sentences = re.split(r'[.!?]+', generated)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        sentence_variance = variance ** 0.5
    else:
        sentence_variance = 0.0

    return {
        "length_ratio": round(length_ratio, 3),
        "word_overlap": round(overlap, 3),
        "vocab_diversity": round(vocab_diversity, 3),
        "sentence_length_std": round(sentence_variance, 2),
        "word_count": len(gen_words),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned humanizer model")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--adapter_path", type=str, default=str(ADAPTER_DIR))
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--compare_base", action="store_true", help="Also generate with base model")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP 4: Model Evaluation")
    log.info("=" * 60)

    # Load validation data
    val_path = SPLITS_DIR / "val.jsonl"
    if not val_path.exists():
        log.error(f"Val file not found: {val_path}")
        return

    val_data = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line.strip()))

    samples = val_data[:args.num_samples]
    log.info(f"Evaluating on {len(samples)} samples")

    # Load fine-tuned model
    log.info("\nLoading fine-tuned model...")
    ft_model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)

    results = []
    for i, sample in enumerate(samples):
        log.info(f"\n── Sample {i + 1}/{len(samples)} ──")
        prompt = sample["user"]
        reference = sample["assistant"]

        # Fine-tuned generation
        ft_output = generate_response(ft_model, tokenizer, prompt)
        ft_metrics = compute_text_metrics(reference, ft_output)

        result = {
            "index": i,
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "reference_preview": reference[:200] + "..." if len(reference) > 200 else reference,
            "finetuned_preview": ft_output[:200] + "..." if len(ft_output) > 200 else ft_output,
            "finetuned_metrics": ft_metrics,
        }

        log.info(f"  FT metrics: {ft_metrics}")
        results.append(result)

    # Compute aggregate metrics
    avg_metrics = {}
    metric_keys = results[0]["finetuned_metrics"].keys()
    for key in metric_keys:
        values = [r["finetuned_metrics"][key] for r in results]
        avg_metrics[key] = round(sum(values) / len(values), 3)

    log.info(f"\n{'=' * 60}")
    log.info("AGGREGATE METRICS (fine-tuned model)")
    for k, v in avg_metrics.items():
        log.info(f"  {k}: {v}")

    # Save results
    eval_output = {
        "num_samples": len(samples),
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "aggregate_metrics": avg_metrics,
        "samples": results,
    }

    eval_path = OUTPUT_DIR / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved → {eval_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
