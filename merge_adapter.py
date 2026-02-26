"""
Step 6: Merge LoRA adapter into base model for simpler deployment.

Usage:
    python merge_adapter.py
    python merge_adapter.py --push_to_hub "your-username/humanizer-merged"
"""

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ADAPTER_DIR, BASE_MODEL, LOG_DIR, MERGED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "merge.log"),
    ],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--adapter_path", type=str, default=str(ADAPTER_DIR))
    parser.add_argument("--output_dir", type=str, default=str(MERGED_DIR))
    parser.add_argument("--push_to_hub", type=str, default=None, help="HF Hub repo name")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Merging LoRA adapter into base model")
    log.info("=" * 60)
    log.info(f"  Base model:   {args.base_model}")
    log.info(f"  Adapter:      {args.adapter_path}")
    log.info(f"  Output:       {args.output_dir}")

    if not Path(args.adapter_path).exists():
        log.error(f"Adapter not found: {args.adapter_path}")
        return

    # Load base model in full precision for merging
    log.info("\nLoading base model (full precision)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and merge adapter
    log.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    log.info("Merging weights...")
    model = model.merge_and_unload()

    # Save
    log.info(f"Saving merged model → {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        log.info(f"Pushing to Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        log.info("Push complete.")

    log.info(f"\n{'=' * 60}")
    log.info("MERGE COMPLETE")
    log.info(f"  Merged model at: {args.output_dir}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
