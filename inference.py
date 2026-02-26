"""
Step 5: Inference — Generate humanized text using the fine-tuned model.

Usage:
    python inference.py --text "Your text to humanize here"
    python inference.py --file input.txt --output output.txt
    python inference.py --interactive
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ADAPTER_DIR, BASE_MODEL, GENERATION, QLORA, SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Global model cache
_model = None
_tokenizer = None


def load_model(base_model: str = BASE_MODEL, adapter_path: str = str(ADAPTER_DIR)):
    """Load the fine-tuned model (cached after first call)."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    log.info(f"Loading base model: {base_model}")
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

    if Path(adapter_path).exists():
        log.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        log.warning(f"Adapter not found at {adapter_path}. Using base model only.")

    model.eval()
    _model = model
    _tokenizer = tokenizer
    log.info("Model loaded successfully.")
    return model, tokenizer


def humanize(
    text: str,
    model=None,
    tokenizer=None,
    temperature: float = None,
    max_new_tokens: int = None,
) -> str:
    """Humanize a single text input."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite the following text in a natural, human-written style:\n\n{text}"},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nRewrite the following text in a natural, human-written style:\n\n{text}\n<|assistant|>\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens or GENERATION["max_new_tokens"],
        "temperature": temperature or GENERATION["temperature"],
        "top_p": GENERATION["top_p"],
        "top_k": GENERATION["top_k"],
        "repetition_penalty": GENERATION["repetition_penalty"],
        "do_sample": GENERATION["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive_mode():
    """Interactive REPL for humanizing text."""
    model, tokenizer = load_model()
    print("\n" + "=" * 60)
    print("HUMANIZER — Interactive Mode")
    print("Type your text and press Enter. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            text = input("Input > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not text:
            continue

        print("\nHumanizing...")
        result = humanize(text, model, tokenizer)
        print(f"\nOutput > {result}\n")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Humanize text using fine-tuned model")
    parser.add_argument("--text", type=str, help="Text to humanize")
    parser.add_argument("--file", type=str, help="Input file (one text per line or single text)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--adapter_path", type=str, default=str(ADAPTER_DIR))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if args.text:
        model, tokenizer = load_model(args.base_model, args.adapter_path)
        result = humanize(args.text, model, tokenizer, args.temperature, args.max_tokens)
        print(result)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            log.info(f"Saved → {args.output}")

    elif args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            log.error(f"File not found: {args.file}")
            return

        model, tokenizer = load_model(args.base_model, args.adapter_path)
        content = input_path.read_text(encoding="utf-8").strip()

        # If file has multiple lines separated by double newlines, process each
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        if len(blocks) <= 1:
            blocks = [content]

        results = []
        for i, block in enumerate(blocks):
            log.info(f"Processing block {i + 1}/{len(blocks)}...")
            result = humanize(block, model, tokenizer, args.temperature, args.max_tokens)
            results.append(result)

        output_text = "\n\n".join(results)

        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            log.info(f"Saved {len(results)} blocks → {args.output}")
        else:
            print(output_text)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
