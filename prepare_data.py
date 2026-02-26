"""
Step 2: Clean, deduplicate, and prepare the unified training dataset.
Reads raw JSONL files → cleans → deduplicates → creates SFT format → splits.

Usage:
    python prepare_data.py
"""

import hashlib
import json
import logging
import random
import re
from pathlib import Path

from tqdm import tqdm

from config import DATA, PROCESSED_DIR, RAW_DIR, SPLITS_DIR, SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/prepare.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Text Cleaning ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning without destroying meaning."""
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove excessive special characters
    text = re.sub(r"[^\w\s.,!?;:'\"\-(){}[\]/\\@#$%&*+=<>~`\n]", "", text)

    # Collapse double spaces left after removals
    text = re.sub(r"  +", " ", text)

    return text.strip()


def is_valid_text(text: str) -> bool:
    """Check if text meets quality thresholds."""
    if not text:
        return False
    if len(text) < DATA["min_text_length"]:
        return False
    if len(text) > DATA["max_text_length"]:
        return False

    # Must contain actual words (not just numbers/symbols)
    word_count = len(text.split())
    if word_count < 10:
        return False

    # Reject if >50% non-alphanumeric
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    if alnum / len(text) < 0.5:
        return False

    return True


# ── Deduplication ─────────────────────────────────────────────────────

def compute_hash(text: str) -> str:
    """MD5 hash of normalized text for exact dedup."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def ngram_set(text: str, n: int = 5) -> set:
    """Character n-gram set for near-dedup via Jaccard similarity."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate(texts: list[dict], threshold: float) -> list[dict]:
    """Exact + near-duplicate removal."""
    log.info(f"  Deduplicating {len(texts)} texts (threshold={threshold})...")

    # Exact dedup
    seen_hashes = set()
    exact_deduped = []
    for item in texts:
        h = compute_hash(item["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            exact_deduped.append(item)
    log.info(f"  After exact dedup: {len(exact_deduped)} (removed {len(texts) - len(exact_deduped)})")

    # Near dedup — sample-based for large datasets to keep it tractable
    if len(exact_deduped) > 50000:
        log.info("  Dataset > 50k: using hash-based dedup only (near-dedup too slow)")
        return exact_deduped

    log.info("  Running near-duplicate detection...")
    ngrams_cache = []
    kept = []
    for item in tqdm(exact_deduped, desc="  Near-dedup"):
        item_ngrams = ngram_set(item["text"])
        is_dup = False
        for existing_ngrams in ngrams_cache[-500:]:  # compare against recent 500
            if jaccard_similarity(item_ngrams, existing_ngrams) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            ngrams_cache.append(item_ngrams)

    log.info(f"  After near-dedup: {len(kept)} (removed {len(exact_deduped) - len(kept)})")
    return kept


# ── SFT Formatting ────────────────────────────────────────────────────

def create_sft_pairs(texts: list[dict]) -> list[dict]:
    """
    Create instruction-style SFT pairs for humanization training.

    Format:
    - system: writing instruction
    - user: "Rewrite this text naturally: {text}"
    - assistant: {text}  (the human text IS the target)

    The model learns to produce human-like text when asked to rewrite.
    """
    sft_data = []
    for item in texts:
        text = item["text"]
        sft_example = {
            "system": SYSTEM_PROMPT,
            "user": f"Rewrite the following text in a natural, human-written style:\n\n{text}",
            "assistant": text,
            "source": item.get("source", "unknown"),
        }
        sft_data.append(sft_example)
    return sft_data


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("STEP 2: Clean, deduplicate, and prepare SFT dataset")
    log.info("=" * 60)

    # Load all raw files
    all_texts = []
    raw_files = list(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        log.error(f"No raw JSONL files found in {RAW_DIR}. Run download_datasets.py first.")
        return

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
        log.info(f"  Loaded {count:,} texts from {raw_file.name}")

    log.info(f"\nTotal raw texts: {len(all_texts):,}")

    # Clean
    log.info("\n── Cleaning ──")
    cleaned = []
    for item in tqdm(all_texts, desc="Cleaning"):
        text = clean_text(item["text"])
        if is_valid_text(text):
            cleaned.append({"text": text, "source": item.get("source", "unknown")})

    log.info(f"After cleaning: {len(cleaned):,} (removed {len(all_texts) - len(cleaned):,})")

    # Deduplicate
    log.info("\n── Deduplication ──")
    deduped = deduplicate(cleaned, DATA["dedup_threshold"])
    log.info(f"After dedup: {len(deduped):,}")

    # Shuffle
    random.seed(DATA["seed"])
    random.shuffle(deduped)

    # Save cleaned corpus
    cleaned_path = PROCESSED_DIR / "human_texts_clean.jsonl"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for item in deduped:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    log.info(f"Cleaned corpus saved → {cleaned_path}")

    # Create SFT pairs
    log.info("\n── Creating SFT Pairs ──")
    sft_data = create_sft_pairs(deduped)
    log.info(f"Created {len(sft_data):,} SFT training examples")

    # Train/validation split
    log.info("\n── Train/Val Split ──")
    val_size = int(len(sft_data) * DATA["val_split"])
    train_data = sft_data[val_size:]
    val_data = sft_data[:val_size]
    log.info(f"Train: {len(train_data):,}  |  Val: {len(val_data):,}")

    # Save splits
    train_path = SPLITS_DIR / "train.jsonl"
    val_path = SPLITS_DIR / "val.jsonl"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        log.info(f"Saved → {path}")

    # Stats
    log.info(f"\n{'=' * 60}")
    log.info("DATA PREPARATION COMPLETE")
    log.info(f"  Raw texts loaded:    {len(all_texts):,}")
    log.info(f"  After cleaning:      {len(cleaned):,}")
    log.info(f"  After dedup:         {len(deduped):,}")
    log.info(f"  Train examples:      {len(train_data):,}")
    log.info(f"  Val examples:        {len(val_data):,}")
    log.info(f"  Output dir:          {SPLITS_DIR}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
