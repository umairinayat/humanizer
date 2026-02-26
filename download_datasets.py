"""
Step 1: Download human-written text from all 5 HuggingFace datasets.
Filters for human-only text and saves raw JSONL files per source.

Usage:
    python download_datasets.py
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from config import DATASETS, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/download.log"),
    ],
)
log = logging.getLogger(__name__)


def extract_human_texts(ds_config: dict) -> list[str]:
    """Load a HF dataset and return only human-written texts."""
    repo = ds_config["repo"]
    text_col = ds_config["text_col"]
    label_col = ds_config["label_col"]
    human_value = ds_config["human_value"]
    subset = ds_config["subset"]
    split = ds_config["split"]

    log.info(f"Loading {repo} (split={split}, subset={subset})...")

    try:
        if subset:
            ds = load_dataset(repo, subset, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(repo, split=split, trust_remote_code=True)
    except Exception as e:
        log.warning(f"Failed to load split='{split}' for {repo}: {e}")
        log.info(f"Trying to load {repo} without specifying split...")
        try:
            ds = load_dataset(repo, trust_remote_code=True)
            # Take the first available split
            available_splits = list(ds.keys())
            log.info(f"Available splits: {available_splits}")
            ds = ds[available_splits[0]]
        except Exception as e2:
            log.error(f"Failed to load {repo}: {e2}")
            return []

    log.info(f"  Loaded {len(ds)} rows. Columns: {ds.column_names}")

    # Detect text column if the configured one doesn't exist
    if text_col not in ds.column_names:
        candidates = ["text", "content", "generation", "Text", "article", "document"]
        found = [c for c in candidates if c in ds.column_names]
        if found:
            text_col = found[0]
            log.warning(f"  Column '{ds_config['text_col']}' not found. Using '{text_col}'.")
        else:
            log.error(f"  No text column found in {ds.column_names}. Skipping.")
            return []

    texts = []

    if label_col is None:
        # All rows are human (e.g., wikipedia_human_written_text)
        for row in tqdm(ds, desc=f"  Extracting [{ds_config['name']}]"):
            t = row.get(text_col, "")
            if t and isinstance(t, str) and len(t.strip()) > 0:
                texts.append(t.strip())
    else:
        # Filter by label
        if label_col not in ds.column_names:
            # Try case-insensitive match
            col_map = {c.lower(): c for c in ds.column_names}
            if label_col.lower() in col_map:
                label_col = col_map[label_col.lower()]
                log.warning(f"  Using case-matched label column: '{label_col}'")
            else:
                log.error(f"  Label column '{label_col}' not found in {ds.column_names}. Skipping.")
                return []

        for row in tqdm(ds, desc=f"  Extracting [{ds_config['name']}]"):
            label = row.get(label_col)
            # Handle string or int comparison
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

    log.info(f"  Extracted {len(texts)} human texts from {ds_config['name']}")
    return texts


def save_raw(name: str, texts: list[str], output_dir: Path) -> Path:
    """Save extracted texts to a JSONL file."""
    out_path = output_dir / f"{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text, "source": name}, f, ensure_ascii=False)
            f.write("\n")
    log.info(f"  Saved {len(texts)} texts → {out_path}")
    return out_path


def main():
    log.info("=" * 60)
    log.info("STEP 1: Download & extract human text from HF datasets")
    log.info("=" * 60)

    total = 0
    for ds_config in DATASETS:
        log.info(f"\n{'─' * 40}")
        log.info(f"Dataset: {ds_config['name']} ({ds_config['repo']})")
        log.info(f"{'─' * 40}")

        texts = extract_human_texts(ds_config)
        if texts:
            save_raw(ds_config["name"], texts, RAW_DIR)
            total += len(texts)
        else:
            log.warning(f"  No human texts extracted from {ds_config['name']}")

    log.info(f"\n{'=' * 60}")
    log.info(f"TOTAL HUMAN TEXTS EXTRACTED: {total:,}")
    log.info(f"Raw files saved to: {RAW_DIR}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
