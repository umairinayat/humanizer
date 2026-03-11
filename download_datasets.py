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


def extract_human_pairs(ds_config: dict) -> list[dict]:
    """Load a HF dataset and return AI -> Human paired texts."""
    repo = ds_config["repo"]
    split = ds_config["split"]
    subset = ds_config["subset"]
    name = ds_config["name"]
    
    log.info(f"Loading {repo} (split={split})...")

    try:
        ds = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
    except Exception as e:
        log.warning(f"Failed to load split='{split}' for {repo}: {e}")
        try:
            ds = load_dataset(repo)
            available_splits = list(ds.keys())
            ds = ds[available_splits[0]]
        except Exception as e2:
            log.error(f"Failed to load {repo}: {e2}")
            return []

    log.info(f"  Loaded {len(ds)} rows. Columns: {ds.column_names}")

    pairs = []
    
    if name == "dmitva/human_ai_generated_text" or repo == "dmitva/human_ai_generated_text":
        # Contains direct parallel pairs: human_text vs ai_text
        for row in tqdm(ds, desc=f"  Extracting [{name}]"):
            ht = row.get("human_text", "")
            at = row.get("ai_text", "")
            if ht and at and len(ht.strip()) > 0 and len(at.strip()) > 0:
                pairs.append({"human_text": ht.strip(), "ai_text": at.strip()})
                
    elif name == "raid" or repo == "liamdugan/raid":
        # Not perfectly paired, but we can match prompts
        # Let's collect human text per prompt, and AI text per prompt
        prompts_to_human = {}
        ai_entries = []
        for row in tqdm(ds, desc=f"  Buffering [{name}]"):
            model = row.get("model")
            prompt = row.get("prompt", "")
            generation = row.get("generation", "")
            if not generation or not prompt:
                continue
            if model == "human":
                prompts_to_human[prompt] = generation.strip()
            else:
                ai_entries.append((prompt, generation.strip()))
                
        # Now pair them
        for prompt, at in ai_entries:
            if prompt in prompts_to_human:
                ht = prompts_to_human[prompt]
                if ht and at:
                    pairs.append({"human_text": ht, "ai_text": at})
                    
    elif name == "ai_and_human_text" or repo == "NabeelShar/ai_and_human_text":
        # Single text column with generated label. No explicit pairs.
        # We can loosely pair text that shares a prompt_name.
        prompts_to_human = {}
        ai_entries = []
        for row in tqdm(ds, desc=f"  Buffering [{name}]"):
            text = row.get("text", "")
            gen = row.get("generated")
            prompt = row.get("prompt_name", "")
            if not text or prompt is None:
                continue
            if int(gen) == 0:
                prompts_to_human[prompt] = text.strip()
            elif int(gen) == 1:
                ai_entries.append((prompt, text.strip()))
                
        for prompt, at in ai_entries:
            if prompt in prompts_to_human:
                ht = prompts_to_human[prompt]
                if ht and at:
                    pairs.append({"human_text": ht, "ai_text": at})
                    
    else:
        log.warning(f"  Dataset {name} does not support pairing (AI/Human pairs not found). Skipping direct parallel export.")

    log.info(f"  Extracted {len(pairs)} pairs from {name}")
    return pairs

def save_raw(name: str, pairs: list[dict], output_dir: Path) -> Path:
    """Save extracted pairs to a JSONL file."""
    out_path = output_dir / f"{name}_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            # We save ai_text and human_text
            json.dump({"ai_text": pair["ai_text"], "human_text": pair["human_text"], "source": name}, f, ensure_ascii=False)
            f.write("\n")
    log.info(f"  Saved {len(pairs)} paired texts → {out_path}")
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

        pairs = extract_human_pairs(ds_config)
        if pairs:
            save_raw(ds_config["name"], pairs, RAW_DIR)
            total += len(pairs)
        else:
            log.warning(f"  No pairs extracted from {ds_config['name']}")

    log.info(f"\n{'=' * 60}")
    log.info(f"TOTAL PAIRS EXTRACTED: {total:,}")
    log.info(f"Raw files saved to: {RAW_DIR}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
