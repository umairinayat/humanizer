"""
Run the full pipeline end-to-end.

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip_download    # Skip dataset download
    python run_pipeline.py --only train       # Run only training
"""

import argparse
import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

STEPS = {
    "download": {
        "script": "download_datasets.py",
        "desc": "Download & extract human text from HF datasets",
    },
    "prepare": {
        "script": "prepare_data.py",
        "desc": "Clean, deduplicate, create SFT splits",
    },
    "train": {
        "script": "train.py",
        "desc": "QLoRA fine-tuning",
    },
    "evaluate": {
        "script": "evaluate.py",
        "desc": "Evaluate fine-tuned model",
    },
    "merge": {
        "script": "merge_adapter.py",
        "desc": "Merge adapter into base model (optional)",
    },
}


def run_step(name: str, extra_args: list[str] = None):
    step = STEPS[name]
    log.info(f"\n{'=' * 60}")
    log.info(f"STEP: {step['desc']}")
    log.info(f"Script: {step['script']}")
    log.info(f"{'=' * 60}\n")

    cmd = [sys.executable, step["script"]]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error(f"Step '{name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    log.info(f"Step '{name}' completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Run humanizer pipeline")
    parser.add_argument("--only", type=str, choices=list(STEPS.keys()),
                        help="Run only this step")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip adapter merge")
    parser.add_argument("--train_args", nargs="*", default=[],
                        help="Extra args for train.py (e.g. --epochs 3)")
    args = parser.parse_args()

    if args.only:
        extra = args.train_args if args.only == "train" else []
        run_step(args.only, extra)
        return

    # Full pipeline
    order = ["download", "prepare", "train", "evaluate"]

    if args.skip_download:
        order.remove("download")

    if not args.skip_merge:
        order.append("merge")

    log.info("Pipeline steps: " + " → ".join(order))

    for step_name in order:
        extra = args.train_args if step_name == "train" else []
        run_step(step_name, extra)

    log.info("\n" + "=" * 60)
    log.info("FULL PIPELINE COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
