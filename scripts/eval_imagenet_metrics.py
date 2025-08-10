#!/usr/bin/env python3
"""
Simple ImageNet evaluation script.

Usage:
    python eval_imagenet_metrics.py your_sample_batch.npz
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.request

# URLs for downloading dependencies
EVALUATOR_URL = "https://raw.githubusercontent.com/lavinal712/ADM-evaluation-suite-pytorch/refs/heads/main/evaluator.py"
REFERENCE_BATCH_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"


def get_cache_dir() -> Path:
    """Get the cache directory from environment variable, consistent with codebase."""
    cache_dir = os.getenv("MINFM_CACHE_DIR")
    if cache_dir is None:
        print("Error: MINFM_CACHE_DIR environment variable is not set.")
        print("Please set the MINFM_CACHE_DIR environment variable to specify the cache directory.")
        sys.exit(1)

    eval_cache_dir = Path(cache_dir) / "eval_imagenet_metrics"
    eval_cache_dir.mkdir(parents=True, exist_ok=True)
    return eval_cache_dir


# Cache paths
CACHE_DIR = get_cache_dir()
EVALUATOR_PATH = CACHE_DIR / "evaluator.py"
REFERENCE_BATCH_PATH = CACHE_DIR / "VIRTUAL_imagenet256_labeled.npz"


def download_file(url: str, dest_path: Path, description: str) -> bool:
    """Download a file from URL to destination path with progress indication."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    try:

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\n✓ Successfully downloaded {description}")
        return True

    except Exception as e:
        print(f"\n✗ Failed to download {description}: {e}")
        return False


def ensure_dependencies():
    """Download dependencies if they don't exist."""
    success = True

    if not EVALUATOR_PATH.exists():
        print("Downloading evaluator script...")
        success &= download_file(EVALUATOR_URL, EVALUATOR_PATH, "evaluator script")

    if not REFERENCE_BATCH_PATH.exists():
        print("Downloading ImageNet reference batch...")
        success &= download_file(REFERENCE_BATCH_URL, REFERENCE_BATCH_PATH, "ImageNet reference batch")

    return success


def main():
    parser = argparse.ArgumentParser(description="Simple ImageNet evaluation script")
    parser.add_argument("sample_npz", help="NPZ file containing generated images")
    args = parser.parse_args()

    print(f"Using cache directory: {CACHE_DIR}")

    if not os.path.isfile(args.sample_npz):
        print(f"Error: File {args.sample_npz} does not exist")
        sys.exit(1)

    # Ensure dependencies are available
    if not ensure_dependencies():
        print("Failed to download dependencies")
        sys.exit(1)

    # Run the evaluator
    cmd = f"python {EVALUATOR_PATH} {REFERENCE_BATCH_PATH} {args.sample_npz}"
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
