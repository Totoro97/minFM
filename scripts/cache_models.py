#!/usr/bin/env python3
"""
Model Caching Script

This script caches various models used in the minFM project:
- T5 and CLIP models (converted to bfloat16 format)
- FLUX.1-dev model safetensors files (ae.safetensors and flux1-dev.safetensors, converted to bfloat16 format)

The cached models are saved to a specified cache directory, which is useful for
pre-caching models to speed up training initialization or avoid huggingface access rate limits.

FLUX model caching requires:
- A Hugging Face authentication token: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Agreement to FLUX model license terms
- Sufficient disk space (FLUX models are large ~20GB+)

FLUX files downloaded directly as safetensors:
- ae.safetensors: VAE/AutoEncoder component
- flux1-dev.safetensors: Transformer/MMDiT component
"""

import json
import os
from pathlib import Path
import shutil
import sys
from urllib import error, request

from huggingface_hub import HfApi
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

# Define all supported models
ENCODER_MODELS_TO_CACHE = {
    "google/t5-v1_1-xxl": (T5EncoderModel, T5Tokenizer),
    "google/flan-t5-xxl": (T5EncoderModel, T5Tokenizer),
    "openai/clip-vit-large-patch14": (CLIPTextModel, CLIPTokenizer),
}

# FLUX model configuration
FLUX_MODEL_NAME = "black-forest-labs/FLUX.1-dev"

# Success file configuration
SUCCESS_FILE_NAME = "cached_models_success.json"


def _format_size(num_bytes: int) -> str:
    """Format byte size into human readable format."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _get_remote_size(url: str, headers: dict | None = None) -> int | None:
    """Get the remote file size."""
    try:
        req = request.Request(url, method="HEAD", headers=headers or {})
        with request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            if length is None:
                return None
            return int(length)
    except Exception:
        return None


def _stream_download(url: str, destination: Path, headers: dict | None = None) -> None:
    """Download a file from URL to destination with progress logging."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _get_remote_size(url, headers=headers)
    if destination.exists() and remote_size is not None:
        local_size = destination.stat().st_size
        if local_size == remote_size:
            print(f"  ✓ Already downloaded: {destination.name} ({_format_size(local_size)})")
            return

    print(f"  Downloading {destination.name}...")
    try:
        req = request.Request(url, headers=headers or {})
        with request.urlopen(req) as resp, open(destination, "wb") as out_f:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total is not None else None
            bytes_read = 0
            next_log_at = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                bytes_read += len(chunk)
                if total_bytes:
                    percent = int(bytes_read * 100 / total_bytes)
                    if percent >= next_log_at:
                        print(f"    Progress {percent}% ({_format_size(bytes_read)} / {_format_size(total_bytes)})")
                        # Log every 10%
                        next_log_at = ((percent // 10) + 1) * 10
            if total_bytes and bytes_read != total_bytes:
                raise OSError(f"Incomplete download for {destination.name}: got {bytes_read}, expected {total_bytes}")
            print(f"  ✓ Downloaded: {destination.name} ({_format_size(bytes_read)})")
    except error.HTTPError as e:
        print(f"HTTP error while downloading {url}: {e}")
        raise
    except error.URLError as e:
        print(f"URL error while downloading {url}: {e}")
        raise


def _build_hf_headers(token: str) -> dict:
    """Build headers for Hugging Face requests."""
    return {"User-Agent": "minFM-flux-downloader/1.0", "Authorization": f"Bearer {token}"}


def load_success_file(cache_dir: Path) -> dict[str, str]:
    """
    Load the success file containing information about successfully cached models.

    Args:
        cache_dir: The cache directory containing the success file.

    Returns:
        Dictionary mapping model names to their cached paths.
    """
    success_file = cache_dir / SUCCESS_FILE_NAME
    if success_file.exists():
        try:
            with open(success_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load success file {success_file}: {e}")
            return {}
    return {}


def save_success_file(cache_dir: Path, success_data: dict[str, str]) -> None:
    """
    Save the success file with information about successfully cached models.

    Args:
        cache_dir: The cache directory where the success file will be saved.
        success_data: Dictionary mapping model names to their cached paths.
    """
    success_file = cache_dir / SUCCESS_FILE_NAME
    try:
        with open(success_file, "w") as f:
            json.dump(success_data, f, indent=2)
    except OSError as e:
        print(f"Warning: Could not save success file {success_file}: {e}")


def is_model_cached(model_name: str, cache_dir: Path, success_data: dict[str, str]) -> bool:
    """
    Check if a model is already successfully cached.

    Args:
        model_name: The name of the model to check.
        cache_dir: The cache directory.
        success_data: Dictionary of successfully cached models.

    Returns:
        True if the model is cached and the cache path exists, False otherwise.
    """
    if model_name not in success_data:
        return False

    cached_path = Path(success_data[model_name])
    if not cached_path.is_absolute():
        cached_path = cache_dir / cached_path

    # Check if path exists
    if not cached_path.exists():
        return False

    try:
        # Handle both directories (encoder models) and files (FLUX models)
        if cached_path.is_dir():
            # For directories, check if they have any files (not just subdirectories)
            return any(cached_path.iterdir())
        elif cached_path.is_file():
            # For files, check if they have non-zero size
            return cached_path.stat().st_size > 0
        else:
            # Path exists but is neither file nor directory
            return False
    except (OSError, PermissionError):
        # If we can't read the path, assume it's not properly cached
        return False


def validate_hf_token() -> str:
    """
    Validates that a Hugging Face token is available and has access to FLUX model.

    Returns:
        The validated token string.

    Raises:
        SystemExit: If no valid token is found or token doesn't have access.
    """
    # Check for token in environment
    token = os.getenv("HF_TOKEN")

    if not token:
        print("\nError: Hugging Face token is required to download FLUX models.")
        print("Please set the HF_TOKEN environment variable:")
        print()
        print("You can get a token from: https://huggingface.co/settings/tokens")
        print("Make sure to accept the FLUX model license at:")
        print(f"  https://huggingface.co/{FLUX_MODEL_NAME}")
        sys.exit(1)

    # Validate token has access to FLUX model
    try:
        api = HfApi()
        # Try to get model info to validate access
        api.model_info(FLUX_MODEL_NAME, token=token)
        print(f"✓ Hugging Face token validated with access to {FLUX_MODEL_NAME}")
        return token
    except Exception as e:
        print(f"\nError: Unable to access {FLUX_MODEL_NAME} with provided token.")
        print("Please ensure:")
        print("1. Your token is valid")
        print("2. You have accepted the model license at:")
        print(f"   https://huggingface.co/{FLUX_MODEL_NAME}")
        print(f"\nError details: {e}")
        sys.exit(1)


def cache_encoder_model(model_name: str, cache_dir: Path, success_data: dict[str, str]) -> bool:
    """
    Loads a model and its tokenizer, converts it to bfloat16, and saves it to a cache directory.

    Args:
        model_name: The name of the model to cache (e.g., 'google/flan-t5-xxl').
        cache_dir: The root directory where the cached models will be stored.
        success_data: Dictionary to track successfully cached models.

    Returns:
        True if the model was cached successfully, False if it was skipped.
    """
    # Check if model is already cached
    if is_model_cached(model_name, cache_dir, success_data):
        print(f"✓ Model '{model_name}' is already cached. Skipping...")
        return False

    # Create target directory
    target_dir = cache_dir / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caching model '{model_name}' to '{target_dir}'...")

    try:
        model_cls, tokenizer_cls = ENCODER_MODELS_TO_CACHE[model_name]

        # Create subdirectories
        tokenizer_dir = target_dir / "tokenizer"
        model_dir = target_dir / "model"
        tokenizer_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        # Load and save tokenizer
        print(f"  Loading tokenizer for {model_name}...")
        tokenizer = tokenizer_cls.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"  Saved tokenizer to {tokenizer_dir}")

        # Load model in bfloat16 on CPU
        print(f"  Loading model in bfloat16 on CPU for {model_name}...")
        model = model_cls.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model.save_pretrained(model_dir)
        print(f"  Saved model to {model_dir}")

        # Update success data
        success_data[model_name] = str(target_dir.relative_to(cache_dir))
        save_success_file(cache_dir, success_data)

        print(f"Successfully cached '{model_name}'.\n")
        return True

    except Exception as e:
        print(f"Failed to cache '{model_name}': {e}")
        # Clean up partial cache
        shutil.rmtree(target_dir, ignore_errors=True)
        sys.exit(1)


def convert_to_bf16(destination: Path) -> None:
    """
    Convert a safetensors file to bfloat16.
    """
    from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file

    state_dict = load_safetensors_file(destination)
    for key, tensor in state_dict.items():
        if tensor.dtype != torch.bfloat16:
            state_dict[key] = tensor.to(torch.bfloat16)

    save_safetensors_file(state_dict, destination)


def cache_flux_model(cache_dir: Path, success_data: dict[str, str], use_bf16: bool = True) -> bool:
    """
    Caches the FLUX model safetensors files directly from Hugging Face.

    Downloads:
    - ae.safetensors (VAE/AutoEncoder)
    - flux1-dev.safetensors (Transformer/MMDiT)

    Args:
        cache_dir: The root directory where the cached models will be stored.
        success_data: Dictionary to track successfully cached models.
        use_bf16: Whether to convert the model to bfloat16 before saving
    Returns:
        True if the model was cached successfully, False if it was skipped.
    """
    # Check if FLUX components are already cached
    flux_vae_key = f"{FLUX_MODEL_NAME}/ae"
    flux_transformer_key = f"{FLUX_MODEL_NAME}/flux1-dev"

    vae_cached = is_model_cached(flux_vae_key, cache_dir, success_data)
    transformer_cached = is_model_cached(flux_transformer_key, cache_dir, success_data)

    if vae_cached and transformer_cached:
        print(f"✓ FLUX model '{FLUX_MODEL_NAME}' components are already cached. Skipping...")
        return False

    # Show what needs to be cached
    components_to_cache = []
    if not vae_cached:
        components_to_cache.append("VAE (ae.safetensors)")
    if not transformer_cached:
        components_to_cache.append("Transformer (flux1-dev.safetensors)")

    if components_to_cache:
        print(f"Caching FLUX model '{FLUX_MODEL_NAME}' components: {', '.join(components_to_cache)}...")
    else:
        # This shouldn't happen due to the check above, but just in case
        print(f"✓ FLUX model '{FLUX_MODEL_NAME}' components are already cached. Skipping...")
        return False

    token = validate_hf_token()
    headers = _build_hf_headers(token)

    # Create FLUX directory structure
    flux_base_dir = cache_dir / FLUX_MODEL_NAME
    flux_base_dir.mkdir(parents=True, exist_ok=True)

    # Define file URLs and destinations
    base_url = f"https://huggingface.co/{FLUX_MODEL_NAME}/resolve/main"
    files_to_download = []

    if not vae_cached:
        vae_url = f"{base_url}/ae.safetensors"
        vae_path = flux_base_dir / "ae.safetensors"
        files_to_download.append((vae_url, vae_path, flux_vae_key, "VAE"))

    if not transformer_cached:
        transformer_url = f"{base_url}/flux1-dev.safetensors"
        transformer_path = flux_base_dir / "flux1-dev.safetensors"
        files_to_download.append((transformer_url, transformer_path, flux_transformer_key, "Transformer"))

    try:
        # Download each required file
        for url, destination, success_key, component_name in files_to_download:
            print(f"  Downloading {component_name}...")
            _stream_download(url, destination, headers=headers)

            # Update success tracking
            success_data[success_key] = str(destination.relative_to(cache_dir))
            save_success_file(cache_dir, success_data)
            print(f"  ✓ {component_name} cached successfully")

            if use_bf16:
                print(f"Converting {component_name} to bfloat16...")
                convert_to_bf16(destination)
                print(f"  Saved {component_name} to {destination} (bf16)")

        print("Successfully cached FLUX model components.\n")

        return True

    except Exception as e:
        print(f"Failed to cache FLUX model: {e}")
        print("Check: disk space, HF token access, and model license agreement")

        # Clean up partial downloads and success data
        for _, destination, success_key, _ in files_to_download:
            if destination.exists():
                try:
                    destination.unlink()
                except Exception:
                    pass
            success_data.pop(success_key, None)

        save_success_file(cache_dir, success_data)
        sys.exit(1)


def main() -> None:
    """Main function to parse arguments and cache models."""

    # Get cache directory from environment variable
    cache_dir = os.getenv("MINFM_CACHE_DIR")
    if cache_dir is None:
        print("Error: MINFM_CACHE_DIR environment variable is not set.")
        print("Please set the MINFM_CACHE_DIR environment variable to specify the cache directory.")
        sys.exit(1)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Using cache directory: {cache_path}")

    # Load existing success data
    success_data = load_success_file(cache_path)
    if success_data:
        print(f"Found {len(success_data)} previously cached models in success file.")

    newly_cached = []
    already_cached = []

    # Cache T5 and CLIP models
    for model_name in ENCODER_MODELS_TO_CACHE:
        if cache_encoder_model(model_name, cache_path, success_data):
            newly_cached.append(model_name)
        else:
            already_cached.append(model_name)

    # Cache FLUX model
    flux_model_display_name = f"{FLUX_MODEL_NAME} (VAE + DENOISER)"
    if cache_flux_model(cache_path, success_data, use_bf16=True):
        newly_cached.append(flux_model_display_name)
    else:
        already_cached.append(flux_model_display_name)

    # Final summary
    print("=" * 80)
    print("CACHING SUMMARY:")
    print(f"  Models newly cached ({len(newly_cached)}):")
    for model in newly_cached:
        print(f"    - {model}")

    print(f"  Models already cached ({len(already_cached)}):")
    for model in already_cached:
        print(f"    - {model}")

    print(f"  All models in success file ({len(success_data)}):")
    for model_key in sorted(success_data.keys()):
        print(f"    - {model_key}")

    print(f"  Success file location: {cache_path / SUCCESS_FILE_NAME}")
    print("=" * 80)

    # Since we only reach this point if all operations succeeded
    if len(newly_cached) == 0:
        print("All models were already cached - no work needed.")
    else:
        print(f"Successfully cached {len(newly_cached)} new model(s).")


if __name__ == "__main__":
    main()
