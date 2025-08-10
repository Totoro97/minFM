#!/usr/bin/env python3
"""
Convert SafeTensors checkpoints to PyTorch Distributed Checkpoint (DCP) format.

This script converts multi-file safetensors checkpoints (like FLUX.1-dev denoiser)
to PyTorch's distributed checkpoint format for use with FSDP training.
"""

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
import torch
import torch.distributed.checkpoint as dcp


def get_safetensor_files(directory: Path) -> list[Path]:
    """Get all .safetensors files in a directory."""
    return [f for f in directory.glob("*.safetensors") if f.is_file()]


def load_sharded_safetensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """
    Load safetensors files from a directory and merge them.

    Args:
        checkpoint_dir: Directory containing safetensors files

    Returns:
        Combined state dictionary from all safetensors files
    """
    safetensor_files = get_safetensor_files(checkpoint_dir)

    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")

    print(f"Found {len(safetensor_files)} safetensors files to merge:")
    for file_path in safetensor_files:
        print(f"  {file_path.name}")

    tensors = {}
    for file_path in safetensor_files:
        print(f"Loading {file_path}...")
        with safe_open(file_path, framework="pt") as sf:
            for key in sf.keys():
                if key in tensors:
                    print(f"Warning: Duplicate key '{key}' found in {file_path.name}")
                tensors[key] = sf.get_tensor(key)

    print(f"Successfully loaded {len(tensors)} parameters total")
    return tensors



def convert_safetensors_dir_to_dcp_core(safetensors_path: Path, output_dir: Path, model_key: str = "model", add_dict_key_prefix: str = None) -> None:
    """
    Convert safetensors checkpoint to PyTorch distributed checkpoint format.

    Args:
        safetensors_path: Directory containing safetensors files
        output_dir: Directory to save the DCP checkpoint
        model_key: Key to use for the model in the DCP (usually "model" or "ema")
    """
    print(f"Converting safetensors from {safetensors_path} to DCP at {output_dir}")
    print(f"Using model key: {model_key}")

    # Load the safetensors checkpoint
    state_dict = load_sharded_safetensors(safetensors_path)

    if add_dict_key_prefix:
        state_dict = {f"{add_dict_key_prefix}.{key}": value for key, value in state_dict.items()}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the DCP state dict
    # DCP expects a nested structure where the model state is under a key
    dcp_state_dict = {model_key: state_dict}

    # Generate simple metadata
    metadata = {
        "converted_from": "safetensors",
        "source_dir": str(safetensors_path),
        "model_key": model_key,
    }

    # Save using PyTorch distributed checkpoint
    print(f"Saving DCP checkpoint to {output_dir}...")
    print(f"Total parameters: {len(state_dict):,}")

    # Calculate approximate memory usage
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    print(f"Total parameter count: {total_params:,}")
    print(f"Approximate memory usage: {total_bytes / (1024**3):.2f} GB")

    dcp.save(dcp_state_dict, checkpoint_id=str(output_dir))

    # Save metadata separately as a regular PyTorch file for easy access
    metadata_file = output_dir / "conversion_metadata.pth"
    torch.save(metadata, metadata_file)

    print("Conversion completed successfully!")
    print(f"DCP checkpoint saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_file}")





def convert_safetensors_dir_to_dcp(safetensors_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    # Perform conversion
    convert_safetensors_dir_to_dcp_core(
        safetensors_path,
        output_dir,
        model_key=args.model_key,
        add_dict_key_prefix=args.add_dict_key_prefix,
    )


def convert_safetensors_file_to_dcp(safetensors_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    state_dict = load_safetensors_file(safetensors_path)

    if args.add_dict_key_prefix:
        state_dict = {f"{args.add_dict_key_prefix}.{key}": value for key, value in state_dict.items()}

    dcp.save(state_dict, checkpoint_id=str(output_dir))


def main():
    """Main CLI interface for the safetensors to DCP conversion script.

    Examples:
    # Convert FLUX denoiser checkpoint
    python safetensors_to_dcfp.py /path/to/black-forest-labs/FLUX.1-dev/denoiser /path/to/output/dcp_checkpoint

    # Convert with custom model key
    python safetensors_to_dcp.py /path/to/safetensors/dir /path/to/output/dcp_checkpoint --model-key ema
    """

    parser = argparse.ArgumentParser(
        description="Convert SafeTensors checkpoints to PyTorch Distributed Checkpoint (DCP) format"
    )

    parser.add_argument(
        "safetensors_path", type=str, help="Path to safetensors file or directory containing safetensors files and index.json"
    )

    parser.add_argument("output_dir", type=str, help="Path to output directory for DCP checkpoint")

    parser.add_argument(
        "--model-key",
        type=str,
        default="model",
        choices=["model", "ema"],
        help="Key to use for the model in the DCP structure (default: model)",
    )

    parser.add_argument(
        "--add_dict_key_prefix", type=str, help="Add a prefix to the dict key"
    )

    args = parser.parse_args()

    safetensors_path = Path(args.safetensors_path)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not safetensors_path.exists():
        print(f"Error: Safetensors directory does not exist: {safetensors_path}")
        return 1

    if safetensors_path.is_dir():
        # Convert a directory of safetensors (diffusers format) files to a DCP checkpoint
        convert_safetensors_dir_to_dcp(safetensors_path, output_dir, args=args)
    else:
        # Convert a single safetensors file to a DCP checkpoint
        convert_safetensors_file_to_dcp(safetensors_path, output_dir, args=args)


if __name__ == "__main__":
    exit(main())
