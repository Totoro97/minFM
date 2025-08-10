#!/bin/bash

# Parse command line arguments
EXCLUDE_IMAGENET=false
HELP=false
UV_PROJECT_ENVIRONMENT=".venv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exclude-imagenet)
            EXCLUDE_IMAGENET=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Cache models and datasets for minFM training"
    echo ""
    echo "Options:"
    echo "  --exclude-imagenet    Skip caching ImageNet dataset"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Required environment variables:"
    echo "  MINFM_CACHE_DIR      Directory to cache models"
    echo "  HF_TOKEN            HuggingFace token for accessing models"
    echo "  MINFM_DATA_DIR      Directory to cache datasets (if not excluding ImageNet)"
    exit 0
fi

# Check if MINFM_CACHE_DIR is set
if [ -z "$MINFM_CACHE_DIR" ]; then
    echo "Error: MINFM_CACHE_DIR is not set"
    exit 1
fi

# Check if HF_TOKEN is set
# Note: Please agree to the terms of service of flux.1-dev and imagenet in huggingface before setting HF_TOKEN
# flux-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev
# imagenet: https://huggingface.co/datasets/ILSVRC/imagenet-1k
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set, check https://huggingface.co/settings/tokens"
    exit 1
fi

# Check if MINFM_DATA_DIR is set (only required if not excluding ImageNet)
if [ "$EXCLUDE_IMAGENET" = false ] && [ -z "$MINFM_DATA_DIR" ]; then
    echo "Error: MINFM_DATA_DIR is not set (required for ImageNet caching)"
    exit 1
fi

# Install dependencies
uv sync --no-dev
source ${UV_PROJECT_ENVIRONMENT}/bin/activate

# Cache all models to $MINFM_CACHE_DIR
python scripts/cache_models.py

# Convert FLUX.1-dev denoiser checkpoint from safetensors to DCP format
python scripts/safetensors_to_dcp.py \
    $MINFM_CACHE_DIR/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors \
    $MINFM_CACHE_DIR/black-forest-labs/FLUX.1-dev/denoiser-dcp \
    --add_dict_key_prefix model

# Cache imagenet to $MINFM_DATA_DIR/imagenet (unless excluded)
if [ "$EXCLUDE_IMAGENET" = false ]; then
    echo "Caching ImageNet dataset..."
    python scripts/cache_imagenet.py --extract
else
    echo "Skipping ImageNet caching (--exclude-imagenet flag specified)"
fi
