#!/bin/bash
export NCCL_DEBUG=WARN
export UV_PROJECT_ENVIRONMENT=".venv"

# @User: Set the following environment variables before running the script
# export MINFM_CACHE_DIR="<minfm-cache-dir>"
# export MINFM_DATA_DIR="<minfm-data-dir>"
: "${MINFM_DATA_DIR:?Error: MINFM_DATA_DIR is not set}"
: "${MINFM_CACHE_DIR:?Error: MINFM_CACHE_DIR is not set}"

# Make sure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Install dependencies
uv sync --no-dev
source ${UV_PROJECT_ENVIRONMENT}/bin/activate

echo "MINFM_DATA_DIR: ${MINFM_DATA_DIR}"
echo "MINFM_CACHE_DIR: ${MINFM_CACHE_DIR}"
echo "UV_PROJECT_ENVIRONMENT: ${UV_PROJECT_ENVIRONMENT}"
echo "HF_HOME: ${HF_HOME}"

echo "python path: $(which python), version: $(python --version)"


if nvidia-smi | grep -q "H100"; then
    echo "H100 found, installing FA3 wheel..."
    uv run pip install https://huggingface.co/datasets/Kai-46/minFM/resolve/main/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
fi


# Get this script's directory
SCRIPT_DIR=$(realpath $(dirname "$0"))
echo "This script's directory: ${SCRIPT_DIR}"
DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/configs/flux_tiny_imagenet.yaml"

if [ -z "$1" ]; then
    echo "No mode provided, using default mode: train"
    MODE="train"
else
    MODE="$1"
fi

if [ -z "$2" ]; then
    echo "No config file provided, using default config..."
    CONFIG="${DEFAULT_CONFIG_PATH}"
else
    CONFIG="$2"
fi


echo "Config: ${CONFIG}"

# Grep the config file for the exp_dir; strip "" if it exists; remove suffix / if it exists
EXP_DIR=$(grep -oP 'exp_dir: \K.*' ${CONFIG} | sed 's:"::g' | sed 's:/*$::')
echo "Experiment directory: ${EXP_DIR}"


# Start background processes
# Only node 0 does this
if [ ${RANK} -eq 0 ]; then
    BG_LOG_DIR="${EXP_DIR}/logs_bg"
    mkdir -p ${BG_LOG_DIR}

    echo "[Background] Starting checkpoint cleaner..."
    python scripts/start_checkpoint_cleaner.py --checkpoint_dir=${EXP_DIR}/checkpoints/ \
        > ${BG_LOG_DIR}/checkpoint_cleaner.log 2>&1 &

    echo "[Background] Starting http server..."
    python scripts/start_http_server.py --directory=${EXP_DIR} \
        > ${BG_LOG_DIR}/http_server.log 2>&1 &
fi

# Run the training script
TRAIN_LOG_DIR="${EXP_DIR}/logs_train"
mkdir -p ${TRAIN_LOG_DIR}
LOG_FILE="${TRAIN_LOG_DIR}/node_${RANK}.log"
echo "Log file: ${LOG_FILE}"

uv run torchrun \
    --nnodes=${WORLD_SIZE} --nproc_per_node=${NUM_OF_GPUS} --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    -m entrypoint --mode=${MODE} --config=${CONFIG} \
    2>&1 | tee ${LOG_FILE}
