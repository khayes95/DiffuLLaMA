#!/bin/bash

# Load conda environment
source /usr/workspace/$USER/x86_miniconda/etc/profile.d/conda.sh
conda activate diffullama

# Set temp directories to avoid disk quota issues
export TMPDIR=/usr/workspace/$USER/tmp
export TORCH_EXTENSIONS_DIR=/usr/workspace/$USER/torch_extensions

# Set environment for distributed training
export RANK=${FLUX_TASK_RANK}
export LOCAL_RANK=${FLUX_TASK_LOCAL_ID}
export WORLD_SIZE=${FLUX_JOB_SIZE}

# Get master node IP address (first node in allocation)
if [ "$RANK" == "0" ]; then
    export MASTER_ADDR=$(hostname -I | awk '{print $1}')
    echo "$MASTER_ADDR" > /tmp/master_addr_${FLUX_JOB_ID}.txt
else
    sleep 2
    export MASTER_ADDR=$(cat /tmp/master_addr_${FLUX_JOB_ID}.txt)
fi

export MASTER_PORT=$((30000 + (${RANDOM} % 1000)))

# AMD GPU settings
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES=${FLUX_TASK_LOCAL_ID}

# Diagnostic output
echo "LAUNCH OK: py $(python3 --version | cut -d' ' -f2) torch $(python3 -c 'import torch; print(torch.__version__)') cuda $(python3 -c 'import torch; print(torch.cuda.is_available())') rank $RANK host $(hostname)"
echo "[DDP] $(date) host=$(hostname) RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Run the training
python3 train.py "$@"
