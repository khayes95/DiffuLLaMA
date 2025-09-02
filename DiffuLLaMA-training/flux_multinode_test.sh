#!/bin/bash

#flux: -N 2
#flux: -n 8  
#flux: --exclusive
#flux: -t 30m
#flux: -q pdebug

# Initialize conda and activate environment
source /usr/workspace/$USER/x86_miniconda/etc/profile.d/conda.sh
conda activate diffullama

# Set environment variables for proper storage usage
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TMPDIR="/p/vast1/$USER/tmp"
export TORCH_EXTENSIONS_DIR="/p/vast1/$USER/.cache/torch_extensions"
export HF_HOME="/p/vast1/$USER/.cache/huggingface"

mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$HF_HOME"

echo "Job started on $(date)"
echo "Flux job ID: $FLUX_JOB_ID"
echo "Running on nodes:"
flux run -N 2 -n 2 hostname

echo "Testing Python import:"
flux run -N 2 -n 2 bash -c "source /usr/workspace/$USER/x86_miniconda/etc/profile.d/conda.sh; conda activate diffullama; python -c 'import torch; print(f\"GPU count: {torch.cuda.device_count()}\")'"

echo "Job completed on $(date)"
