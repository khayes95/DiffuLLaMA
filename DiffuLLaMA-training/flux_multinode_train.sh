#!/bin/bash

#flux: -N 4
#flux: -n 16
#flux: --exclusive
#flux: -t 30m
#flux: -q pdebug
#flux: --output=multinode_train_%j.out
#flux: --error=multinode_train_%j.err

# Set environment variables
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TMPDIR="/p/vast1/$USER/tmp"
export TORCH_EXTENSIONS_DIR="/p/vast1/$USER/.cache/torch_extensions"
export HF_HOME="/p/vast1/$USER/.cache/huggingface"

mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$HF_HOME"

BASE="/g/g10/$USER/DiffuLLaMA/DiffuLLaMA-training"
OUTPUT_DIR="/p/vast1/$USER/test_output_flux_4n4g"

cd "$BASE"

echo "Starting multi-node training on $(date)"
echo "Flux job ID: $FLUX_JOB_ID"
echo "Nodes: $(flux getattr size)"

# Use accelerate with DeepSpeed for proper multi-node coordination
flux run -N 4 -n 16 accelerate launch \
    --config_file /dev/stdin \
    --deepspeed_config_file accelerate_configs/zero3_offload_fixed.json \
    train.py \
    --batch-size 1 \
    --gradient-accumulate-every 1 \
    --output-dir "$OUTPUT_DIR" \
    --seed 2829 \
    --max-train-steps 1 \
    --learning-rate 1.5e-5 \
    --dataset "/p/vast1/$USER/packs/fineweb512_train_slim" \
    --model "/p/vast1/$USER/models/Llama-2-7b-hf" \
    --seq-length 511 \
    --parallel_mode data_parallel << ACCEL_CONFIG
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_processes: 16
num_machines: 4
machine_rank: 0
main_training_function: main
ACCEL_CONFIG

echo "Training completed on $(date)"
