#!/bin/bash

# Exit on error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting DiffuLLaMA Training Setup${NC}"

# Set base directory (use current directory or DiffuLLaMA-training)
if [[ $PWD == *"DiffuLLaMA-training"* ]]; then
    export BASE="$PWD"
else
    export BASE="/p/vast1/$USER/DiffuLLaMA/DiffuLLaMA-training"
fi
echo "Base directory: $BASE"

# Set environment variables for proper storage usage
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TMPDIR="/p/vast1/$USER/tmp"
export TORCH_EXTENSIONS_DIR="/p/vast1/$USER/.cache/torch_extensions"
export HF_HOME="/p/vast1/$USER/.cache/huggingface"

# Create necessary directories
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$HF_HOME"

echo "Environment setup:"
echo "  TMPDIR=$TMPDIR"
echo "  TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

# Set memory optimization for AMD GPUs
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
echo "Set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"

# Check if accelerate config exists, if not create a MINIMAL one for DeepSpeed
ACC_CFG="/usr/workspace/$USER/ds_cfgs/acc_single_gpu_minimal.yaml"
mkdir -p "$(dirname "$ACC_CFG")"
echo -e "${YELLOW}Creating minimal accelerate config for DeepSpeed...${NC}"

# Create a MINIMAL accelerate config that won't conflict with DeepSpeed
cat > "$ACC_CFG" << 'YAMLEOF'
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_processes: 1
machine_rank: 0
num_machines: 1
main_training_function: main
YAMLEOF

echo -e "${GREEN}Created minimal accelerate config at: $ACC_CFG${NC}"

# DeepSpeed configuration options
echo -e "\n${YELLOW}Select DeepSpeed configuration:${NC}"
echo "1) zero3_offload.json (Full CPU offloading - may have CPUAdam issues)"
echo "2) zero3_gpu_only.json (GPU only - needs ~90GB memory)"
echo "3) zero3.json (Standard Zero-3)"
echo "4) zero3_offload_fixed.json (Fixed CPU offload with standard AdamW)"
echo "5) zero3_param_offload.json (Params on CPU, optimizer on GPU - RECOMMENDED)"
echo -n "Enter choice [1-5]: "
read -r choice

# Use local accelerate_configs directory
CONFIG_DIR="$BASE/accelerate_configs"

case $choice in
    1)
        DS_JSON="$CONFIG_DIR/zero3_offload.json"
        echo -e "${GREEN}Selected: zero3_offload.json${NC}"
        ;;
    2)
        DS_JSON="$CONFIG_DIR/zero3_gpu_only.json"
        echo -e "${GREEN}Selected: zero3_gpu_only.json${NC}"
        ;;
    3)
        DS_JSON="$CONFIG_DIR/zero3.json"
        echo -e "${GREEN}Selected: zero3.json${NC}"
        ;;
    4)
        DS_JSON="$CONFIG_DIR/zero3_offload_fixed.json"
        echo -e "${GREEN}Selected: zero3_offload_fixed.json (Fixed CPU offload)${NC}"
        ;;
    5)
        DS_JSON="$CONFIG_DIR/zero3_param_offload.json"
        echo -e "${GREEN}Selected: zero3_param_offload.json (Params CPU, Optimizer GPU)${NC}"
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Defaulting to zero3_param_offload.json${NC}"
        DS_JSON="$CONFIG_DIR/zero3_param_offload.json"
        ;;
esac

# Check if DeepSpeed config exists
if [ ! -f "$DS_JSON" ]; then
    echo -e "${RED}Error: DeepSpeed config not found at $DS_JSON${NC}"
    echo "Please ensure the file exists before running this script."
    exit 1
fi

# Training parameters (you can modify these)
BATCH_SIZE=1
GRADIENT_ACCUMULATE=1
OUTPUT_DIR="/p/vast1/$USER/test_output_seq512"
SEED=2829
MAX_STEPS=1
LEARNING_RATE=1.5e-5
DATASET="/p/vast1/$USER/packs/fineweb512_train_slim"
MODEL="/p/vast1/$USER/models/Llama-2-7b-hf"
SEQ_LENGTH=511  # Using 511 to match 512-token shards
PARALLEL_MODE="data_parallel"

# Path to train.py
TRAIN_SCRIPT="$BASE/train.py"

# Check if train.py exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}Error: train.py not found at $TRAIN_SCRIPT${NC}"
    echo "Please ensure you're running this from the correct directory."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo -e "\n${GREEN}Training Configuration:${NC}"
echo "================================"
echo "Accelerate Config: $ACC_CFG"
echo "DeepSpeed Config: $DS_JSON"
echo "Train Script: $TRAIN_SCRIPT"
echo "Output Directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATE"
echo "Max Steps: $MAX_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Sequence Length: $SEQ_LENGTH"
echo "PYTORCH_HIP_ALLOC_CONF: expandable_segments:True"
echo "================================"

# Ask for confirmation
echo -n -e "\n${YELLOW}Proceed with training? (y/n): ${NC}"
read -r confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Run training
echo -e "\n${GREEN}Starting training...${NC}\n"

# The actual training command
accelerate launch \
  --config_file "$ACC_CFG" \
  --deepspeed_config_file "$DS_JSON" \
  "$TRAIN_SCRIPT" \
    --batch-size $BATCH_SIZE \
    --gradient-accumulate-every $GRADIENT_ACCUMULATE \
    --output-dir "$OUTPUT_DIR" \
    --seed $SEED \
    --max-train-steps $MAX_STEPS \
    --learning-rate $LEARNING_RATE \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --seq-length $SEQ_LENGTH \
    --parallel_mode $PARALLEL_MODE

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Training completed successfully!${NC}"
else
    echo -e "\n${RED}Training failed. Please check the error messages above.${NC}"
    exit 1
fi
