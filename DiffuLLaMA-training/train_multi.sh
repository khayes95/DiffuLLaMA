#!/bin/bash

# Exit on error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}DiffuLLaMA Multi-GPU/Multi-Node Training Setup${NC}"

# Set base directory
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

# Select training configuration
echo -e "\n${YELLOW}Select training configuration:${NC}"
echo "1) Single GPU (1 node, 1 GPU) - Debug mode"
echo "2) Multi-GPU (1 node, 4 GPUs)"
echo "3) Multi-Node (custom configuration)"
echo -n "Enter choice [1-3]: "
read -r config_choice

case $config_choice in
    1)
        NUM_NODES=1
        NUM_GPUS=1
        NODE_RANK=0
        MASTER_ADDR="localhost"
        MASTER_PORT=29500
        echo -e "${GREEN}Selected: Single GPU Debug Mode${NC}"
        ;;
    2)
        NUM_NODES=1
        NUM_GPUS=4
        NODE_RANK=0
        MASTER_ADDR="localhost"
        MASTER_PORT=29500
        echo -e "${GREEN}Selected: 1 Node, 4 GPUs${NC}"
        ;;
    3)
        echo -n "Enter number of nodes: "
        read -r NUM_NODES
        echo -n "Enter GPUs per node: "
        read -r NUM_GPUS
        echo -n "Enter this node's rank (0 for master): "
        read -r NODE_RANK
        if [ "$NODE_RANK" -eq 0 ]; then
            MASTER_ADDR=$(hostname -i | awk '{print $1}')
            echo -e "${GREEN}This is the master node. Master address: $MASTER_ADDR${NC}"
            echo -e "${YELLOW}Share this with other nodes: MASTER_ADDR=$MASTER_ADDR${NC}"
        else
            echo -n "Enter master node address: "
            read -r MASTER_ADDR
        fi
        MASTER_PORT=29500
        echo -e "${GREEN}Selected: $NUM_NODES Nodes, $NUM_GPUS GPUs per node${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))

# Create appropriate accelerate config
ACC_CFG="/tmp/acc_config_${NUM_NODES}n${NUM_GPUS}g.yaml"
echo -e "${YELLOW}Creating accelerate config for $NUM_NODES nodes, $NUM_GPUS GPUs...${NC}"

cat > "$ACC_CFG" << YAMLEOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_processes: $NUM_GPUS
machine_rank: $NODE_RANK
num_machines: $NUM_NODES
main_training_function: main
YAMLEOF

if [ "$NUM_NODES" -gt 1 ]; then
    cat >> "$ACC_CFG" << YAMLEOF
deepspeed_hostfile: $BASE/hostfile
main_process_ip: $MASTER_ADDR
main_process_port: $MASTER_PORT
rdzv_backend: c10d
same_network: true
YAMLEOF
fi

echo -e "${GREEN}Created accelerate config at: $ACC_CFG${NC}"

# Select DeepSpeed configuration
echo -e "\n${YELLOW}Select DeepSpeed configuration:${NC}"
echo "1) zero3_offload_fixed.json (Params on CPU, optimizer on GPU)"
echo "2) zero3_gpu_only.json (Everything on GPU - needs more memory)"
echo "3) zero3.json (Standard Zero-3)"
echo -n "Enter choice [1-3]: "
read -r ds_choice

CONFIG_DIR="$BASE/accelerate_configs"

case $ds_choice in
    1)
        DS_JSON="$CONFIG_DIR/zero3_offload_fixed.json"
        echo -e "${GREEN}Selected: zero3_offload_fixed.json${NC}"
        ;;
    2)
        DS_JSON="$CONFIG_DIR/zero3_gpu_only.json"
        echo -e "${GREEN}Selected: zero3_gpu_only.json${NC}"
        ;;
    3)
        DS_JSON="$CONFIG_DIR/zero3.json"
        echo -e "${GREEN}Selected: zero3.json${NC}"
        ;;
    *)
        DS_JSON="$CONFIG_DIR/zero3_offload_fixed.json"
        echo -e "${YELLOW}Invalid choice. Defaulting to zero3_offload_fixed.json${NC}"
        ;;
esac

# Training parameters
BATCH_SIZE=1
GRADIENT_ACCUMULATE=1
OUTPUT_DIR="/p/vast1/$USER/test_output_${NUM_NODES}n${NUM_GPUS}g"
SEED=2829
MAX_STEPS=1
LEARNING_RATE=1.5e-5
DATASET="/p/vast1/$USER/packs/fineweb512_train_slim"
MODEL="/p/vast1/$USER/models/Llama-2-7b-hf"
SEQ_LENGTH=511
PARALLEL_MODE="data_parallel"

# For multi-node, create hostfile if needed
if [ "$NUM_NODES" -gt 1 ] && [ "$NODE_RANK" -eq 0 ]; then
    echo -e "\n${YELLOW}Creating hostfile for multi-node training...${NC}"
    cat > "$BASE/hostfile" << EOF
$(hostname) slots=$NUM_GPUS
EOF
    echo "Add other nodes to $BASE/hostfile in format: hostname slots=$NUM_GPUS"
    echo -n "Press Enter when hostfile is ready..."
    read -r
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo -e "\n${GREEN}Training Configuration:${NC}"
echo "================================"
echo "Nodes: $NUM_NODES (Rank: $NODE_RANK)"
echo "GPUs per node: $NUM_GPUS"
echo "Total GPUs: $TOTAL_GPUS"
if [ "$NUM_NODES" -gt 1 ]; then
    echo "Master: $MASTER_ADDR:$MASTER_PORT"
fi
echo "Accelerate Config: $ACC_CFG"
echo "DeepSpeed Config: $DS_JSON"
echo "Output Directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Max Steps: $MAX_STEPS"
echo "Sequence Length: $SEQ_LENGTH"
echo "================================"

# Ask for confirmation
echo -n -e "\n${YELLOW}Proceed with training? (y/n): ${NC}"
read -r confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Set environment variables for multi-node
if [ "$NUM_NODES" -gt 1 ]; then
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$TOTAL_GPUS
    export RANK=$((NODE_RANK * NUM_GPUS))
    
    echo -e "\n${GREEN}Multi-node environment variables set:${NC}"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "RANK=$RANK"
fi

# Run training
echo -e "\n${GREEN}Starting training...${NC}\n"

TRAIN_SCRIPT="$BASE/train.py"

# Launch command differs for single vs multi-node
if [ "$NUM_NODES" -eq 1 ]; then
    # Single node (possibly multi-GPU)
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
else
    # Multi-node
    accelerate launch \
      --config_file "$ACC_CFG" \
      --deepspeed_config_file "$DS_JSON" \
      --num_machines $NUM_NODES \
      --num_processes $TOTAL_GPUS \
      --machine_rank $NODE_RANK \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
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
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Training completed successfully!${NC}"
else
    echo -e "\n${RED}Training failed. Check the error messages above.${NC}"
    exit 1
fi