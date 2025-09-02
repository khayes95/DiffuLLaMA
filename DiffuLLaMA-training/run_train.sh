#!/usr/bin/env bash
set -euo pipefail

# --- ROCm that matches torch.version.hip (you showed 6.0.x) ---
module purge
module load rocm/6.0.3

# --- Use conda GCC/G++ >= 9 for any JIT builds ---
# (install if missing: conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12)
export CC="$(which x86_64-conda-linux-gnu-cc || true)"
export CXX="$(which x86_64-conda-linux-gnu-c++ || true)"

# --- Keep ALL caches off $HOME ---
BASE="/p/vast1/$USER"
export HF_HOME="$BASE/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$BASE/torch"
export XDG_CACHE_HOME="$BASE/xdg"
export TORCH_EXTENSIONS_DIR="$BASE/caches/torch_ext"
export DEEPSPEED_JIT_TMP="$TORCH_EXTENSIONS_DIR"
export WANDB_DIR="$BASE/wandb"
export PIP_CACHE_DIR="$BASE/pip"

# Belt-and-suspenders: tell DeepSpeed not to build fused adam
export DS_BUILD_FUSED_ADAM=0

# (Optional) show what compilers/ROCm we actually picked
$CC --version || true
$CXX --version || true
hipcc --version || true

# --- Config paths (you already created these) ---
CFGROOT="/usr/workspace/$USER/ds_cfgs"
ACC_CFG="$CFGROOT/single_node_ds.yaml"
DS_JSON="$CFGROOT/zero3_gpu_only.json"   # make sure this uses "optimizer": {"type": "Adam", "torch_adam": true}

# --- Launch exactly like before ---
accelerate launch \
  --config_file "$ACC_CFG" \
  --deepspeed_config_file "$DS_JSON" \
  train.py \
    --batch-size 1 \
    --gradient-accumulate-every 1 \
    --output-dir "$BASE/test_output" \
    --seed 2829 \
    --max-train-steps 5 \
    --learning-rate 1.5e-5 \
    --dataset /p/vast1/pretrain/datasets/diffusion/dolma_v1-6_sample_llama2_pkds \
    --model "/p/vast1/$USER/models/Llama-2-7b-hf" \
    --seq-length 512 \
    --parallel_mode data_parallel
