# LLNL TUO implementation with Accelerate 

## Quick Start

### 1. Get Allocation

```bash
cd ~/DiffuLLaMA

# sample for 2 nodes with 1 GPU each
flux alloc -N 2 -n 2 -g 2 -t 60m

# sample for 2 nodes with 2 GPUs each (so 4 GPUs total)
flux alloc -N 2 -g 2 -t 60m

# for 3 nodes with 1 GPU each
flux alloc -N 3 -n 3 -g 3 -t 60m
```

### 2. Verify Node Health (not needed but can be a good sanity check during debugging)

```bash
flux run -N 2 -n 2 --label-io hostname
# MUST see responses from all nodes. If any node doesn't respond, exit and get new alloc.
```

### 3. Initial Setup (This only needs to be run once for the alloc)

Create the launch script and set up cache redirection. The launch script handles node coordination using a shared file for MASTER_ADDR discovery since flux doesn't propagate environment variables between nodes properly. It waits up to 120 seconds for filesystem sync between nodes.

```bash
# Create the launch script (save as working_multinode_launch.sh in /usr/workspace/$USER/)
# This script is available in the scripts/ directory of this repo
chmod +x /usr/workspace/$USER/working_multinode_launch.sh
```

Make sure to set up cache redirection to avoid disk quota issues (may not be necessary):

```bash
rm -rf ~/.cache/torch_extensions
mkdir -p /usr/workspace/$USER/torch_extensions
ln -s /usr/workspace/$USER/torch_extensions ~/.cache/torch_extensions
```

### 4. Run Training

#### 2 nodes with 1 GPU each (so 2 GPUs total)

```bash
flux run -N 2 -n 2 -g 1 --label-io bash /usr/workspace/$USER/working_multinode_launch.sh \
    --batch-size 1 \
    --gradient-accumulate-every 1 \
    --output-dir /usr/workspace/$USER/test_output_multinode \
    --seed 2829 \
    --max-train-steps 5 \
    --learning-rate 1.5e-05 \
    --dataset /p/vast1/$USER/packs/fineweb512_train_slim \
    --model /p/vast1/$USER/models/Llama-2-7b-hf \
    --seq-length 511 \
    --parallel_mode data_parallel
```

#### 2 nodes with 2 GPUs each (4 GPUs total)

```bash
flux run -N 2 -n 4 --tasks-per-node=2 -g 1 --label-io bash /usr/workspace/$USER/working_multinode_launch.sh \
    --batch-size 1 \
    --gradient-accumulate-every 1 \
    --output-dir /usr/workspace/$USER/test_output_multinode \
    --seed 2829 \
    --max-train-steps 5 \
    --learning-rate 1.5e-05 \
    --dataset /p/vast1/$USER/packs/fineweb512_train_slim \
    --model /p/vast1/$USER/models/Llama-2-7b-hf \
    --seq-length 511 \
    --parallel_mode data_parallel
```

## Expected Output

it will show:
- Both ranks starting and finding MASTER_ADDR
    - Normally several attempts 20-60 seconds for 2 nodes
- Model loading on all GPUs
- Training progress bar with loss values
- "Training Finished" and "Saving Finished" messages from all ranks
    - saved model



# DiffuLLaMA-training
### from Monte
Fork of https://github.com/HKUNLP/DiffuLLaMA for LLNL/Nexus

## Quickstart

1. Use CUDA 12.4. 
- On Nexus:
    ```
    module load cuda/12.4.1
    module load gcc/11.2.0
    ```

2. Use Python 3.10.
    ```
    conda create -n diffullama python=3.10
    ```

3. Use Pytorch 2.4.
    ```
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    ```

4. Build Flash Attention v2.7 locally because the prebuilt wheels use a different version of glibc than we have available on Nexus (2.32 vs 2.28). Even with Ninja and 32 cores it takes 30 minutes. You need ~8GB of memory per worker.
    ```
    cd .. && git clone git@github.com:Dao-AILab/flash-attention.git && cd flash-attention
    git checkout v2.7.4
    export TORCH_CUDA_ARCH_LIST="8.6;9.0"   # A5000/A6000 + H100
    pip install ninja
    python setup.py install
    ```

4. Install the rest of the dependencies.
    ```
    pip install -r requirements.txt
    ```



Original README:
--------

# Overview
Training scripts for training large diffusion language models (e.g., Llama 7B).


## Installation
The code is tested on Python 3.10.12, with several 4xgh200 nodes on a slurm based cluster with The NVIDIA container image for PyTorch, release 24.07, available on [NGC](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html). Since, the container includes several packages not listed in the requirements.txt, we include a pip-freeze of the package list for reference. We only implement the flash-attention version for LLama models. For code without flash-attention, please refer to the Llama factory diffusion adaption code in our repo.

```bash
pip install -r requirements.txt
```

## Data processing

We borrow data processing and dataloaders from TinyLlama. Please preprocess and tokenize the dataset following [them](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)

## Usage
For multi node runs, we prepare a list of node names in the cluster written in hostnames.txt that the base node can login into via ssh. 
```python
bash multi_node.sh
```
For single node runs, we do not need a list of node names. We directly use the accelerate command in. Note that the command uses the number of nodes to identify the world size, which can be set based on the machine. 
```python
bash run_distributed.sh
```





## Acknowledgements
This work is built on top of the following papers/repositories:
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)
- [Yunchang](https://github.com/feifeibear/long-context-attention)
- [EasyContext](https://github.com/jzhang38/EasyContext/tree/main)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)


