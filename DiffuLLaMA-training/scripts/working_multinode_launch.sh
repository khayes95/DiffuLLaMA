#!/bin/bash
source /usr/workspace/$USER/x86_miniconda/etc/profile.d/conda.sh
conda activate diffullama

export RANK=${FLUX_TASK_RANK}
export LOCAL_RANK=${FLUX_TASK_LOCAL_ID}
export WORLD_SIZE=${FLUX_JOB_SIZE}

echo "[DEBUG] Rank $RANK starting, FLUX_JOB_ID=${FLUX_JOB_ID}"

# Disable FusedAdam compilation
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=1

if [ "$RANK" == "0" ]; then
    export MASTER_ADDR=$(hostname -I | awk '{print $1}')
    echo "$MASTER_ADDR" > /usr/workspace/$USER/master_${FLUX_JOB_ID}.txt
    echo "[RANK 0] Wrote MASTER_ADDR=$MASTER_ADDR"
    sync
else
    for i in {1..120}; do
        if [ -f /usr/workspace/$USER/master_${FLUX_JOB_ID}.txt ]; then
            export MASTER_ADDR=$(cat /usr/workspace/$USER/master_${FLUX_JOB_ID}.txt)
            echo "[RANK $RANK] Got MASTER_ADDR=$MASTER_ADDR after $i attempts"
            break
        fi
        [ $((i % 10)) -eq 0 ] && echo "[RANK $RANK] Waiting... attempt $i/120"
        sleep 1
    done
    
    if [ -z "$MASTER_ADDR" ]; then
        echo "[ERROR] Rank $RANK failed to get MASTER_ADDR"
        exit 1
    fi
fi

export MASTER_PORT=30500
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export ROCR_VISIBLE_DEVICES=${FLUX_TASK_LOCAL_ID}

echo "[RANK $RANK] Final env: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

cd /p/vast1/$USER/DiffuLLaMA/DiffuLLaMA-training
python3 train.py "$@"
