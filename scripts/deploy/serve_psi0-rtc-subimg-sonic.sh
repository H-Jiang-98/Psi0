#!/bin/bash

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"

export CUDA_VISIBLE_DEVICES=0
echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"

serve_psi0_sonic_subgoal \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}
