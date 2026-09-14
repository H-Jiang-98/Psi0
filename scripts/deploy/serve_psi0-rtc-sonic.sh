#!/bin/bash
#
# Env:
#   PSI_RTC_INIT_PREV=1  seed the FIRST action chunk with the pseudo prev-action the
#                        client ships on its first frame (its current pose encoded as an
#                        action), instead of predicting chunk #0 open-loop. Requires a
#                        client that sends state.init_prev_action (psi_rtc_sonic_client
#                        with the same env var set). Default off.

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"

export CUDA_VISIBLE_DEVICES=0
echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"

serve_psi0_sonic \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}
