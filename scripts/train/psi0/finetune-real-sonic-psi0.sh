#!/bin/bash

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw pick-toys"
    exit 1
fi

export task="$1"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${2:-$default_exp}

echo "Task: $task"
echo "Experiment name: $exp"

# --- warm start ---------------------------------------------------------------
# Released Psi-0 SONIC post-train (VLM + action header, 12 blocks, state token, combined temb).
# The knobs below assume THIS header shape; the old AMO checkpoints (postpre.1by1.pad36...,
# 6 blocks, no state token) do not fit them. Override with INIT_DIR=... if staged elsewhere.
INIT_DIR="${INIT_DIR:-/hfm/cache/checkpoints/psi0/postpre.sonic1.0.unifolm.2609092156.40k}"
for f in config.json model.safetensors action_header.safetensors; do
    [ -s "$INIT_DIR/$f" ] || { echo "FATAL: $INIT_DIR/$f missing" >&2; exit 1; }
done
echo "Warm start from $INIT_DIR (VLM + action header)"

# --- robustness knobs (ported from finetune-sonic-psi-dream-baseline.sh) --------
STATE_DROP_PROB="${STATE_DROP_PROB:-0.1}"     # per-sample state drop -> learned null token; 0 disables
STATE_JITTER="${STATE_JITTER:-10}"            # frames; 0 disables the temporal state aug
STATE_JITTER_PROB="${STATE_JITTER_PROB:-0.5}"
STATE_NOISE_STD="${STATE_NOISE_STD:-0.05}"    # in normalized [-1,1] units; 0 disables
VIEW_AUG_MIN_SCALE="${VIEW_AUG_MIN_SCALE:-0.85}"
VIEW_AUG_PROB="${VIEW_AUG_PROB:-1.0}"
echo "State aug: drop=${STATE_DROP_PROB} (learned null token) jitter=+-${STATE_JITTER}f p=${STATE_JITTER_PROB} noise=${STATE_NOISE_STD}; view aug: min_scale=${VIEW_AUG_MIN_SCALE} p=${VIEW_AUG_PROB}"

# --train.name=finetune is REQUIRED: only FinetuneTrainer implements the frozen CLIP pooled
# text encoder behind --model.combined-temb, the per-component VLM optimizer groups
# (--model.*-lr) and the state_drop_frac logging.
args="
finetune_real_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=5000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=/hfm/data/sonic/lerobot \
--data.train_repo_ids=$task \
--data.transform.repack.pad-action-dim=80 \
--data.transform.repack.pad-state-dim=45 \
--data.transform.repack.state-temporal-jitter=$STATE_JITTER \
--data.transform.repack.state-temporal-jitter-prob=$STATE_JITTER_PROB \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.state-noise-std=$STATE_NOISE_STD \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=80 \
--data.transform.field.pad-state-dim=45 \
--data.transform.model.img-aug \
--data.transform.model.view-aug \
--data.transform.model.view-aug-min-scale=$VIEW_AUG_MIN_SCALE \
--data.transform.model.view-aug-prob=$VIEW_AUG_PROB \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=$INIT_DIR \
--model.pretrained-action-header-path=$INIT_DIR \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=80 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=45 \
--model.dropout=0.0 \
--model.state-feature-dropout=0.0 \
--model.view_feature_dim=2048 \
--model.tune-vlm \
--model.lang-backbone-lr=1e-6 \
--model.vision-tower-lr=1e-5 \
--model.mm-projector-lr=1e-4 \
--model.gradient-checkpointing \
--model.no-use_film \
--model.qk-norm=rms_norm \
--model.combined-temb \
--model.num-blocks=12 \
--model.vlm-layer-indices 3 5 8 10 12 14 17 19 21 23 26 28 \
--model.state-drop-prob=$STATE_DROP_PROB \
--model.state-as-action-token \
--model.state-null-token \
--model.pooled-text-encoder=clip \
--model.pooled-text-encoder-path=openai/clip-vit-large-patch14 \
--model.pooled-projection-dim=768 \
--model.pooled-cache-path=clip_pooled_cache.pt \
--model.no-rtc \
--model.max-delay=8
"

# Find an available TCP port starting at 29500 and increment until a free port is found.
find_free_port() {
    start_port=${1:-29500}
    port=${start_port}
    while true; do
        # Use Python socket bind test; binding to 0.0.0.0:port will fail if port is in use.
        CHECK_PORT=${port} python - <<'PY'
import os,sys,socket
port = int(os.environ.get('CHECK_PORT','0'))
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('0.0.0.0', port))
    sock.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
        if [ $? -eq 0 ]; then
            echo ${port}
            return 0
        fi
        port=$((port+1))
        # avoid infinite loop in pathological cases
        if [ ${port} -gt $((start_port+1000)) ]; then
            echo "Failed to find free port after 1000 attempts" >&2
            return 1
        fi
    done
}

MAIN_PORT=$(find_free_port 29500)
if [ -z "${MAIN_PORT}" ]; then
    echo "Could not find free main process port, aborting." >&2
    exit 1
fi

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=${MAIN_PORT} scripts/train.py \
    ${args}