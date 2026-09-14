#!/bin/bash
# VARIANT (2026-09-11): identical to finetune-real-sonic-psi0-2.8B.sh except
#   - warm start from the sonic1.1 post-train run (state-as-action-token, state-drop 0.5)
#     .runs/posttrain/sonic1.1.us.flow1000.cosine.lr1.0e-04.b256.gpus8.2609092158 ckpt_40000
#   - --model.state-as-action-token ON (header layout must match the post-trained header:
#     state_pos + obs_strat=None; the old context-token header would silently mismatch)
#   - --model.state-drop-prob=0.0 (was 0.1)
#   - --model.state-null-token ON: fresh param, never substituted at drop 0; kept in the graph
#     by the all-True keep in ActionTransformerModel.forward so DDP does not stall on it
# Usage: sbatch scripts/train/slurm_job.sh scripts/train/psi0/finetune-real-sonic-psi0-2.8B-sonic1.1-sat.sh psi0 <exp>
#   Resume: TS=<yymmddHHMM of existing run dir> sbatch --export=ALL scripts/train/slurm_job.sh <this script> psi0 <exp>
#   (fixes --timestamp and adds --train.resume_from_checkpoint=latest; also makes walltime requeues resume instead of restarting)
#
# psi0 (real sonic) finetune with the VLM unfrozen and a DEEPER, layerwise-conditioned
# action header -- the "2.8B" variant of finetune-real-sonic-psi0.sh (2.13B Qwen VLM
# backbone plus the 12-block action header).
#
# Data, task CLI, action/state dims, resolution, checkpoints and the step schedule are
# UNCHANGED from finetune-real-sonic-psi0.sh. The model knobs below are ported from
# finetune-sonic-neck-zedmini-as-a-baseline-270x480-10x-vlm-lr-combined-dit-layerwise-
# 12blocks.sh, INCLUDING its --model.no-rtc (the base script enables RTC; this one does
# not). Deliberately NOT ported: its 270x480 canvas (this script stays at 240x320) and
# its 100k-step schedule.
#
# TRAINER: --train.name=finetune is REQUIRED here, not a preference. FinetuneTrainer is
#   the only trainer that implements the two features this script depends on:
#     - the frozen CLIP pooled text encoder that feeds combined_temb
#       (compute_pooled_projections, finetune.py:220)
#     - per-component VLM optimizer groups
#       (vlm_trainable_components:416, _vlm_group_of, create_optimizers:443)
#   SonicTrainer has NEITHER -- `grep pooled_text_encoder src/psi/trainers/sonic.py`
#   returns nothing, and its optimizer never reads vision_tower_lr / mm_projector_lr.
#   Run this with --train.name=sonic and combined_temb crashes for want of a
#   pooled_projections vector, while two of the three VLM LRs are silently ignored.
#
# Ported knobs
#   --model.tune-vlm             unfreeze the VLM (was --model.no-tune-vlm). With no
#                                --model.tune-mm-* set, this selects all three components.
#   --model.lang-backbone-lr     1e-6, 10x the 1e-7 default -> group 'lang_backbone'
#   --model.vision-tower-lr      1e-5, 10x the 1e-6 default -> group 'vision_tower'
#   --model.mm-projector-lr      1e-4, 10x the 1e-5 default -> group 'mm_projector'
#                                create_optimizers emits one AdamW group per component
#                                (plus action_header and other at --train.learning_rate)
#                                and logs each group's LR and trainable-param count.
#   --model.gradient-checkpointing
#                                required, not optional: tuning the VLM loads it in fp32
#                                (~32GB/GPU) and batch 16 needs ~71GB without it, which
#                                will not fit an 80GB card.
#   --model.num-blocks           6 -> 12
#   --model.vlm-layer-indices    one VLM layer fused per action block, evenly spread over
#                                the 28-layer Qwen VLM. MUST stay len()==num-blocks:
#                                Psi0Model builds the header with
#                                layerwise_vlm_fusion=(vlm_layer_indices is not None) and
#                                _select_vlm_views stacks one hidden state per index.
#                                finetune_sonic_psi0_config asserts that pairing but
#                                finetune_real_psi0_config (used here) does NOT, so a
#                                mismatch fails late instead of at config time.
#   --model.qk-norm              rms_norm, stabilizes multi-task finetuning
#   --model.combined-temb        SD3-style global adaLN conditioning, fed by the pooled
#                                CLIP vector below. RTC is OFF here, so the timestep is
#                                the plain (B,) form; the ND embedding would equally
#                                accept RTC's (B,Tp) per-token timestep if re-enabled.
#   --model.pooled-*             frozen CLIPTextModelWithProjection over the instruction.
#                                768 must equal CLIP-L's projection_dim -- the trainer
#                                asserts it. pooled_cache_path is RELATIVE, so the cache
#                                lands in project_dir next to run_config.json; the trainer
#                                pre-warms it from every task instruction up front, so
#                                CLIP is never invoked inside the train/eval loop.
#   --model.state-drop-prob      0.8 proprioceptive dropout. NOTE this one comes from the
#                                sibling ...-12blocks_dropout.sh; the plain -12blocks.sh
#                                does not set it (its default is 0.0).
#
# Usage: bash scripts/train/psi0/finetune-real-sonic-psi0-2.8B.sh <task> [exp]

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source "${PSI_VENV:-$([ -d /workspace/.venv-psi ] && echo /workspace/.venv-psi || echo .venv-psi)}/bin/activate"
DATA_ROOT=${DATA_ROOT:-.data/teleop5_round2_wm_train_val_full_cleaned}
TRAIN_IDS=${TRAIN_IDS:-teleop5_lerobot_psix_train/g1}
VAL_IDS=${VAL_IDS:-teleop5_lerobot_psix_val/g1}
# Norm stats: this pack ships meta/stats.json (min/max/q01/q99 for action,
# action.body_token_v1_1, action.neck, observation.state) -- there is no stats_psi0.json
# here, unlike the g1_sonic/psix_sonic packs. Path is resolved relative to the project
# root by resolve_path().
STATS=${STATS:-$DATA_ROOT/$TRAIN_IDS/meta/stats.json}

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 psi0 baseline"
    exit 1
fi

export task="$1"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${2:-$default_exp}

echo "Task: $task"
echo "Experiment name: $exp"

# --- warm-start preflight ----------------------------------------------------
# Split the post-trained checkpoint into the VLM dir + action_header.safetensors that
# the two --model flags below want. ~11 GB, written once and reused; the exporter
# stages into <dir>.partial and renames, so a waiting node never sees a half file.
POSTTRAIN_RUN="${POSTTRAIN_RUN:-.runs/posttrain/sonic1.1.us.flow1000.cosine.lr1.0e-04.b256.gpus8.2609092158}"
CKPT_STEP="${CKPT_STEP:-40000}"
CKPT_DIR="$POSTTRAIN_RUN/checkpoints/ckpt_$CKPT_STEP"
INIT_DIR="$POSTTRAIN_RUN/posttrained/ckpt_$CKPT_STEP"

if [ ! -s "$INIT_DIR/model.safetensors" ] || [ ! -s "$INIT_DIR/action_header.safetensors" ]; then
    [ -s "$CKPT_DIR/model.safetensors" ] || { echo "FATAL: missing $CKPT_DIR/model.safetensors" >&2; exit 1; }
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Exporting post-trained weights: $CKPT_DIR -> $INIT_DIR"
        python3 scripts/export_psi0_ckpt.py "$CKPT_DIR" "$INIT_DIR" \
            || { echo "FATAL: export_psi0_ckpt.py failed" >&2; exit 1; }
    else
        echo "Waiting for node 0 to export $INIT_DIR ..."
        for _ in $(seq 1 180); do
            [ -s "$INIT_DIR/model.safetensors" ] && [ -s "$INIT_DIR/action_header.safetensors" ] && break
            sleep 10
        done
    fi
fi
for f in config.json model.safetensors action_header.safetensors; do
    [ -s "$INIT_DIR/$f" ] \
        || { echo "FATAL: $INIT_DIR/$f missing (rerun: python3 scripts/export_psi0_ckpt.py $CKPT_DIR $INIT_DIR)" >&2; exit 1; }
done
echo "Warm start from $INIT_DIR (VLM + action header, ckpt_$CKPT_STEP)"

args="
finetune_sonic_psi0_config \
--seed=292285 \
--exp=$exp \
${TS:+--timestamp=$TS --train.resume_from_checkpoint=latest} \
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
--train.validation_steps=5000 \
--train.val_num_batches=100 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$DATA_ROOT \
--data.train_repo_ids $TRAIN_IDS \
--data.val_repo_ids $VAL_IDS \
--data.transform.repack.action-keys action.body_token_v1_1 action[:14] action.neck \
--data.transform.repack.dataset-name=teleop5round2full \
--data.transform.repack.pad-action-dim=80 \
--data.transform.repack.pad-state-dim=45 \
--data.transform.repack.instruction-key=task_description \
--data.transform.field.stat-path=$STATS \
--data.transform.field.stat-action-keys action.body_token_v1_1 action[:14] action.neck \
--data.transform.field.stat-state-keys observation.state \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=80 \
--data.transform.field.pad-state-dim=45 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 270 480 \
--data.transform.model.center_crop.size 270 480 \
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
--model.state-drop-prob=0.0 \
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
