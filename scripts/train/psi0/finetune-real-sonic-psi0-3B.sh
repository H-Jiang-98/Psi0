#!/bin/bash
# psi0 (real sonic) finetune with the VLM unfrozen and a DEEPER, layerwise-conditioned
# action header -- the "3B" variant of finetune-real-sonic-psi0.sh (~2.13B VLM backbone
# + ~1B 12-block header).
#
# Data, task CLI, action/state dims, resolution, checkpoints, RTC and the step schedule
# are UNCHANGED from finetune-real-sonic-psi0.sh. Only the model-side knobs below are
# ported from finetune-sonic-neck-zedmini-psi0.sh (the 270x480-10x-vlm-lr-combined-dit-
# layerwise-12blocks baseline). Deliberately NOT ported: its 270x480 canvas (this script
# stays at 240x320), --model.no-rtc, and its 100k-step schedule.
#
# Ported knobs
#   --model.tune-vlm             unfreeze the VLM (was --model.no-tune-vlm)
#   --model.lang-backbone-lr     1e-6, 10x the 1e-7 default
#   --model.vision-tower-lr      1e-5, 10x the 1e-6 default   [see CAVEAT 1]
#   --model.mm-projector-lr      1e-4, 10x the 1e-5 default   [see CAVEAT 1]
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
#   --model.combined-temb        SD3-style global adaLN conditioning   [see CAVEAT 2]
#   --model.pooled-*             frozen CLIP ViT-L/14 pooled instruction embedding
#                                (768 = CLIP-L text projection dim)   [see CAVEAT 2]
#   --model.state-drop-prob      0.8 proprioceptive dropout
#
# Trainer: --train.name=finetune (the base script used sonic). Trainer.instantiate maps
#   the name to psi.trainers.<name>.<Name>Trainer, and the two modules are byte-identical
#   apart from the class name and ONE thing: the validation L1 breakdown. SonicTrainer
#   splits the denormed error at [64] into latent_action / hand_joints; FinetuneTrainer
#   splits at [14,28,31,32,33,34,35] into err_l1_{hand_joints,arm_joints,torso_rpy,height,
#   vx,vy,vyaw,target_yaw}. Optimizer groups, VLM handling, the model call and the loss
#   are the same either way, so this only changes which val metrics reach W&B. With
#   action-dim=78 the last np.split bucket is dims 35:78, so `err_l1_target_yaw` actually
#   covers 43 dims -- read that one with care (the sonic labels were equally approximate).
#
# CAVEAT 1 -- vision-tower-lr / mm-projector-lr are INERT on this training path.
#   FinetuneTrainer.create_optimizers (src/psi/trainers/finetune.py:162-202, identical
#   in sonic.py) sorts parameters into exactly three groups -- action_header, vlm_model,
#   other -- and gives the whole vlm_model group a single LR: lang_backbone_lr. It never
#   reads vision_tower_lr or mm_projector_lr. Only Qwen3vlMixin (qwen3vl_mixin.py:299-439,
#   used by the pretrain configs) builds per-component groups. So in practice the ENTIRE
#   VLM -- vision tower, projector and LLM alike -- trains at 1e-6 here. The two flags are
#   kept for parity with the baseline and to be correct if the trainer gains per-component
#   groups later. Note the zedmini header's claim that each component gets its own group
#   "(FinetuneTrainer.vlm_trainable_components)" is stale: that attribute exists nowhere
#   in the repo.
#
# CAVEAT 2 -- combined-temb will FAIL on the first forward pass as the code stands.
#   With combined_temb=True the header does
#       temb = self.time_ins_embed(timestep, pooled_projections)   (psi0.py:1232)
#   and CombinedTimestepTextProjEmbeddingsND.forward immediately dereferences that arg
#   (self.text_embedder(pooled_projection), then pooled_projection.dtype, psi0.py:99-100).
#   But nothing on the training path ever supplies it: FinetuneTrainer's model(...) call
#   (finetune.py:573-583) passes no pooled_projections, no data transform or collator adds
#   the key, and the model does not build a CLIP encoder of its own. Repo-wide,
#   pooled_projections is produced ONLY by src/psi/deploy/serve_psi0_sonic.py:569, i.e.
#   deploy-side. The four --model.pooled-* flags are likewise read only by that deploy
#   file (serve_psi0_sonic.py:415-428) and are inert config during training.
#   => Training-side wiring (build the frozen CLIP encoder, embed the instruction, put
#      pooled_projections in the batch, thread it through compute_loss) must be added
#      before this script can run. Until then, drop --model.combined-temb and the four
#      --model.pooled-* lines to train, or add the wiring.
#   RTC note: this script keeps --model.rtc, and the ND embedding does handle the (B,Tp)
#   per-token timestep that RTC produces, so the two are compatible once wired.
#
# Usage: bash scripts/train/psi0/finetune-real-sonic-psi0-3B.sh <task> [exp]

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
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=/hfm/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=/hfm/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=78 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=43 \
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
--model.state-drop-prob=0.8 \
--model.pooled-text-encoder=clip \
--model.pooled-text-encoder-path=openai/clip-vit-large-patch14 \
--model.pooled-projection-dim=768 \
--model.pooled-cache-path=clip_pooled_cache.pt \
--model.rtc \
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
