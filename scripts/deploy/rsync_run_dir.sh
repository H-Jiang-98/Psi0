#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-nebula100}"
REMOTE_PROJECT_DIR="${2:-/hfm/songlin/psi0}"
RUN_DIR="${3:-.runs/psix_finetune/mem.rss26r.flow1000.cosine.lr5.0e-05.b128.gpus8.2606090813}"
CKPT_STEP="${4:-40000}"
LOCAL_BASE="${5:-$(pwd)}"


# Extract the run subfolder (e.g. psix_finetune) from the parent of RUN_DIR.
SUBFOLDER="$(basename "$(dirname "${RUN_DIR}")")"

DEST="${LOCAL_BASE}/.runs/${SUBFOLDER}/"
mkdir -p "${DEST}"

# Max number of rsync attempts before giving up (override via MAX_RETRIES env var).
MAX_RETRIES="${MAX_RETRIES:-50}"

# Only download the target checkpoint: include checkpoints/ckpt_${CKPT_STEP}, exclude all others.
#
# Resume support:
#   --append-verify resumes large files by appending, then checksums the whole
#     file to guarantee integrity (checkpoints are immutable, so this is safe).
RSYNC_CMD=(
    rsync -avz --progress
    --append-verify
    --timeout=60
    --exclude='wandb/'
    --exclude='checkpoints/*/random_states_*.pkl'
    --exclude='checkpoints/*/optimizer.bin'
    --exclude='checkpoints/*/scheduler.bin'
    --include='checkpoints/'
    --include="checkpoints/ckpt_${CKPT_STEP}/***"
    --exclude='checkpoints/*'
    "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/${RUN_DIR}"
    "${DEST}"
)

echo "Running:"
printf '%q ' "${RSYNC_CMD[@]}"
echo

# Retry on failure (e.g. dropped connection) until rsync succeeds or we run out
# of attempts. Each retry resumes from where the previous one left off.
attempt=1
while true; do
    rc=0
    "${RSYNC_CMD[@]}" || rc=$?
    if (( rc == 0 )); then
        echo "rsync completed successfully."
        break
    fi

    if (( attempt >= MAX_RETRIES )); then
        echo "rsync failed after ${attempt} attempts (last exit code ${rc}); giving up." >&2
        exit "${rc}"
    fi

    echo "rsync exited with code ${rc} (attempt ${attempt}/${MAX_RETRIES}); retrying in 5s..." >&2
    attempt=$(( attempt + 1 ))
    sleep 5
done
