#!/usr/bin/env bash
# Download all 33 SIMPLE task zips from the USC-PSI-Lab/psi-data HF dataset
# and unzip each into $LOCAL_DIR/<task>/ with no redundant nesting.
#
# Env toggles:
#   LOCAL_DIR=...  override the destination (default: /mnt/beegfs/shared/hfm/data/simple-33)
#   KEEP_ZIP=0     remove each .zip after a successful extract (default: keep)
#   FORCE=1        re-extract even if $LOCAL_DIR/<task>/ already exists
set -uo pipefail

REPO="USC-PSI-Lab/psi-data"
LOCAL_DIR="${LOCAL_DIR:-/hfm/data/simple-33}"
KEEP_ZIP="${KEEP_ZIP:-1}"
FORCE="${FORCE:-0}"

SIMPLE_33_TASKS=(
  "G1WholebodyBendHandoverTeleop-v0"
  "G1WholebodyBendPickAndPlaceTeleop-v0"
  "G1WholebodyBendPickMP-v0"
  "G1WholebodyBendPickPlaceMP-v0"
  "G1WholebodyBendPickPlaceOnSofaMP-v0"
  "G1WholebodyBendPickTeleop-v0"
  "G1WholebodyCloseDoorTeleop-v0"
  "G1WholebodyHandoverTeleop-v0"
  "G1WholebodyLocomotionPickBetweenTablesMP-v0"
  "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
  "G1WholebodyOpenFaucetTeleop-v0"
  "G1WholebodyOpenOvenTeleop-v0"
  "G1WholebodyOpenTrashCanTeleop-v0"
  "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0"
  "G1WholebodyPickBendPlaceMP-v0"
  "G1WholebodyPickPlaceMP-v0"
  "G1WholebodyPushOfficeChairTeleop-v0"
  "G1WholebodyTabletopGraspMP-v0"
  "G1WholebodyTabletopHandoverMP-v0"
  "G1WholebodyTurnPickMP-v0"
  "G1WholebodyTurnXMoveBendHandoverMP-v0"
  "G1WholebodyTurnXMoveBendPickMP-v0"
  "G1WholebodyTurnXMoveHandoverMP-v0"
  "G1WholebodyTurnYMoveBendPickMP-v0"
  "G1WholebodyTurnYMovePickMP-v0"
  "G1WholebodyXMoveBendPickMP-v0"
  "G1WholebodyXMoveBendPickTeleop-v0"
  "G1WholebodyXMoveHandoverMP-v0"
  "G1WholebodyXMovePickMP-v0"
  "G1WholebodyXMovePickTeleop-v0"
  "G1WholebodyYMoveBendPickMP-v0"
  "G1WholebodyYMoveHandoverMP-v0"
  "G1WholebodyYMovePickMP-v0"
)

command -v unzip >/dev/null 2>&1 || { echo "error: 'unzip' not found in PATH" >&2; exit 1; }
command -v hf >/dev/null 2>&1 || { echo "error: 'hf' not found in PATH (activate the venv first)" >&2; exit 1; }
mkdir -p "$LOCAL_DIR"

total=${#SIMPLE_33_TASKS[@]}
failed=()

# Extract $zip into $dest, stripping a single redundant leading "<task>/" dir
# so we never end up with $dest/<task>/<task>/...
extract_flat() {
  local zip="$1" dest="$2" task="$3"
  local stage
  stage="$(mktemp -d "${LOCAL_DIR}/.stage_${task}.XXXXXX")" || return 1

  if ! unzip -q -o "$zip" -d "$stage"; then
    rm -rf "$stage"
    return 1
  fi

  # If the archive's sole top-level entry is a dir named exactly "<task>",
  # promote its contents; otherwise keep the archive layout as-is.
  local root="$stage"
  shopt -s nullglob dotglob
  local entries=("$stage"/*)
  shopt -u nullglob dotglob
  if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ] \
     && [ "$(basename "${entries[0]}")" = "$task" ]; then
    root="${entries[0]}"
  fi

  rm -rf "$dest"
  mv "$root" "$dest"
  rm -rf "$stage"   # no-op if we moved $stage itself
  return 0
}

for i in "${!SIMPLE_33_TASKS[@]}"; do
  task="${SIMPLE_33_TASKS[$i]}"
  n=$((i + 1))
  zip_path="${LOCAL_DIR}/simple/${task}.zip"
  dest="${LOCAL_DIR}/${task}"

  if [ "$FORCE" != "1" ] && [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "==> [$n/$total] ${task}: already extracted, skipping (FORCE=1 to redo)"
    continue
  fi

  echo "==> [$n/$total] downloading simple/${task}.zip"
  if ! hf download "$REPO" "simple/${task}.zip" \
      --local-dir "$LOCAL_DIR" \
      --repo-type=dataset; then
    echo "    FAILED (download): ${task}" >&2
    failed+=("$task")
    continue
  fi

  echo "    unzipping -> ${dest}"
  if ! extract_flat "$zip_path" "$dest" "$task"; then
    echo "    FAILED (unzip): ${task}" >&2
    failed+=("$task")
    continue
  fi

  [ "$KEEP_ZIP" = "1" ] || rm -f "$zip_path"
  echo "    ok: ${task}"
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All $total tasks downloaded and extracted under $LOCAL_DIR/<task>/"
else
  echo "${#failed[@]}/$total failed:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  echo "Re-run this script to retry (hf download resumes from cache)." >&2
  exit 1
fi
