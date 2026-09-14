#!/usr/bin/env bash
# Stage the psi0 posttrain packs onto node-local NVMe.
#
# Why: the packs hold 88,868 files (17,468 parquet + 34,936 videos + 36,456 images).
# With 24 ranks x 8 dataloader workers randomly opening/closing video files, beegfs's
# metadata server saturates and the client wedges in uninterruptible D state
# (FhgfsOpsRemoting_closefileEx) -- jobs 1370/1372. Node-local ext4 has no such limit.
#
# Only what the posttrain recipe reads is copied (~50G): meta, data (parquet) and the
# `observation.images.egocentric` videos. The unused `_wm` videos (119G) and subgoal
# images stay on beegfs, reachable via symlink so the pack is still complete.
#
# Run as root on each node (files owned by root => they do not count against the
# user's node-local quota).
#
# Usage: sudo bash scripts/data/stage_pack_local.sh [pack ...]
#   No args -> the default pair below. The stage is shared by every dataset and user
#   on the node, so it is additive: staging one pack never removes another. What is
#   already there is listed at the end so stale packs are visible; delete those by
#   hand, and only after checking `squeue` that no running job still reads them.
set -euo pipefail
SRC=${SRC:-/mnt/beegfs/shared/hfm/data}
DST=${DST:-/var/lib/h100-job-cache/data}
KEEP=${KEEP:-observation.images.egocentric}

# rsync is not installed on every node (node6), so fall back to cp.
if command -v rsync >/dev/null 2>&1; then
    copy_tree() { mkdir -p "$2"; rsync -a "$1/" "$2/"; }
else
    copy_tree() { mkdir -p "$2"; cp -a "$1/." "$2/"; }
fi

PACKS=("$@")
[ ${#PACKS[@]} -eq 0 ] && PACKS=(psix_sonic_v1_train psix_sonic_v1_val)

mkdir -p "$DST"
for pack in "${PACKS[@]}"; do
    [ -d "$SRC/$pack/data" ] || { echo "[$(hostname)] SKIP $pack: no $SRC/$pack/data" >&2; continue; }
    if [ -d "$DST/$pack/data" ]; then
        echo "[$(hostname)] $pack already staged, refreshing"
    fi
    echo "[$(hostname)] staging $pack ..."
    mkdir -p "$DST/$pack/videos"
    copy_tree "$SRC/$pack/meta" "$DST/$pack/meta"
    copy_tree "$SRC/$pack/data" "$DST/$pack/data"
    # only the video key the posttrain recipe reads
    while IFS= read -r d; do
        rel=${d#"$SRC/$pack/videos/"}
        copy_tree "$d" "$DST/$pack/videos/$rel"
    done < <(find "$SRC/$pack/videos" -mindepth 2 -maxdepth 2 -type d -name "$KEEP")
    # unused-but-declared assets: reachable without occupying local disk
    for ch in "$SRC/$pack/videos"/chunk-*; do
        [ -d "$ch" ] || continue
        c=$(basename "$ch")
        for other in "$ch"/*/; do
            k=$(basename "$other")
            [ "$k" = "$KEEP" ] && continue
            mkdir -p "$DST/$pack/videos/$c"
            ln -sfn "$other" "$DST/$pack/videos/$c/$k"
        done
    done
    [ -d "$SRC/$pack/images" ] && ln -sfn "$SRC/$pack/images" "$DST/$pack/images" || true
    echo "[$(hostname)] $pack -> $(du -sh --exclude=images "$DST/$pack" 2>/dev/null | cut -f1)"
done
chmod -R a+rX "$DST"
echo "[$(hostname)] stage now holds:"
for d in "$DST"/*/; do
    [ -d "$d" ] || continue
    echo "[$(hostname)]   $(basename "$d")  $(du -sh --exclude=images "$d" 2>/dev/null | cut -f1)"
done
echo "[$(hostname)] STAGE_DONE $(df -h /var/lib | tail -1)"
