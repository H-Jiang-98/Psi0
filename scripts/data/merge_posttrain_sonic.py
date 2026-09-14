#!/usr/bin/env python
"""Build the psi0 post-training pack `.data/psix_sonic_v1_{train,val}` end to end.

This is the whole conversion, reproducibly, in the order it was actually done:

  1. merge      g1-only sonic-v1 sources -> one LeRobot pack (scripts/data/merge_sonic_v1.py)
  2. quantiles  recompute q01/q99 MASK-AWARE and rewrite the stats files
  3. video      normalise every mp4 to one resolution (scripts/data/normalize_pack_resolution.py)
  4. prune      drop assets nothing in the psi0 path reads, and UNDECLARE them
  5. names      write the confirmed joint names into meta/info.json

Run it all, or pick stages:

    python scripts/data/merge_posttrain_sonic.py                 # everything
    python scripts/data/merge_posttrain_sonic.py --stages 2,5    # just re-do stats + names
    python scripts/data/merge_posttrain_sonic.py --dry-run

Why each stage exists
---------------------
1. merge_sonic_v1.py's SOURCES_TRAIN also lists he_h1_tr / he_h1_va. This pack is g1
   only, so they are filtered out here rather than by editing that file.

2. THE ONE REAL FIX. merge_sonic_v1.merge_stats() masks count/mean/std/min/max by
   per-dim validity, but computes q01/q99 from a pooled raw subsample that still
   contains the zero padding. Only zmsonic has a neck, so ~93% of the pooled neck rows
   are exactly 0.0 and the quantiles collapse toward zero:

       state[43]  valid-rows-only -0.4847/+0.6750 (width 1.160)
                  as shipped      -0.0383/+0.3574 (width 0.396)   2.9x too narrow

   Under `bounds_q99` the normaliser maps [q01,q99] -> [-1,1] and CLIPS, so a bound
   that tight flattens most real neck motion to +/-1. Stage 2 recomputes the quantiles
   using only rows where the dim is genuinely present.

3. merge_sonic_v1 copies mp4s byte-for-byte, so a merged pack inherits mixed
   resolutions: zmsonic is 384x672 (aspect 1.75), everything else 480x640 (1.33). The
   training transform is v2.Resize((270,480)), which does NOT preserve aspect, so mixed
   input gets different geometric distortion -- and zmsonic is 100% of val.

4. The psi0 path reads image_keys=["observation.images.egocentric"] only
   (transform_psi0_sonic.py) and has no subgoal handling, so the subgoal images (2.5G)
   and the egocentric_wm track (112G) are dead weight. NOTE: lerobot's
   _get_query_timestamps() iterates EVERY declared video key, so deleting the files
   without also removing them from info.json/modality.json breaks __getitem__.

5. observation.state is one opaque `joint_positions [0:45]` span in modality.json, so
   nothing downstream can recover the joint order. The names below were confirmed by
   driving g1_body29_hand14.urdf in viser; the neck pair (not in that URDF) was settled
   by head-camera phase correlation -- state[44] pans the image vertically (r=-0.75)
   and not horizontally (r=0.01), so [43]=yaw, [44]=pitch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/beegfs/scratch/songlinwei/psi0")
SCRIPTS = ROOT / "scripts/data"
DROP_SOURCES = {"he_h1_tr", "he_h1_va"}      # g1-only pack
VIDEO_H, VIDEO_W = 480, 640
KEEP_VIDEO = "observation.images.egocentric"
DROP_VIDEO = "observation.images.egocentric_wm"
DROP_IMAGE_DIR = "images"

# ---------------------------------------------------------------- joint names
HAND = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    # Both hands are thumb -> MIDDLE -> INDEX. An earlier revision had the right hand
    # as index-before-middle; that was wrong and showed up in viser as the right
    # index/middle fingers moving as each other.
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
ARM = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEG = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
NECK = ["neck_yaw_joint", "neck_pitch_joint"]

STATE_NAMES = HAND + ARM + LEG + WAIST + NECK                      # 45
ACTION_NAMES = (HAND + ARM + ["torso_roll", "torso_pitch", "torso_yaw"]
                + ["base_height", "base_vx", "base_vy", "base_vyaw", "target_yaw"])   # 36
MASK_NAMES = ["body_token_%02d" % i for i in range(64)] + HAND + NECK                 # 80
assert (len(STATE_NAMES), len(ACTION_NAMES), len(MASK_NAMES)) == (45, 36, 80)


def _load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_sonic_v1", SCRIPTS / "merge_sonic_v1.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["merge_sonic_v1"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- stage 1
def stage_merge(out_root: Path, workers: int, limit: int | None, dry: bool) -> None:
    m = _load_merge_module()
    m.SOURCES_TRAIN = [s for s in m.SOURCES_TRAIN if s[0] not in DROP_SOURCES]
    print("[1/5] merge  train sources: %s" % [s[0] for s in m.SOURCES_TRAIN])
    print("             val   sources: %s" % [s[0] for s in m.SOURCES_VAL])
    if dry:
        return
    argv = sys.argv
    sys.argv = ["merge_sonic_v1", "--out-root", str(out_root), "--workers", str(workers)]
    if limit:
        sys.argv += ["--limit-episodes", str(limit)]
    try:
        m.main()
    finally:
        sys.argv = argv


# ---------------------------------------------------------------- stage 2
def _valid_mask_for(emb: str, key: str, dim: int) -> np.ndarray:
    """Per-dim validity, mirroring merge_sonic_v1.stat_valid()."""
    m = _load_merge_module()
    return np.asarray(m.stat_valid(emb)[key], bool) if key in m.stat_valid(emb) else np.ones(dim, bool)


def stage_quantiles(pack: Path, subsample: int, dry: bool) -> dict:
    """Recompute q01/q99 using only rows where each dim is actually present."""
    print("[2/5] mask-aware quantiles for %s" % pack.name)
    eps = [json.loads(l) for l in open(pack / "meta/episodes.jsonl")]
    emb_of = {e["episode_index"]: e.get("source_embodiment", "g1") for e in eps}
    stats = json.loads((pack / "meta/stats_psi0.json").read_text())

    COLS = {"observation.state": 45, "action": 36, "action.body_token": 64,
            "action.neck": 2, "timestamp": 1}
    buckets = {k: [[] for _ in range(d)] for k, d in COLS.items()}

    for e in eps:
        i = e["episode_index"]
        f = pack / ("data/chunk-%03d/episode_%06d.parquet" % (i // 1000, i))
        df = pd.read_parquet(f, columns=[c for c in COLS if c in
                                         pd.read_parquet(f).columns])
        for key, dim in COLS.items():
            if key not in df.columns:
                continue
            a = df[key].to_numpy()
            a = np.stack([np.asarray(x, np.float32) for x in a]) if a.dtype == object \
                else np.asarray(a, np.float32).reshape(-1, 1)
            a = a[::subsample]
            v = _valid_mask_for(emb_of[i], key, dim)
            for d in range(dim):
                if v[d]:
                    buckets[key][d].append(a[:, d])

    out = {}
    for key, dim in COLS.items():
        if key not in stats:
            continue
        q01 = np.array(stats[key]["q01"], float)
        q99 = np.array(stats[key]["q99"], float)
        changed = 0
        for d in range(dim):
            if not buckets[key][d]:
                continue
            col = np.concatenate(buckets[key][d])
            n01, n99 = float(np.percentile(col, 1)), float(np.percentile(col, 99))
            if abs(n01 - q01[d]) > 1e-6 or abs(n99 - q99[d]) > 1e-6:
                changed += 1
            q01[d], q99[d] = n01, n99
        stats[key]["q01"] = np.round(q01, 8).tolist()
        stats[key]["q99"] = np.round(q99, 8).tolist()
        out[key] = changed
        print("       %-20s %d/%d dims updated" % (key, changed, dim))
    if not dry:
        for fn in ("stats_psi0.json", "stats.json"):
            p = pack / "meta" / fn
            if p.exists():
                bak = p.with_suffix(".json.pre_maskq.bak")
                if not bak.exists():
                    shutil.copy2(p, bak)
                p.write_text(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------- stage 3
def stage_resolution(pack: Path, workers: int, dry: bool) -> None:
    print("[3/5] normalise video to %dx%d" % (VIDEO_H, VIDEO_W))
    if dry:
        return
    subprocess.run([sys.executable, str(SCRIPTS / "normalize_pack_resolution.py"), str(pack),
                    "--height", str(VIDEO_H), "--width", str(VIDEO_W),
                    "--workers", str(workers)], check=True)


# ---------------------------------------------------------------- stage 4
def stage_prune(pack: Path, dry: bool) -> None:
    print("[4/5] prune assets the psi0 path never reads")
    info_p = pack / "meta/info.json"
    info = json.loads(info_p.read_text())
    mod_p = pack / "meta/modality.json"
    mod = json.loads(mod_p.read_text())

    imgs = pack / DROP_IMAGE_DIR
    vids = list(pack.glob("videos/chunk-*/" + DROP_VIDEO))
    print("       subgoal images: %s" % ("present" if imgs.exists() else "absent"))
    print("       %s dirs: %d" % (DROP_VIDEO, len(vids)))
    if dry:
        return

    if imgs.exists():
        shutil.rmtree(imgs)
    for d in vids:
        shutil.rmtree(d)
    # Undeclare, or lerobot still queries the key for every sample and __getitem__ dies.
    info["features"].pop(DROP_VIDEO, None)
    nvid = sum(1 for v in info["features"].values() if v.get("dtype") == "video")
    info["total_videos"] = info["total_episodes"] * nvid
    mod.get("video", {}).pop("egocentric_wm", None)
    info_p.write_text(json.dumps(info, indent=2))
    mod_p.write_text(json.dumps(mod, indent=2))
    print("       video keys now: %s" % [k for k, v in info["features"].items() if v.get("dtype") == "video"])


# ---------------------------------------------------------------- stage 5
def stage_names(pack: Path, dry: bool) -> None:
    print("[5/5] write joint names")
    p = pack / "meta/info.json"
    info = json.loads(p.read_text())
    f = info["features"]
    plan = [("observation.state", STATE_NAMES), ("action", ACTION_NAMES),
            ("action.neck", NECK), ("action.mask", MASK_NAMES)]
    for key, names in plan:
        if key not in f:
            continue
        assert f[key]["shape"] == [len(names)], (key, f[key]["shape"], len(names))
        print("       %-20s %d names" % (key, len(names)))
        if not dry:
            f[key]["names"] = names
    if not dry:
        p.write_text(json.dumps(info, indent=2))


# ---------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(ROOT / ".data"))
    ap.add_argument("--train-name", default="psix_sonic_v1_train")
    ap.add_argument("--val-name", default="psix_sonic_v1_val")
    ap.add_argument("--stages", default="1,2,3,4,5",
                    help="comma-separated subset, e.g. --stages 2,5")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--subsample", type=int, default=20, help="1-in-N rows for quantiles")
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    want = {int(x) for x in a.stages.split(",") if x.strip()}
    out_root = Path(a.out_root)
    tr, va = out_root / a.train_name, out_root / a.val_name

    if 1 in want:
        stage_merge(out_root, a.workers, a.limit_episodes, a.dry_run)
    if 2 in want:
        # Stats are computed on TRAIN and val SHARES them verbatim -- one global
        # normalisation for both splits (train scripts assert the files are identical).
        stats = stage_quantiles(tr, a.subsample, a.dry_run)
        if not a.dry_run:
            for fn in ("stats_psi0.json", "stats.json"):
                src, dst = tr / "meta" / fn, va / "meta" / fn
                if src.exists():
                    shutil.copy2(src, dst)
            print("       val now shares train stats verbatim")
    for pack in (tr, va):
        if 3 in want:
            stage_resolution(pack, a.workers, a.dry_run)
        if 4 in want:
            stage_prune(pack, a.dry_run)
        if 5 in want:
            stage_names(pack, a.dry_run)
    print("\ndone%s" % (" (dry run)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
