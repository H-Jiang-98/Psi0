#!/usr/bin/env python
"""Merge the .data.sonic-v1 LeRobot datasets into ONE psi0 post-training train/val pair.

Goal
----
Produce `.data/psix_sonic_v1_{train,val}` whose action/state layout matches the
reference finetune dataset (.data/g1_sonic_lerobot_0810_merged_train), so the sonic
psi0 path consumes it with the SonicRepackTransform defaults:

    action_keys = ["action.body_token", "action[:14]", "action.neck"]   -> 64+14+2 = 80
    state_keys  = ["observation.state"]                                 -> 45

Canonical layout written here (G1 convention)
---------------------------------------------
    action (36)           = [hands(14) | arms(14) | torso(4) | base(4)]
    observation.state(45) = [hands(14) | arms(14) | legs(12) | waist(3) | neck(2)]
    action.body_token(64) = passthrough (embodiment-agnostic; the 1/16-grid v1 token,
                            NOT body_token_v1_1 which is continuous on zmsonic)
    action.neck(2)        = real for zmsonic, zeros elsewhere
    action.mask(80)       = validity of the REPACKED vector [token(64)|hands(14)|neck(2)]

Embodiment normalisation
------------------------
H1 carries 12 finger DoF and a 39-D state vs G1's 14 / 43. We pad the finger block
12 -> 14 with zeros, which shifts arms/legs/torso into exactly the G1 slots:

    H1 action 34: hands[0:12] arms[12:26] torso+base[26:34]
              ->  hands[0:14] arms[14:28] torso+base[28:36]
    H1 state  39: hands[0:12] arms[12:26] legs[26:38] torso_joint[38]
              ->  hands[0:14] arms[14:28] legs[28:40] waist[40:43] neck[43:45]

The two padded finger slots are marked invalid in action.mask, as is the neck block
for every source except zmsonic. `actions_mask` is honoured by the training loss
(finetune.py: loss_action = (loss_action * mask).sum(1)), so padded dims contribute
nothing to the gradient. Stats are likewise computed only over valid dims, otherwise
the zero fill would drag the neck/finger normalisation bounds toward zero.

Video resolution
----------------
mp4s are copied byte-for-byte, so this script does NOT change resolution -- a merged
pack inherits whatever its sources had, and meta/info.json can only declare ONE shape
per video key. The sonic-v1 sources are NOT uniform:

    uni / he_g1 / he_h1 / psi0   480x640 (aspect 1.333)   16,825 eps
    zmsonic train+val            384x672 (aspect 1.750)      712 eps

That matters because the training transform is `v2.Resize((270, 480))` with an
explicit 2-tuple, which does NOT preserve aspect: 480x640 gets stretched ~33%
horizontally while 384x672 is left essentially untouched. Since zmsonic is 100% of
VAL and only 4% of TRAIN, val geometry would not match the training distribution.

So this script now REFUSES to leave that silent: it compares every source's declared
video shape and, if they differ, prints the exact normalisation command to run:

    python scripts/data/normalize_pack_resolution.py <pack> --height H --width W

Usage
-----
    python scripts/data/merge_sonic_v1.py --out-root .data --limit-episodes 2   # dry run
    python scripts/data/merge_sonic_v1.py --out-root .data --workers 16         # full
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/mnt/beegfs/scratch/songlinwei/psi0")
SRC_ROOT = ROOT / ".data.sonic-v1"

A_DIM, S_DIM, TOK_DIM, HAND_DIM, NECK_DIM = 36, 45, 64, 14, 2
MASK_DIM = TOK_DIM + HAND_DIM + NECK_DIM          # 80, repacked order
CHUNK = 1000
VIDEO_KEYS = ["observation.images.egocentric", "observation.images.egocentric_wm"]
IMAGE_KEY = "observation.images.subgoal"

# embodiment -> (n finger DoF, source state width, has real neck)
SPECS = {
    "g1":     dict(hands=14, state=43, neck=False),
    "g1neck": dict(hands=14, state=45, neck=True),
    "h1":     dict(hands=12, state=39, neck=False),
}

HE = SRC_ROOT / "pretrain/he_lerobot_psix_train_val_wm"
SOURCES_TRAIN = [
    ("uni",      SRC_ROOT / "pretrain/Unifolm_lerobot_psix_wm/g1",                              "g1"),
    ("he_g1_tr", HE / "train/g1",                                                               "g1"),
    ("he_g1_va", HE / "val/g1",                                                                 "g1"),
    # ("he_h1_tr", HE / "train/h1",                                                               "h1"),
    # ("he_h1_va", HE / "val/h1",                                                                 "h1"),
    ("psi0_tr",  SRC_ROOT / "posttrain/psi0_lerobot_psix_train_val_wm/train/g1",                "g1"),
    ("psi0_va",  SRC_ROOT / "posttrain/psi0_lerobot_psix_train_val_wm/val/g1",                  "g1"),
    ("zm_tr",    SRC_ROOT / "posttrain/zmsonic_lerobot_psix_train_val_wm/zmsonic_lerobot_psix_train/g1", "g1neck"),
]
SOURCES_VAL = [
    ("zm_va",    SRC_ROOT / "posttrain/zmsonic_lerobot_psix_train_val_wm/zmsonic_lerobot_psix_val/g1",   "g1neck"),
]

# The 22 columns every source shares (eepose action.hands.*/action.wrists.* are
# embodiment-specific and unused by the psi0 path, so they are dropped).
KEEP_COLS = [
    "observation.state", "action", "action.body_token", "action.body_token_v1_1",
    "timestamp", "frame_index", "episode_index", "index", "task_index",
    "task_description", "next.done", "memory_desc", "memory_items",
    "subtask_prompt", "next_subtask", "prev_subtask",
    "sub_task_index", "sub_goal_frame_index", "sub_task_delta", "sub_goal_image_path",
]
# Columns whose per-dim stats we track globally.
STAT_COLS = OrderedDict([
    ("observation.state", S_DIM), ("action", A_DIM),
    ("action.body_token", TOK_DIM), ("action.neck", NECK_DIM), ("timestamp", 1),
])


# ---------------------------------------------------------------- layout helpers
def canon_action(a: np.ndarray, emb: str) -> np.ndarray:
    """(N, 34|36) -> (N, 36) in G1 slot order."""
    n = a.shape[0]
    if SPECS[emb]["hands"] == HAND_DIM:
        assert a.shape[1] == A_DIM, f"{emb}: action {a.shape}"
        return a.astype(np.float32)
    out = np.zeros((n, A_DIM), np.float32)          # h1: 34 -> 36
    out[:, 0:12] = a[:, 0:12]                       # fingers (12), slots 12:14 stay 0
    out[:, 14:36] = a[:, 12:34]                     # arms(14) + torso/base(8)
    return out


def canon_state(s: np.ndarray, emb: str) -> np.ndarray:
    """(N, 39|43|45) -> (N, 45) in G1 slot order."""
    n = s.shape[0]
    out = np.zeros((n, S_DIM), np.float32)
    if emb == "g1neck":
        assert s.shape[1] == 45
        out[:] = s
    elif emb == "g1":
        assert s.shape[1] == 43
        out[:, 0:43] = s                            # neck slots stay 0
    else:                                           # h1: 39 -> 45
        assert s.shape[1] == 39
        out[:, 0:12] = s[:, 0:12]                   # fingers, slots 12:14 stay 0
        out[:, 14:28] = s[:, 12:26]                 # arms
        out[:, 28:40] = s[:, 26:38]                 # legs
        out[:, 40:41] = s[:, 38:39]                 # torso_joint -> waist_yaw
    return out


def action_mask_row(emb: str) -> np.ndarray:
    """Validity of the repacked vector [token(64) | hands(14) | neck(2)]."""
    m = np.ones(MASK_DIM, np.float32)
    if SPECS[emb]["hands"] != HAND_DIM:              # h1 padded finger slots
        m[TOK_DIM + 12: TOK_DIM + 14] = 0.0
    if not SPECS[emb]["neck"]:
        m[TOK_DIM + HAND_DIM:] = 0.0
    return m


def stat_valid(emb: str) -> dict[str, np.ndarray]:
    """Per-dim validity used when accumulating global stats (raw column order)."""
    st = np.ones(S_DIM, bool)
    ac = np.ones(A_DIM, bool)
    if not SPECS[emb]["neck"]:
        st[43:45] = False
    if SPECS[emb]["hands"] != HAND_DIM:              # h1
        st[12:14] = False                            # padded fingers
        st[41:43] = False                            # waist roll/pitch absent
        ac[12:14] = False
    return {
        "observation.state": st,
        "action": ac,
        "action.body_token": np.ones(TOK_DIM, bool),
        "action.neck": np.full(NECK_DIM, SPECS[emb]["neck"]),
        "timestamp": np.ones(1, bool),
    }


# ---------------------------------------------------------------- per-episode work
def process_episode(job: dict) -> dict:
    src, emb = Path(job["src"]), job["emb"]
    old_ep, new_ep, idx0 = job["old_ep"], job["new_ep"], job["index_offset"]
    out = Path(job["out"])
    sub = int(np.random.default_rng(new_ep).integers(0, job["subsample"])) if job["subsample"] > 1 else 0

    src_pq = src / f"data/chunk-{old_ep // CHUNK:03d}/episode_{old_ep:06d}.parquet"
    tbl = pq.read_table(src_pq, columns=[c for c in KEEP_COLS if c in pq.ParquetFile(src_pq).schema_arrow.names])
    n = tbl.num_rows

    cols: dict[str, pa.Array] = {}
    for c in tbl.column_names:
        cols[c] = tbl[c]

    act = canon_action(np.asarray(tbl["action"].to_pylist(), np.float32), emb)
    sta = canon_state(np.asarray(tbl["observation.state"].to_pylist(), np.float32), emb)
    tok = np.asarray(tbl["action.body_token"].to_pylist(), np.float32)
    if "action.neck" in pq.ParquetFile(src_pq).schema_arrow.names:
        neck = np.asarray(pq.read_table(src_pq, columns=["action.neck"])["action.neck"].to_pylist(), np.float32)
    else:
        neck = np.zeros((n, NECK_DIM), np.float32)
    mask = np.broadcast_to(action_mask_row(emb), (n, MASK_DIM)).copy()

    # --- global stats accumulation (valid dims only) -------------------------
    valid = stat_valid(emb)
    acc, sample = {}, {}
    for name, arr in (("observation.state", sta), ("action", act),
                      ("action.body_token", tok), ("action.neck", neck),
                      ("timestamp", np.asarray(tbl["timestamp"].to_pylist(), np.float32).reshape(n, 1))):
        v = valid[name]
        cnt = np.where(v, n, 0).astype(np.int64)
        a_ = np.where(v[None, :], arr, 0.0)
        acc[name] = dict(
            count=cnt, sum=a_.sum(0, dtype=np.float64), sumsq=(a_.astype(np.float64) ** 2).sum(0),
            min=np.where(v, arr.min(0, initial=np.inf), np.inf),
            max=np.where(v, arr.max(0, initial=-np.inf), -np.inf),
        )
        sample[name] = arr[sub::job["subsample"]] if job["subsample"] > 1 else arr

    # Per-episode stats in the shape LeRobot's aggregate_stats() demands:
    # per-dim min/max/mean/std plus a 1-element count. These describe the bytes
    # actually written (zero fill included) and are NOT the mask-aware numbers
    # used for stats_psi0.json.
    epstat = {}
    for name, arr in (("observation.state", sta), ("action", act),
                      ("action.body_token", tok), ("action.neck", neck),
                      ("timestamp", np.asarray(tbl["timestamp"].to_pylist(), np.float32).reshape(n, 1))):
        epstat[name] = dict(min=arr.min(0).tolist(), max=arr.max(0).tolist(),
                            mean=arr.mean(0).tolist(), std=arr.std(0).tolist(), count=[int(n)])

    # --- rewrite index columns ----------------------------------------------
    newcols, names = [], []
    for c in tbl.column_names:
        if c == "action":
            arr = pa.array(list(act), type=pa.list_(pa.float32(), A_DIM))
        elif c == "observation.state":
            arr = pa.array(list(sta), type=pa.list_(pa.float32(), S_DIM))
        elif c == "episode_index":
            arr = pa.array(np.full(n, new_ep, np.int64))
        elif c == "index":
            arr = pa.array(np.arange(idx0, idx0 + n, dtype=np.int64))
        elif c == "task_index":
            arr = pa.array(np.full(n, job["task_index"], np.int64))
        elif c == "sub_goal_image_path":
            old = f"episode_{old_ep:06d}"; new = f"episode_{new_ep:06d}"
            arr = pa.array([None if v is None else v.replace(old, new) for v in tbl[c].to_pylist()])
        else:
            arr = tbl[c]
        newcols.append(arr); names.append(c)
    newcols.append(pa.array(list(neck), type=pa.list_(pa.float32(), NECK_DIM))); names.append("action.neck")
    newcols.append(pa.array(list(mask), type=pa.list_(pa.float32(), MASK_DIM))); names.append("action.mask")

    dst_pq = out / f"data/chunk-{new_ep // CHUNK:03d}/episode_{new_ep:06d}.parquet"
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(newcols, names=names), dst_pq, compression="snappy")

    # --- copy media ----------------------------------------------------------
    for vk in VIDEO_KEYS:
        s = src / f"videos/chunk-{old_ep // CHUNK:03d}/{vk}/episode_{old_ep:06d}.mp4"
        if s.exists():
            d = out / f"videos/chunk-{new_ep // CHUNK:03d}/{vk}/episode_{new_ep:06d}.mp4"
            d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists():
                shutil.copy2(s, d)
    s_img = src / f"images/{IMAGE_KEY}/episode_{old_ep:06d}"
    if s_img.is_dir():
        d_img = out / f"images/{IMAGE_KEY}/episode_{new_ep:06d}"
        if not d_img.exists():
            d_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(s_img, d_img)

    return dict(new_ep=new_ep, length=n, acc=acc, epstat=epstat,
                sample={k: v.tolist() for k, v in sample.items()},
                index_offset=idx0, src_name=job["src_name"], emb=emb)


# ---------------------------------------------------------------- meta helpers
def load_meta(src: Path) -> dict:
    m = {}
    for f in ("info", "tasks", "episodes"):
        p = src / f"meta/{f}.json" if f == "info" else src / f"meta/{f}.jsonl"
        if f == "info":
            m["info"] = json.load(open(p))
        else:
            m[f] = [json.loads(l) for l in open(p) if l.strip()]
    return m


def build_jobs(sources, out: Path, limit: int | None, subsample: int):
    """Assign global episode ids / frame offsets / task ids across all sources."""
    jobs, ep_meta, tasks, task_of = [], [], [], {}
    new_ep, idx = 0, 0
    for name, src, emb in sources:
        meta = load_meta(src)
        by_idx = {t["task_index"]: t for t in meta["tasks"]}
        eps = meta["episodes"]
        if limit:
            eps = eps[:limit]
        for e in eps:
            old = e["episode_index"]
            # resolve this episode's task text -> merged task table
            tid = (e.get("tasks") or [0])[0]
            trow = by_idx.get(tid, {"task": f"{name}_{tid}", "description": ""})
            key = (trow.get("task", ""), trow.get("description", ""))
            if key not in task_of:
                task_of[key] = len(tasks)
                tasks.append({"task_index": len(tasks), "task": key[0],
                              "category": trow.get("category", ""), "description": key[1]})
            jobs.append(dict(src=str(src), src_name=name, emb=emb, old_ep=old, new_ep=new_ep,
                             index_offset=idx, task_index=task_of[key], out=str(out),
                             subsample=subsample))
            row = dict(e)
            row.update(episode_index=new_ep, tasks=[task_of[key]],
                       dataset_from_index=idx, dataset_to_index=idx + e["length"] - 1,
                       source_dataset=name, source_embodiment=emb,
                       source_episode_index=old)
            ep_meta.append(row)
            new_ep += 1
            idx += e["length"]
    return jobs, ep_meta, tasks, idx


def merge_stats(accs: list[dict], samples: list[dict]) -> dict:
    out = {}
    for name, dim in STAT_COLS.items():
        cnt = np.zeros(dim, np.int64); s = np.zeros(dim); sq = np.zeros(dim)
        mn = np.full(dim, np.inf); mx = np.full(dim, -np.inf)
        for a in accs:
            r = a[name]
            cnt += np.asarray(r["count"]); s += np.asarray(r["sum"]); sq += np.asarray(r["sumsq"])
            mn = np.minimum(mn, np.asarray(r["min"])); mx = np.maximum(mx, np.asarray(r["max"]))
        safe = np.maximum(cnt, 1)
        mean = s / safe
        var = np.maximum(sq / safe - mean ** 2, 0.0)
        # quantiles from the subsample, valid rows only (invalid dims are exactly 0
        # in every contributing row, so drop dims with no valid source)
        pool = [np.asarray(x[name], np.float32) for x in samples if len(x[name])]
        q01 = np.zeros(dim); q99 = np.zeros(dim)
        if pool:
            P = np.concatenate(pool, 0)
            q01 = np.percentile(P, 1, axis=0); q99 = np.percentile(P, 99, axis=0)
        dead = cnt == 0
        mn = np.where(dead, 0.0, mn); mx = np.where(dead, 0.0, mx)
        out[name] = {k: np.asarray(v, np.float64).round(8).tolist() for k, v in
                     dict(mean=mean, std=np.sqrt(var), min=mn, max=mx, q01=q01, q99=q99).items()}
        out[name]["count"] = cnt.tolist()
    return out


def write_meta(out: Path, template: Path, ep_meta, tasks, total_frames, ep_stats, stats):
    info = json.load(open(template / "meta/info.json"))
    feats = info["features"]
    for k in list(feats):
        if k not in KEEP_COLS + VIDEO_KEYS:
            feats.pop(k)
    feats["action"] = {**feats["action"], "shape": [A_DIM], "names": None}
    feats["observation.state"] = {**feats["observation.state"], "shape": [S_DIM], "names": None}
    feats["action.neck"] = {"dtype": "float32", "shape": [NECK_DIM], "names": None}
    feats["action.mask"] = {"dtype": "float32", "shape": [MASK_DIM], "names": None}
    info.update(total_episodes=len(ep_meta), total_frames=total_frames,
                total_tasks=len(tasks), total_videos=len(ep_meta) * len(VIDEO_KEYS),
                total_chunks=(len(ep_meta) + CHUNK - 1) // CHUNK, chunks_size=CHUNK,
                splits={"train": f"0:{len(ep_meta)}"})
    (out / "meta").mkdir(parents=True, exist_ok=True)
    json.dump(info, open(out / "meta/info.json", "w"), indent=2)
    for fn, rows in (("episodes.jsonl", ep_meta), ("tasks.jsonl", tasks),
                     ("episodes_stats.jsonl", ep_stats)):
        with open(out / f"meta/{fn}", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    shutil.copy2(template / "meta/modality.json", out / "meta/modality.json")
    json.dump(stats, open(out / "meta/stats.json", "w"), indent=2)
    json.dump(stats, open(out / "meta/stats_psi0.json", "w"), indent=2)


def run(sources, out: Path, limit, workers, subsample) -> tuple[list, list, int]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    jobs, ep_meta, tasks, total = build_jobs(sources, out, limit, subsample)
    print(f"[{out.name}] {len(jobs)} episodes, {total} frames, {len(tasks)} tasks", flush=True)
    accs, samples, ep_stats, done = [], [], [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_episode, j): j for j in jobs}
        for fu in as_completed(futs):
            r = fu.result()
            accs.append(r["acc"]); samples.append(r["sample"])
            ep_stats.append({"episode_index": r["new_ep"], "stats": r["epstat"]})
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"  [{out.name}] {done}/{len(jobs)}", flush=True)
    ep_stats.sort(key=lambda r: r["episode_index"])
    return accs, samples, ep_stats, ep_meta, tasks, total


def audit_resolutions(sources, out: Path) -> None:
    """Report the video shapes across sources; a merged pack can declare only one."""
    shapes: dict[tuple, list[str]] = {}
    for name, src, _ in sources:
        feats = json.load(open(Path(src) / "meta/info.json"))["features"]
        for key in VIDEO_KEYS:
            sh = tuple(feats.get(key, {}).get("shape", []) or [])
            if sh:
                shapes.setdefault(sh, []).append(f"{name}/{key.split('.')[-1]}")
    if len(shapes) <= 1:
        return
    print("\n*** MIXED VIDEO RESOLUTIONS ***", flush=True)
    for sh, who in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
        print(f"    {sh}: {len(who)} source/key pairs  e.g. {who[:3]}", flush=True)
    majority = max(shapes, key=lambda k: len(shapes[k]))
    h, w = majority[0], majority[1]
    print(
        "    info.json can declare only ONE shape per key, and the training transform\n"
        "    v2.Resize((270,480)) does NOT preserve aspect ratio, so mixed inputs get\n"
        "    different geometric distortion. Normalise the pack before training:\n"
        f"      python scripts/data/normalize_pack_resolution.py {out} --height {h} --width {w}\n",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / ".data"))
    ap.add_argument("--train-name", default="psix_sonic_v1_train")
    ap.add_argument("--val-name", default="psix_sonic_v1_val")
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--subsample", type=int, default=20, help="1-in-N frames kept for q01/q99")
    a = ap.parse_args()
    out_root = Path(a.out_root)

    tr_out, va_out = out_root / a.train_name, out_root / a.val_name
    audit_resolutions(SOURCES_TRAIN, tr_out)
    audit_resolutions(SOURCES_VAL, va_out)
    accs, samples, ep_stats, ep_meta, tasks, total = run(SOURCES_TRAIN, tr_out, a.limit_episodes,
                                                         a.workers, a.subsample)
    stats = merge_stats(accs, samples)           # global stats: computed on TRAIN
    write_meta(tr_out, Path(SOURCES_TRAIN[-1][1]), ep_meta, tasks, total, ep_stats, stats)

    vaccs, vsamples, vep_stats, vep_meta, vtasks, vtotal = run(SOURCES_VAL, va_out, a.limit_episodes,
                                                               a.workers, a.subsample)
    # val SHARES the train stats file verbatim (single global normalisation)
    write_meta(va_out, Path(SOURCES_VAL[-1][1]), vep_meta, vtasks, vtotal, vep_stats, stats)

    print(f"\nTRAIN {tr_out}: {len(ep_meta)} eps / {total} frames")
    print(f"VAL   {va_out}: {len(vep_meta)} eps / {vtotal} frames")
    print("global stats keys:", list(stats.keys()))
    for k in ("action.neck", "action", "observation.state"):
        c = stats[k]["count"]
        print(f"  {k:20s} valid-frame count per dim (head): {c[:6]} ... tail {c[-4:]}")


if __name__ == "__main__":
    main()
