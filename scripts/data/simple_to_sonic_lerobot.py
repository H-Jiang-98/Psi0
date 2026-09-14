"""Convert a SIMPLE G1-wholebody LeRobot dataset into the schema consumed by
`scripts/train/psi0/finetune-sonic-neckless-psi0.sh` (SonicRepackTransform).

Unlike raw_sonic_to_psix.py, the input here is ALREADY LeRobot v2.1 -- this is a
schema remap, not a raw ingest, so the original parquet/video layout is preserved
and only the fields the sonic loader needs are rewritten.

What the sonic loader requires (src/psi/config/transform_psi0_sonic.py):
  image_keys      = ["observation.images.egocentric"]
  state_keys      = ["observation.state"]                       (pad_state_dim=43)
  action_keys     = ["action.body_token", "action[:14]"]        (pad_action_dim=78)
  instruction_key = "task_description"
  field.stat_path = <repo>/meta/stats_psi0.json, keyed by base feature name

Source layout (from its meta/modality.json):
  action           (78) = [token(64) | left_hand(7) | right_hand(7)]
  observation.state(43) = [left_leg(6) right_leg(6) waist(3)
                           left_arm(7) left_hand(7) right_arm(7) right_hand(7)]

Mapping applied:
  action.body_token (64) = action[0:64]
  action            (36) = [action[64:78] (hands) | zeros(22)]
                           Matches the reference g1_v30 action width, whose
                           modality.json defines hand_joints[0:14]; the remaining
                           torso/base dims are unused on this training path and are
                           zeroed, exactly as raw_sonic_to_psix.py does.
  observation.state (43) = passthrough (see --reorder-state)
  task_description       = per-episode task string from meta/tasks.jsonl
  video key renamed      = observation.images.ego_view -> .egocentric (symlinked)

STATE ORDERING CAVEAT: the reference dataset's modality.json documents state only
as an opaque `joint_positions[0,43)`, so its internal joint order cannot be
verified from metadata. Both datasets are 43D G1 wholebody, so passthrough is the
default. `--reorder-state hands_arms_legs` emits [hands(14)|arms(14)|legs(15)] --
the ordering raw_sonic_to_psix.py documents for the psix path -- if that turns out
to be what the pretrained sonic checkpoint expects.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_VIDEO_KEY = "observation.images.ego_view"
DST_VIDEO_KEY = "observation.images.egocentric"
TOKEN_DIM = 64
HAND_DIM = 14
ACTION_DIM = 36  # hands(14) + zeroed torso/base(22), matching reference g1_v30

# source observation.state slices, from its meta/modality.json
S_LEFT_LEG = (0, 6)
S_RIGHT_LEG = (6, 12)
S_WAIST = (12, 15)
S_LEFT_ARM = (15, 22)
S_LEFT_HAND = (22, 29)
S_RIGHT_ARM = (29, 36)
S_RIGHT_HAND = (36, 43)


def reorder_state(s: np.ndarray) -> np.ndarray:
    """[legs|arms interleaved with hands] -> [hands(14)|arms(14)|legs(15)]."""
    sl = lambda a, b: s[:, a:b]
    return np.concatenate(
        [
            sl(*S_LEFT_HAND), sl(*S_RIGHT_HAND),          # hands 14
            sl(*S_LEFT_ARM), sl(*S_RIGHT_ARM),            # arms 14
            sl(*S_LEFT_LEG), sl(*S_RIGHT_LEG), sl(*S_WAIST),  # legs 15
        ],
        axis=1,
    )


def stats_of(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).tolist(),
        "count": [int(a.shape[0])],
    }


def merge_stats(per_ep: list[dict], key: str) -> dict:
    """Exact min/max/count; mean/std/quantiles pooled by frame count."""
    mins = np.array([e[key]["min"] for e in per_ep])
    maxs = np.array([e[key]["max"] for e in per_ep])
    ns = np.array([e[key]["count"][0] for e in per_ep], dtype=np.float64)
    means = np.array([e[key]["mean"] for e in per_ep])
    stds = np.array([e[key]["std"] for e in per_ep])
    w = ns / ns.sum()
    mean = (means * w[:, None]).sum(0)
    # pooled variance = E[var] + var of means
    var = ((stds**2 + (means - mean) ** 2) * w[:, None]).sum(0)
    return {
        "min": mins.min(0).tolist(),
        "max": maxs.max(0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "q01": np.array([e[key]["q01"] for e in per_ep]).min(0).tolist(),
        "q99": np.array([e[key]["q99"] for e in per_ep]).max(0).tolist(),
        "count": [int(ns.sum())],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source level dir (contains data/ meta/ videos/)")
    ap.add_argument("--out", required=True, help="output repo dir")
    ap.add_argument("--reorder-state", choices=["none", "hands_arms_legs"], default="none")
    ap.add_argument("--link-videos", action="store_true", default=True,
                    help="symlink videos instead of copying (default)")
    ap.add_argument("--copy-videos", dest="link_videos", action="store_false")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    assert (src / "meta" / "info.json").exists(), f"no meta/info.json under {src}"
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()}

    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    ep_files = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
    episodes, ep_stats, total_frames = [], [], 0

    for pq in tqdm(ep_files, desc="episodes"):
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].values).astype(np.float32)          # (T,78)
        state = np.stack(df["observation.state"].values).astype(np.float32)  # (T,43)
        assert act.shape[1] == 78 and state.shape[1] == 43, f"unexpected dims in {pq.name}"

        body_token = act[:, :TOKEN_DIM]                                  # (T,64)
        hands = act[:, TOKEN_DIM:]                                       # (T,14)
        new_action = np.zeros((len(df), ACTION_DIM), dtype=np.float32)
        new_action[:, :HAND_DIM] = hands
        if args.reorder_state == "hands_arms_legs":
            state = reorder_state(state).astype(np.float32)

        ti = int(df["task_index"].iloc[0])
        task_str = tasks.get(ti, "")

        df["action"] = list(new_action)
        df["action.body_token"] = list(body_token)
        df["observation.state"] = list(state)
        df["task_description"] = task_str
        df["next.done"] = [False] * (len(df) - 1) + [True]

        ep_idx = int(df["episode_index"].iloc[0])
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet", index=False)

        episodes.append({"episode_index": ep_idx, "tasks": [task_str], "length": len(df)})
        ep_stats.append({
            "episode_index": ep_idx,
            "stats": {
                "action": stats_of(new_action),
                "action.body_token": stats_of(body_token),
                "observation.state": stats_of(state),
            },
        })
        total_frames += len(df)

    # ---- videos: expose the source clips under the key the loader expects ----
    vsrc = src / "videos" / "chunk-000" / SRC_VIDEO_KEY
    vdst = out / "videos" / "chunk-000" / DST_VIDEO_KEY
    vdst.parent.mkdir(parents=True, exist_ok=True)
    if vdst.exists() or vdst.is_symlink():
        if vdst.is_symlink():
            vdst.unlink()
        else:
            shutil.rmtree(vdst)
    if args.link_videos:
        vdst.symlink_to(vsrc.resolve(), target_is_directory=True)
    else:
        shutil.copytree(vsrc, vdst)

    # ---- meta ----
    feats = dict(info["features"])
    feats[DST_VIDEO_KEY] = feats.pop(SRC_VIDEO_KEY)
    feats["action"] = {"dtype": "float32", "shape": [ACTION_DIM],
                       "names": [f"a{i}" for i in range(ACTION_DIM)]}
    feats["action.body_token"] = {"dtype": "float32", "shape": [TOKEN_DIM],
                                  "names": [f"t{i}" for i in range(TOKEN_DIM)]}
    feats["observation.state"] = {"dtype": "float32", "shape": [43],
                                  "names": [f"s{i}" for i in range(43)]}
    feats["task_description"] = {"dtype": "string", "shape": [1], "names": None}
    feats["next.done"] = {"dtype": "bool", "shape": [1], "names": None}
    info_out = dict(info)
    info_out.update({"features": feats, "total_episodes": len(episodes),
                     "total_frames": total_frames, "total_videos": len(episodes)})
    (out / "meta" / "info.json").write_text(json.dumps(info_out, indent=2))
    (out / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes))
    (out / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in ep_stats))
    shutil.copy(src / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")

    (out / "meta" / "modality.json").write_text(json.dumps({
        "state": {"joint_positions": {"start": 0, "end": 43}},
        "action": {
            "hand_joints": {"start": 0, "end": 14},
            "body_token": {"start": 0, "end": 64, "original_key": "action.body_token"},
        },
        "video": {"egocentric": {"original_key": DST_VIDEO_KEY}},
        "annotation": {"task_description": {}},
    }, indent=2))

    merged = {k: merge_stats([e["stats"] for e in ep_stats], k)
              for k in ("action", "action.body_token", "observation.state")}
    (out / "meta" / "stats_psi0.json").write_text(json.dumps(merged, indent=2))
    shutil.copy(out / "meta" / "stats_psi0.json", out / "meta" / "stats.json")

    print(f"\nwrote {len(episodes)} episodes / {total_frames} frames -> {out}")
    print(f"  action {ACTION_DIM}D (hands in [0:14]) + action.body_token {TOKEN_DIM}D "
          f"= {ACTION_DIM and TOKEN_DIM + HAND_DIM}D consumed by the loader")
    print(f"  state 43D (reorder={args.reorder_state})   video key -> {DST_VIDEO_KEY}")


if __name__ == "__main__":
    main()
