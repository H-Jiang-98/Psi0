#!/usr/bin/env python3
"""Carve a task subset out of a LeRobot v2/v3 pack into a new standalone dataset.

Selects every episode whose ``task_index`` is in ``--task-indices`` and writes a
fresh pack with the same on-disk layout (``data/``, ``videos/``, ``images/``,
``meta/``), so the result loads with the *same* embodiment tag as the source.

What it does
------------
* Re-indexes the kept episodes to a contiguous ``0..N-1`` range: renames the
  parquet / video / subgoal-image files, rewrites the parquet ``episode_index``,
  the GLOBAL ``index`` column (re-based to the new frame offsets) and any string
  column embedding an ``episode_XXXXXX`` path.
* Remaps ``task_index`` to a contiguous ``0..T-1`` range and emits the matching
  ``meta/tasks.jsonl`` (``--keep-task-indices`` keeps the source numbering).
* Rewrites ``meta/info.json`` counts + ``splits``, and emits filtered/re-indexed
  ``meta/episodes.jsonl``, ``meta/episodes_stats.jsonl`` and
  ``meta/merge_manifest.json`` (the manifest's per-episode provenance is kept and
  gains a ``source_split_episode_index`` field).
* Recomputes every ``meta/stats*.json`` over the SUBSET (``--stats copy`` keeps
  the source's whole-dataset stats instead, e.g. to normalize identically to a
  model trained on the full pack).
* Copies every other ``meta/*`` file verbatim (modality.json, embodiment.json, ...).

Examples
--------
    # all pick & place episodes of the merged g1-sonic train pack
    python scripts/data/split_lerobot_by_task.py \
        --src /home/songlin/hfm/data/g1_sonic_lerobot_0810_merged_train \
        --dst /home/songlin/hfm/data/g1_sonic_lerobot_0810_merged_train_pnp \
        --task-indices 0 10 11 22

    # inspect the selection without writing anything
    python scripts/data/split_lerobot_by_task.py --src ... --task-indices 0 10 11 22 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_lerobot_train_val import dump_json, dump_jsonl, load_jsonl, place_dir, place_file  # noqa: E402


# ------------------------------------------------------------------------ stats
def compute_stats(parquet_files: list[Path], info: dict) -> dict:
    """mean/std/min/max/q01/q99 per float feature -- mirrors the gear loader's
    ``calculate_dataset_statistics`` (same formula as split_lerobot_train_val)."""
    float_feats = [k for k, v in info["features"].items() if "float" in v["dtype"]]
    per_feat: dict[str, list[np.ndarray]] = {f: [] for f in float_feats}
    for pf in parquet_files:
        t = pq.read_table(pf, columns=float_feats)
        for f in float_feats:
            a = np.asarray(t.column(f).to_pylist(), dtype=np.float32)
            if a.ndim == 1:  # scalar float column -> (rows, 1)
                a = a[:, None]
            per_feat[f].append(a)
    stats = {}
    for f in float_feats:
        arr = np.vstack(per_feat[f])
        stats[f] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Source pack root (folder holding data/ videos/ meta/).")
    ap.add_argument("--dst", required=True, type=Path, help="Output pack root.")
    ap.add_argument("--task-indices", required=True, type=int, nargs="+",
                    help="SOURCE task indices to keep, e.g. --task-indices 0 10 11 22")
    ap.add_argument("--keep-task-indices", action="store_true",
                    help="Keep the source task_index values instead of remapping to 0..T-1.")
    ap.add_argument("--stats", choices=["recompute", "copy"], default="recompute",
                    help="stats*.json: recompute over the subset (default) or copy the source's.")
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink",
                    help="How to place videos/images (parquet is always rewritten). Default hardlink.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing --dst.")
    ap.add_argument("--dry-run", action="store_true", help="Print the selection; write nothing.")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    info = json.loads((src / "meta" / "info.json").read_text())
    episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    tasks = {int(t["task_index"]): t["task"] for t in load_jsonl(src / "meta" / "tasks.jsonl")}

    want = sorted(dict.fromkeys(args.task_indices))
    missing = [t for t in want if t not in tasks]
    if missing:
        raise SystemExit(f"--task-indices not in meta/tasks.jsonl: {missing}")

    # episodes.jsonl carries task *text*, not task_index -> resolve via the parquet-free text map.
    text_to_index = {v: k for k, v in tasks.items()}
    keep: list[tuple[int, int, dict]] = []  # (source_episode_index, source_task_index, meta row)
    for e in episodes:
        ep = int(e["episode_index"])
        texts = e.get("tasks") or []
        ti = text_to_index.get(texts[0]) if texts else None
        if ti is None:
            raise SystemExit(f"episode {ep}: task text not found in tasks.jsonl: {texts!r}")
        if ti in want:
            keep.append((ep, ti, e))
    if not keep:
        raise SystemExit(f"No episodes match task indices {want}")

    task_map = {t: (t if args.keep_task_indices else i) for i, t in enumerate(want)}

    print(f"Source : {src}  ({len(episodes)} episodes, {len(tasks)} tasks)")
    for t in want:
        n = sum(1 for _, ti, _ in keep if ti == t)
        frames = sum(int(e["length"]) for _, ti, e in keep if ti == t)
        print(f"  task {t:>3} -> {task_map[t]:>3}  {n:>4} eps  {frames:>7} frames  | {tasks[t]}")
    print(f"Keep   : {len(keep)} episodes, {sum(int(e['length']) for _, _, e in keep)} frames")
    if args.dry_run:
        print("[dry-run] nothing written")
        return

    if dst.exists():
        if not args.force:
            raise SystemExit(f"Output exists (use --force to overwrite): {dst}")
        shutil.rmtree(dst)

    chunks_size = info["chunks_size"]
    data_tmpl, video_tmpl = info["data_path"], info["video_path"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    string_feats = [k for k, v in info["features"].items() if v["dtype"] == "string"]
    img_root = src / "images"
    img_subdirs = [d.name for d in img_root.iterdir() if d.is_dir()] if img_root.is_dir() else []

    stats_path = src / "meta" / "episodes_stats.jsonl"
    ep_stats = {int(e["episode_index"]): e for e in load_jsonl(stats_path)} if stats_path.exists() else {}
    manifest_path = src / "meta" / "merge_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest_eps = {int(e["episode_index"]): e for e in manifest["episodes"]} if manifest else {}

    new_eps, new_ep_stats, new_manifest_eps, out_parquets = [], [], [], []
    running = 0  # global frame offset in the new pack
    for new_idx, (orig, src_task, row) in enumerate(keep):
        chunk = new_idx // chunks_size
        old_tok, new_tok = f"episode_{orig:06d}", f"episode_{new_idx:06d}"

        t = pq.read_table(src / data_tmpl.format(episode_chunk=orig // chunks_size, episode_index=orig))
        n = t.num_rows

        def put(table, name, values):
            typ = table.schema.field(name).type
            return table.set_column(table.schema.get_field_index(name), name, pa.array(values, type=typ))

        t = put(t, "episode_index", [new_idx] * n)
        t = put(t, "index", list(range(running, running + n)))  # global row index, re-based
        if "task_index" in t.schema.names:
            t = put(t, "task_index", [task_map[src_task]] * n)
        if old_tok != new_tok:
            for f in string_feats:
                vals = t.column(f).to_pylist()
                if any(v and old_tok in v for v in vals):
                    t = put(t, f, [v.replace(old_tok, new_tok) if v else v for v in vals])
        dst_pq = dst / data_tmpl.format(episode_chunk=chunk, episode_index=new_idx)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst_pq)
        out_parquets.append(dst_pq)

        for vk in video_keys:
            s = src / video_tmpl.format(episode_chunk=orig // chunks_size, episode_index=orig, video_key=vk)
            if s.exists():
                place_file(s, dst / video_tmpl.format(episode_chunk=chunk, episode_index=new_idx, video_key=vk), args.mode)

        for sub in img_subdirs:
            s = img_root / sub / old_tok
            if s.is_dir():
                place_dir(s, dst / "images" / sub / new_tok, args.mode)

        line = dict(row)
        line["episode_index"] = new_idx
        line["length"] = n
        line["dataset_from_index"] = running
        line["dataset_to_index"] = running + n - 1  # inclusive, matches source convention
        new_eps.append(line)
        if orig in ep_stats:
            st = dict(ep_stats[orig])
            st["episode_index"] = new_idx
            new_ep_stats.append(st)
        if orig in manifest_eps:
            m = dict(manifest_eps[orig])
            m["source_split_episode_index"] = orig
            m["episode_index"] = new_idx
            m["task_index"] = task_map[src_task]
            new_manifest_eps.append(m)
        running += n

    # ------------------------------------------------------------------- meta
    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    new_info = dict(info)
    new_info["total_episodes"] = len(keep)
    new_info["total_frames"] = running
    new_info["total_tasks"] = len(want)
    new_info["total_videos"] = len(keep) * len(video_keys)
    new_info["total_chunks"] = (len(keep) - 1) // chunks_size + 1
    if isinstance(info.get("splits"), dict):
        new_info["splits"] = {k: f"0:{len(keep)}" for k in info["splits"]}
    dump_json(meta / "info.json", new_info)

    dump_jsonl(meta / "episodes.jsonl", new_eps)
    dump_jsonl(meta / "tasks.jsonl", [{"task_index": task_map[t], "task": tasks[t]} for t in want])
    if new_ep_stats:
        dump_jsonl(meta / "episodes_stats.jsonl", new_ep_stats)
    if manifest is not None:
        new_manifest = dict(manifest)
        new_manifest["episodes"] = new_manifest_eps
        new_manifest["split_from"] = str(src)
        new_manifest["split_task_indices"] = want
        dump_json(meta / "merge_manifest.json", new_manifest)

    stats_files = sorted(f.name for f in (src / "meta").glob("stats*.json"))
    if args.stats == "recompute":
        subset_stats = compute_stats(out_parquets, info)
        for name in stats_files:
            dump_json(meta / name, subset_stats)
    else:
        for name in stats_files:
            shutil.copy2(src / "meta" / name, meta / name)

    handled = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl",
               "merge_manifest.json", *stats_files}
    for f in (src / "meta").iterdir():
        if f.is_file() and f.name not in handled:
            shutil.copy2(f, meta / f.name)

    print(f"\nWrote -> {dst}  ({len(keep)} eps, {running} frames, {len(want)} tasks)")
    print(f"stats*.json: {args.stats} ({', '.join(stats_files) or 'none'})")


if __name__ == "__main__":
    main()
