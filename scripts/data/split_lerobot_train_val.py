#!/usr/bin/env python3
"""Split a LeRobot v3 (PsiX-style) dataset into train / val folders.

Both output folders keep the exact same on-disk layout as the source (``data/``,
``videos/``, ``images/``, ``meta/``) so either one can be loaded by the gear
dataloader with the *same* embodiment tag as the source.

What it does per split
----------------------
* Re-indexes the chosen episodes to a contiguous ``0..N-1`` range: renames the
  parquet / video / subgoal-image files and rewrites the ``episode_index`` column
  plus any string column that embeds an ``episode_XXXXXX`` path (e.g.
  ``sub_goal_image_path``). The per-episode ``index`` / ``frame_index`` columns are
  left untouched -- this dataset family stores ``index`` per-episode, so carrying
  the rows verbatim reproduces the source convention exactly.
* Rewrites ``meta/info.json`` counts and emits per-split ``meta/episodes.jsonl`` and
  ``meta/episodes_stats.jsonl`` (filtered + re-indexed).
* Copies every other ``meta/*`` file verbatim (tasks.jsonl, modality.json,
  paraphrases.json, ...).

Whole-dataset stats are preserved
---------------------------------
Both splits share ONE ``meta/stats.json`` = the statistics of the *full* source
dataset, so train and val normalize identically. If the source already has
``meta/stats.json`` it is copied byte-for-byte; otherwise it is computed once over
ALL source parquets with the same formula the gear loader uses
(``calculate_dataset_statistics``) and written identically into both splits.

Embodiment-agnostic: feature names, video keys and path templates are read from
``meta/info.json``, so this works for cleanup_table, g1_neck_0617, he_0609_psix, etc.

Example
-------
    python scripts/data/split_lerobot_train_val.py \
        --src   .data/real/cleanup_table_2026-07-01/g1 \
        --train .data/real/cleanup_table_2026-07-01_train/g1 \
        --val   .data/real/cleanup_table_2026-07-01_val/g1 \
        --val-episodes 6,10,16
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- io
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=4))


def place_file(src: Path, dst: Path, mode: str) -> None:
    """Materialize ``src`` at ``dst`` via hardlink (default), symlink, or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-device etc. -> fall back to copy
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return
    shutil.copy2(src, dst)


def place_dir(src_dir: Path, dst_dir: Path, mode: str) -> None:
    for f in sorted(src_dir.rglob("*")):
        if f.is_file():
            place_file(f, dst_dir / f.relative_to(src_dir), mode)


# ------------------------------------------------------------------------ stats
def compute_full_stats(src: Path, info: dict) -> dict:
    """Full-dataset stats over every source parquet -- mirrors the gear loader's
    ``calculate_dataset_statistics`` (float features only, mean/std/min/max/q01/q99)."""
    float_feats = [k for k, v in info["features"].items() if "float" in v["dtype"]]
    parquet_files = sorted((src / "data").rglob("episode_*.parquet"))
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
            "max": np.max(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


# ------------------------------------------------------------------------ split
def build_split(
    src: Path,
    dst: Path,
    orig_indices: list[int],
    info: dict,
    ep_meta: dict[int, dict],
    stats_meta: dict[int, dict],
    stats_json_obj: dict,
    mode: str,
) -> dict:
    """Write one split. ``orig_indices`` are source episode indices (already sorted)."""
    chunks_size = info["chunks_size"]
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    string_feats = [k for k, v in info["features"].items() if v["dtype"] == "string"]
    img_root = src / "images"
    img_subdirs = [d.name for d in img_root.iterdir() if d.is_dir()] if img_root.is_dir() else []

    new_eps, new_stats = [], []
    running = 0  # global frame offset within this split
    for new_idx, orig in enumerate(orig_indices):
        chunk = new_idx // chunks_size
        old_tok, new_tok = f"episode_{orig:06d}", f"episode_{new_idx:06d}"

        # --- parquet: relabel episode_index (+ image-path strings), keep everything else ---
        src_pq = src / data_tmpl.format(episode_chunk=orig // chunks_size, episode_index=orig)
        t = pq.read_table(src_pq)
        n = t.num_rows
        ei = t.schema.get_field_index("episode_index")
        t = t.set_column(ei, "episode_index",
                         pa.array([new_idx] * n, type=t.schema.field("episode_index").type))
        if old_tok != new_tok:
            for f in string_feats:
                vals = t.column(f).to_pylist()
                if any(v and old_tok in v for v in vals):
                    vals = [v.replace(old_tok, new_tok) if v else v for v in vals]
                    fi = t.schema.get_field_index(f)
                    t = t.set_column(fi, f, pa.array(vals, type=t.schema.field(f).type))
        dst_pq = dst / data_tmpl.format(episode_chunk=chunk, episode_index=new_idx)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst_pq)

        # --- videos (one file per video key) ---
        for vk in video_keys:
            s = src / video_tmpl.format(episode_chunk=orig // chunks_size, episode_index=orig, video_key=vk)
            if s.exists():
                place_file(s, dst / video_tmpl.format(episode_chunk=chunk, episode_index=new_idx, video_key=vk), mode)

        # --- subgoal image folders ---
        for sub in img_subdirs:
            s = img_root / sub / old_tok
            if s.is_dir():
                place_dir(s, dst / "images" / sub / new_tok, mode)

        # --- meta rows ---
        line = dict(ep_meta[orig])
        line["episode_index"] = new_idx
        line["length"] = n
        line["dataset_from_index"] = running
        line["dataset_to_index"] = running + n - 1  # inclusive, matches source convention
        new_eps.append(line)
        running += n
        if orig in stats_meta:
            st = dict(stats_meta[orig])
            st["episode_index"] = new_idx
            new_stats.append(st)

    # --- meta files ---
    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    new_info = dict(info)
    new_info["total_episodes"] = len(orig_indices)
    new_info["total_frames"] = running
    new_info["total_videos"] = len(orig_indices) * len(video_keys)
    new_info["total_chunks"] = (len(orig_indices) - 1) // chunks_size + 1 if orig_indices else 0
    dump_json(meta / "info.json", new_info)
    dump_jsonl(meta / "episodes.jsonl", new_eps)
    if new_stats:
        dump_jsonl(meta / "episodes_stats.jsonl", new_stats)
    dump_json(meta / "stats.json", stats_json_obj)  # full-dataset stats, identical in both splits

    handled = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"}
    for f in (src / "meta").iterdir():
        if f.is_file() and f.name not in handled:
            shutil.copy2(f, meta / f.name)

    return {"episodes": len(orig_indices), "frames": running}


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Source dataset root (the folder holding data/ videos/ meta/).")
    ap.add_argument("--train", required=True, type=Path, help="Output train root.")
    ap.add_argument("--val", required=True, type=Path, help="Output val root.")
    ap.add_argument("--val-episodes", required=True,
                    help="Comma-separated SOURCE episode indices to put in val, e.g. '6,10,16'.")
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink",
                    help="How to place videos/images (parquet is always rewritten). Default hardlink.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing train/val output dirs.")
    args = ap.parse_args()

    src: Path = args.src
    info = json.loads((src / "meta" / "info.json").read_text())
    ep_meta = {int(e["episode_index"]): e for e in load_jsonl(src / "meta" / "episodes.jsonl")}
    stats_path = src / "meta" / "episodes_stats.jsonl"
    stats_meta = ({int(e["episode_index"]): e for e in load_jsonl(stats_path)} if stats_path.exists() else {})

    all_eps = sorted(ep_meta)
    val_eps = sorted(int(x) for x in args.val_episodes.split(",") if x.strip() != "")
    bad = set(val_eps) - set(all_eps)
    if bad:
        raise SystemExit(f"--val-episodes not in source {all_eps}: {sorted(bad)}")
    train_eps = [e for e in all_eps if e not in set(val_eps)]

    for out in (args.train, args.val):
        if out.exists():
            if not args.force:
                raise SystemExit(f"Output exists (use --force to overwrite): {out}")
            shutil.rmtree(out)

    # whole-dataset stats: reuse source stats.json if present, else compute once over ALL parquets.
    src_stats = src / "meta" / "stats.json"
    if src_stats.exists():
        stats_json_obj = json.loads(src_stats.read_text())
        stats_origin = f"copied from {src_stats}"
    else:
        stats_json_obj = compute_full_stats(src, info)
        stats_origin = "computed over full source dataset"

    print(f"Source : {src}  ({len(all_eps)} episodes)")
    print(f"Stats  : {stats_origin}  ({len(stats_json_obj)} features)")
    print(f"Val    : {val_eps}")
    print(f"Train  : {train_eps}")

    tr = build_split(src, args.train, train_eps, info, ep_meta, stats_meta, stats_json_obj, args.mode)
    va = build_split(src, args.val, val_eps, info, ep_meta, stats_meta, stats_json_obj, args.mode)
    print(f"\nTrain -> {args.train}  ({tr['episodes']} eps, {tr['frames']} frames)")
    print(f"Val   -> {args.val}  ({va['episodes']} eps, {va['frames']} frames)")


if __name__ == "__main__":
    main()
