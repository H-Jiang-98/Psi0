"""Re-fill the dead-neck episodes so their state normalises to exactly zero.

Eleven episodes of g1_sonic_lerobot_0810_merged_train have a dead neck encoder, and
their observation.state was overwritten wholesale (all 45 dims, every frame) with one
constant vector -- see meta/state_zeroed.README.json. The point of that constant is to
imitate what ActionTransformerModel's state dropout produces, i.e. a NORMALISED zero.

The original vector was the midpoint of the corrected q01/q99, which normalises to 0 only
under action_norm_type=bounds_q99. Every sonic recipe (post-train included) uses plain
`bounds`, which populate_stats() resolves to min/max -- so those episodes were landing on
a per-dim spread of roughly [-0.29, +0.29] instead of 0. This rewrites the constant as the
midpoint of the same stats' min/max, which normalises to exactly 0 under `bounds`.

    normalised = 2 * (x - min) / (max - min) - 1   ->   0  iff  x = (min + max) / 2

The global normaliser is unaffected: meta/stats.json is the corrected file, computed over
the 611 clean episodes with these 11 excluded, so changing their contents cannot move it.
meta/state_zeroed_backup.npz still holds the pre-zeroing rows, so the original data is
recoverable either way.

Also refreshes these episodes' entries in meta/episodes_stats.jsonl, which still described
the data as it was BEFORE the original zeroing (non-zero std on a constant episode).

Usage:
    python scripts/data/refill_zeroed_state.py PACK [--write]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_KEY = "observation.state"


def episode_path(pack: Path, idx: int) -> Path:
    return pack / f"data/chunk-{idx // 1000:03d}/episode_{idx:06d}.parquet"


def rewrite_state(path: Path, vector: np.ndarray) -> int:
    """Set every row's observation.state to `vector`, preserving schema and compression."""
    table = pq.read_table(path)
    i = table.schema.get_field_index(STATE_KEY)
    field = table.schema.field(i)
    n = table.num_rows
    block = np.tile(vector.astype(np.float32), (n, 1))

    flat = pa.array(block.reshape(-1), type=field.type.value_type)
    if pa.types.is_fixed_size_list(field.type):
        column = pa.FixedSizeListArray.from_arrays(flat, block.shape[1])
    else:
        offsets = pa.array(np.arange(n + 1, dtype=np.int32) * block.shape[1])
        column = pa.ListArray.from_arrays(offsets, flat)

    compression = pq.ParquetFile(path).metadata.row_group(0).column(i).compression.lower()
    tmp = path.with_suffix(".refill.tmp")
    pq.write_table(table.set_column(i, field, column), tmp,
                   compression="none" if compression == "uncompressed" else compression)
    tmp.replace(path)
    return n


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".pre_refill.bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path)
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()

    meta = args.pack / "meta"
    readme_path = meta / "state_zeroed.README.json"
    if not readme_path.exists():
        raise SystemExit(f"{readme_path} not found -- this pack has no zeroed episodes")
    readme = json.loads(readme_path.read_text())
    episodes = readme["episodes"]

    stats = json.loads((meta / "stats.json").read_text())[STATE_KEY]
    lo = np.array(stats["min"], dtype=np.float64)
    hi = np.array(stats["max"], dtype=np.float64)
    new_fill = (lo + hi) / 2.0
    old_fill = np.array(readme["fill_vector"], dtype=np.float64)

    def normalised(v):
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    print(f"pack      {args.pack}")
    print(f"episodes  {episodes}")
    print(f"old fill  normalises under bounds to [{normalised(old_fill).min():+.4f}, "
          f"{normalised(old_fill).max():+.4f}]")
    print(f"new fill  normalises under bounds to [{normalised(new_fill).min():+.4f}, "
          f"{normalised(new_fill).max():+.4f}]")
    print(f"max |new normalised| = {np.abs(normalised(new_fill)).max():.3e}")
    if not args.write:
        print("\ndry run, nothing written")
        return

    rows = 0
    for idx in episodes:
        p = episode_path(args.pack, idx)
        if not p.exists():
            raise SystemExit(f"missing {p}")
        rows += rewrite_state(p, new_fill)
    print(f"\nrewrote {len(episodes)} episodes, {rows} rows")

    # episodes_stats: these episodes are one constant vector, so describe them as such.
    stats_path = meta / "episodes_stats.jsonl"
    records, touched = [], 0
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("episode_index") in episodes:
            block = rec.get("stats", rec).get(STATE_KEY)
            if isinstance(block, dict):
                for key in block:
                    value = block[key]
                    if isinstance(value, list) and len(value) == len(new_fill):
                        block[key] = ([0.0] * len(new_fill) if key == "std"
                                      else new_fill.tolist())
                touched += 1
        records.append(rec)
    if touched:
        backup(stats_path)
        stats_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"episodes_stats.jsonl: refreshed {touched} record(s)")

    backup(readme_path)
    readme["fill_vector"] = new_fill.tolist()
    readme["why"] = (
        "these 11 episodes have a dead neck encoder; the midpoint of meta/stats.json's "
        "min/max normalises to exactly 0.0 under action_norm_type=bounds, matching what "
        "ActionTransformerModel's state dropout (obs*keep) produces"
    )
    readme["fill_basis"] = (
        "midpoint of corrected min/max (bounds). Superseded the earlier midpoint of "
        "corrected q01/q99, which only normalised to 0 under bounds_q99 -- a mode no "
        "sonic recipe uses; under bounds it left these episodes spread over about "
        "[-0.29, +0.29]. Previous vector kept in state_zeroed.README.json.pre_refill.bak."
    )
    readme_path.write_text(json.dumps(readme, indent=4) + "\n")
    print("state_zeroed.README.json: fill_vector and rationale updated")


if __name__ == "__main__":
    main()
