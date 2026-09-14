"""Permute a pack's observation.state columns into another pack's joint order.

The g1_sonic_lerobot_0810_merged_* finetune packs and the psix_sonic_v1_* post-train
packs hold the same 45 joints in different orders:

    finetune   leg(12) | waist(3) | arm(14) | hand(14) | neck(2)
    posttrain  hand(14) | arm(14) | leg(12) | waist(3) | neck(2)

A model post-trained on the second and finetuned on the first sees every state dimension
permuted, so this rewrites the finetune pack into the post-train order. Both packs name
every column in meta/info.json, so the permutation is derived by matching names -- never
by hardcoded spans (see scripts/viz/viz_episode_g1_sonic.py for why that matters).

It rewrites, in place:
    data/**/*.parquet          the observation.state column
    meta/info.json             features["observation.state"]["names"]
    meta/modality.json         the whole `state` section, copied from the reference
    meta/stats*.json           every 45-long array under an "observation.state" key
    meta/episodes_stats.jsonl  the same, per episode
    meta/state_zeroed.README.json   its 45-long fill_vector
    meta/state_zeroed_backup.npz    the (N, 45) rows it holds for restore
    meta/g1_sonic_mapping.json      the source-mapping spans, plus a note

Anything else under meta/ that mentions observation.state and is not handled is reported
as a warning rather than silently left inconsistent -- a stale span or an unpermuted stat
vector is exactly the kind of thing that renders plausibly and trains wrong.

Every file it edits is backed up next to itself as *.pre_reorder.bak.

Usage:
    python scripts/data/reorder_state_columns.py PACK [--reference .data/psix_sonic_v1_train]
        [--jobs N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_KEY = "observation.state"


def state_names(pack: Path) -> list[str]:
    feature = json.loads((pack / "meta/info.json").read_text())["features"][STATE_KEY]
    names = feature.get("names")
    if not isinstance(names, list):
        raise SystemExit(
            f"{pack}/meta/info.json does not name {STATE_KEY}; run "
            "scripts/data/backfill_joint_names.py on it first"
        )
    return names


def build_perm(src: list[str], ref: list[str]) -> list[int]:
    """new[:, j] = old[:, perm[j]] so that the columns end up in `ref` order."""
    if sorted(src) != sorted(ref):
        raise SystemExit(
            "packs do not hold the same joints: "
            f"only in source {sorted(set(src) - set(ref))}, "
            f"only in reference {sorted(set(ref) - set(src))}"
        )
    if len(set(src)) != len(src):
        raise SystemExit("source names are not unique; cannot build a permutation")
    index = {name: i for i, name in enumerate(src)}
    return [index[name] for name in ref]


# ------------------------------------------------------------------------- parquet


def reorder_parquet(job: tuple[str, list[int]]) -> tuple[str, int]:
    path, perm = job
    file = Path(path)
    table = pq.read_table(file)
    idx = table.schema.get_field_index(STATE_KEY)
    if idx < 0:
        raise RuntimeError(f"{file} has no {STATE_KEY} column")
    field = table.schema.field(idx)
    values = np.asarray(table.column(idx).to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(perm):
        raise RuntimeError(f"{file}: {STATE_KEY} is {values.shape}, expected (N, {len(perm)})")
    new = np.ascontiguousarray(values[:, perm])

    flat = pa.array(new.reshape(-1), type=field.type.value_type)
    if pa.types.is_fixed_size_list(field.type):
        column = pa.FixedSizeListArray.from_arrays(flat, new.shape[1])
    else:
        offsets = pa.array(np.arange(new.shape[0] + 1, dtype=np.int32) * new.shape[1])
        column = pa.ListArray.from_arrays(offsets, flat)

    compression = pq.ParquetFile(file).metadata.row_group(0).column(idx).compression.lower()
    out = table.set_column(idx, field, column)
    tmp = file.with_suffix(".reorder.tmp")
    pq.write_table(out, tmp, compression="none" if compression == "uncompressed" else compression)
    tmp.replace(file)
    return path, new.shape[0]


# ---------------------------------------------------------------------------- meta


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".pre_reorder.bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def permute_state_stats(node, perm: list[int], hits: list[str], trail: str = "") -> int:
    """Permute every len(perm) numeric array living under an `observation.state` key."""
    count = 0
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{trail}/{key}"
            if key == STATE_KEY and isinstance(value, dict):
                for stat, vec in value.items():
                    if isinstance(vec, list) and len(vec) == len(perm) \
                            and all(isinstance(x, (int, float)) for x in vec):
                        value[stat] = [vec[i] for i in perm]
                        hits.append(f"{where}/{stat}")
                        count += 1
            else:
                count += permute_state_stats(value, perm, hits, where)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            count += permute_state_stats(item, perm, hits, f"{trail}[{i}]")
    return count


def rewrite_json(path: Path, perm: list[int]) -> list[str]:
    data = json.loads(path.read_text())
    hits: list[str] = []
    if permute_state_stats(data, perm, hits):
        backup(path)
        path.write_text(json.dumps(data, indent=4) + "\n")
    return hits


def rewrite_jsonl(path: Path, perm: list[int]) -> int:
    lines, touched = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        hits: list[str] = []
        touched += 1 if permute_state_stats(record, perm, hits) else 0
        lines.append(json.dumps(record))
    if touched:
        backup(path)
        path.write_text("\n".join(lines) + "\n")
    return touched


def rewrite_npz(path: Path, perm: list[int]) -> list[str]:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    touched = []
    for key, value in arrays.items():
        if value.ndim == 2 and value.shape[1] == len(perm):
            arrays[key] = np.ascontiguousarray(value[:, perm])
            touched.append(key)
    if touched:
        backup(path)
        np.savez_compressed(path, **arrays)
    return touched


def rewrite_mapping(path: Path, blocks: list[tuple[str, int, int]]) -> None:
    """g1_sonic_mapping.json records where each source block landed; move the spans."""
    data = json.loads(path.read_text())
    state = data.get("state")
    if not isinstance(state, dict):
        return
    spans = {"state.hand_joints": None, "state.qpos": None, "state.neck": None}
    for name, lo, hi in blocks:
        if name == "hand":
            spans["state.hand_joints"] = (lo, hi)
        elif name == "neck":
            spans["state.neck"] = (lo, hi)
    # arm+leg+waist are what used to be `qpos`; they stay contiguous, in a new inner order.
    body = [(lo, hi) for name, lo, hi in blocks if name in ("arm", "leg", "waist")]
    if body:
        spans["state.qpos"] = (min(lo for lo, _ in body), max(hi for _, hi in body))
    for key, span in spans.items():
        if key in state and span is not None:
            state[key]["start"], state[key]["end"] = span
    data.setdefault("notes", []).append(
        "observation.state was reordered to the psix_sonic_v1 column order "
        "(hand|arm|leg|waist|neck) by scripts/data/reorder_state_columns.py; "
        "state.qpos now spans arm+leg+waist and is NO LONGER the Unitree 29-DoF order. "
        "meta/info.json names every column -- read the order from there."
    )
    backup(path)
    path.write_text(json.dumps(data, indent=4) + "\n")


def blocks_of(names: list[str]) -> list[tuple[str, int, int]]:
    """Contiguous joint-group spans of a name list, for reporting and the mapping file."""
    def group(name: str) -> str:
        if "_hand_" in name:
            return "hand"
        if any(j in name for j in ("shoulder", "elbow", "wrist")):
            return "arm"
        if any(j in name for j in ("hip", "knee", "ankle")):
            return "leg"
        if name.startswith("waist"):
            return "waist"
        if name.startswith("neck"):
            return "neck"
        return "other"

    out, start = [], 0
    tags = [group(n) for n in names]
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            out.append((tags[start], start, i))
            start = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path)
    ap.add_argument("--reference", type=Path, default=Path(".data/psix_sonic_v1_train"))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_names = state_names(args.pack)
    ref_names = state_names(args.reference)
    perm = build_perm(src_names, ref_names)

    print(f"source    {args.pack}")
    for tag, lo, hi in blocks_of(src_names):
        print(f"  [{lo:>2}:{hi:>2}] {tag}")
    print(f"reference {args.reference}")
    for tag, lo, hi in blocks_of(ref_names):
        print(f"  [{lo:>2}:{hi:>2}] {tag}  <- source[{perm[lo]}:{perm[hi - 1] + 1}]")
    if perm == list(range(len(perm))):
        print("already in the reference order, nothing to do")
        return

    parquets = sorted(args.pack.rglob("data/**/*.parquet"))
    meta = args.pack / "meta"
    print(f"\n{len(parquets)} parquet files, meta at {meta}")
    if args.dry_run:
        print("dry run, nothing written")
        return

    with ProcessPoolExecutor(args.jobs) as ex:
        rows = sum(n for _, n in ex.map(reorder_parquet, [(str(p), perm) for p in parquets]))
    print(f"parquet: {len(parquets)} files, {rows} rows permuted")

    # info.json: the names ARE the order.
    info_path = meta / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][STATE_KEY]["names"] = list(ref_names)
    backup(info_path)
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"info.json: {STATE_KEY} names -> reference order")

    # modality.json: the source's qpos/hand_joints/neck spans no longer describe anything.
    mod_path, ref_mod = meta / "modality.json", args.reference / "meta/modality.json"
    if mod_path.exists() and ref_mod.exists():
        mod = json.loads(mod_path.read_text())
        mod["state"] = json.loads(ref_mod.read_text()).get("state", {})
        backup(mod_path)
        mod_path.write_text(json.dumps(mod, indent=4) + "\n")
        print(f"modality.json: state section <- {ref_mod}")

    handled = {"info.json", "modality.json"}
    for path in sorted(meta.glob("*.json")):
        if path.name in handled or path.name.endswith(".bak"):
            continue
        hits = rewrite_json(path, perm)
        if path.name == "state_zeroed.README.json":
            data = json.loads(path.read_text())
            vec = data.get("fill_vector")
            if isinstance(vec, list) and len(vec) == len(perm):
                data["fill_vector"] = [vec[i] for i in perm]
                backup(path)
                path.write_text(json.dumps(data, indent=4) + "\n")
                hits = hits + ["fill_vector"]
        if hits:
            handled.add(path.name)
            print(f"{path.name}: permuted {len(hits)} vector(s) [{', '.join(hits[:4])}]")

    for path in sorted(meta.glob("*.jsonl")):
        n = rewrite_jsonl(path, perm)
        if n:
            handled.add(path.name)
            print(f"{path.name}: permuted {n} record(s)")

    for path in sorted(meta.glob("*.npz")):
        keys = rewrite_npz(path, perm)
        if keys:
            handled.add(path.name)
            print(f"{path.name}: permuted {len(keys)} array(s)")

    mapping = meta / "g1_sonic_mapping.json"
    if mapping.exists():
        rewrite_mapping(mapping, blocks_of(ref_names))
        handled.add(mapping.name)
        print(f"{mapping.name}: state spans moved, note added")

    stale = [p.name for p in sorted(meta.iterdir())
             if p.is_file() and not p.name.endswith(".bak") and p.name not in handled
             and STATE_KEY in p.read_text(errors="ignore")]
    if stale:
        print(f"\nWARNING: these meta files mention {STATE_KEY} and were NOT changed -- "
              f"check them by hand: {', '.join(stale)}")


if __name__ == "__main__":
    main()
