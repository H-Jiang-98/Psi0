#!/usr/bin/env python3
"""Split a merged g1-sonic LeRobot pack into _train / _val, holding out N episodes
per TASK GROUP (not per task -- most tasks are variants of the same skill).

Grouping comes from meta/merge_manifest.json: each episode records the
``source_dataset`` it was merged from (e.g. pick_pillow_new_3), and the base name
(trailing ``_<n>`` and a ``_new`` / ``_clean`` qualifier stripped) is the group.
For g1_sonic_lerobot_0810_merged this yields 9 groups over 50 tasks:

    pick_cloth  pick_pillow  throw_flipper  pick_place  clean
    throw_rubbish  storage  cabinat  push_chair

``--group-by task-regex`` groups by instruction text instead (no manifest needed);
the two agree on this pack and the report prints a cross-check either way.

Selection is deterministic: within each group the episode whose length is closest
to the group median is held out (``--strategy median``, the default), so val is
never a degenerate truncated take. ``random`` (with ``--seed``) and ``first`` are
also available.

Actual splitting is delegated to split_lerobot_train_val.py, which re-indexes
episodes contiguously, rewrites the parquet ``episode_index``, hardlinks videos,
and gives BOTH splits the same whole-dataset meta/stats.json so they normalize
identically.

Example
-------
    python scripts/data/split_sonic_by_task_group.py \
        --src .data/g1_sonic_lerobot_0810_merged

    # -> .data/g1_sonic_lerobot_0810_merged_train
    #    .data/g1_sonic_lerobot_0810_merged_val

    # inspect the grouping without writing anything
    python scripts/data/split_sonic_by_task_group.py --src ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Source-dataset names carry typos / qualifiers; map to readable group labels.
GROUP_ALIASES = {"cabinat": "cabinet"}

# Fallback grouping by instruction text. First match wins, so ORDER MATTERS:
# object patterns must precede destination patterns, e.g. "pillow" before
# "laundry basket" ("scoop up the pillow near the laundry basket" is pick_pillow),
# and "shoe" before "bed" / "shoe rack" appears in pick_pillow tasks too.
TASK_PATTERNS: list[tuple[str, str]] = [
    ("pillow",                              "pick_pillow"),
    ("shoes",                               "throw_flipper"),
    ("laundry basket",                      "pick_cloth"),
    ("trash can",                           "throw_rubbish"),
    ("dustpan",                             "clean"),
    ("drawer of the kitchen island",        "storage"),
    ("cabinet",                             "cabinet"),
    ("backrest of the chair",               "push_chair"),
    ("place it into",                       "pick_place"),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def base_group(source_dataset: str) -> str:
    """pick_pillow_new_3 -> pick_pillow ; clean_clean_1 -> clean ; cabinat_8 -> cabinet"""
    s = re.sub(r"_\d+$", "", source_dataset)
    s = re.sub(r"_(new|clean)$", "", s)
    return GROUP_ALIASES.get(s, s)


def group_by_task_text(task: str) -> str:
    t = task.lower()
    for needle, group in TASK_PATTERNS:
        if needle in t:
            return group
    return "UNGROUPED"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, type=Path,
                    help="Merged pack root (folder holding data/ videos/ meta/).")
    ap.add_argument("--train", type=Path, default=None, help="Default: <src>_train")
    ap.add_argument("--val", type=Path, default=None, help="Default: <src>_val")
    ap.add_argument("--per-group", type=int, default=1,
                    help="Episodes held out per task group (default 1).")
    ap.add_argument("--group-by", choices=["manifest", "task-regex"], default="manifest")
    ap.add_argument("--strategy", choices=["median", "random", "first"], default="median",
                    help="Which episode(s) to hold out within a group (default median length).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --strategy random.")
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the grouping and the chosen val episodes; write nothing.")
    args = ap.parse_args()

    src: Path = args.src
    train = args.train or src.with_name(src.name + "_train")
    val = args.val or src.with_name(src.name + "_val")

    episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    ep_len = {int(e["episode_index"]): int(e["length"]) for e in episodes}
    ep_task = {int(e["episode_index"]): (e["tasks"][0] if e.get("tasks") else "") for e in episodes}

    manifest_path = src / "meta" / "merge_manifest.json"
    manifest_groups: dict[int, str] = {}
    if manifest_path.exists():
        for e in json.loads(manifest_path.read_text())["episodes"]:
            manifest_groups[int(e["episode_index"])] = base_group(e["source_dataset"])

    if args.group_by == "manifest":
        if not manifest_groups:
            raise SystemExit(
                f"No {manifest_path}; re-run with --group-by task-regex (verify the report!)"
            )
        groups_of = manifest_groups
    else:
        groups_of = {i: group_by_task_text(t) for i, t in ep_task.items()}

    # cross-check the two independent groupings when both are available
    if manifest_groups:
        text_groups = {i: group_by_task_text(t) for i, t in ep_task.items()}
        disagree = {i for i in manifest_groups if manifest_groups[i] != text_groups[i]}
        if disagree:
            print(f"WARNING: manifest and task-text grouping disagree on {len(disagree)} episodes")
            for i in sorted(disagree)[:10]:
                print(f"  ep{i:04d} manifest={manifest_groups[i]} text={text_groups[i]}  {ep_task[i][:60]}")
        else:
            print(f"Cross-check OK: manifest and task-text grouping agree on all {len(manifest_groups)} episodes")

    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in groups_of.items():
        by_group[g].append(i)
    for g in by_group:
        by_group[g].sort()

    if "UNGROUPED" in by_group:
        raise SystemExit(f"{len(by_group['UNGROUPED'])} episodes did not match any task pattern")

    rng = random.Random(args.seed)
    val_eps: list[int] = []
    print(f"\n{'group':16s} {'eps':>5s} {'tasks':>6s}  held out")
    for g in sorted(by_group):
        eps = by_group[g]
        if len(eps) <= args.per_group:
            raise SystemExit(f"group {g} has only {len(eps)} episodes; cannot hold out {args.per_group}")
        if args.strategy == "first":
            pick = eps[: args.per_group]
        elif args.strategy == "random":
            pick = sorted(rng.sample(eps, args.per_group))
        else:  # median length -- representative, avoids truncated takes
            ordered = sorted(eps, key=lambda i: ep_len[i])
            mid = len(ordered) // 2
            # walk outwards from the median so --per-group>1 stays near the centre
            order = sorted(range(len(ordered)), key=lambda k: abs(k - mid))
            pick = sorted(ordered[k] for k in order[: args.per_group])
        val_eps.extend(pick)
        n_tasks = len({ep_task[i] for i in eps})
        lens = ", ".join(f"ep{i}({ep_len[i]}f)" for i in pick)
        print(f"{g:16s} {len(eps):5d} {n_tasks:6d}  {lens}")

    val_eps.sort()
    n_val_tasks = len({ep_task[i] for i in val_eps})
    total_tasks = len({t for t in ep_task.values()})
    print(f"\nval: {len(val_eps)} episodes over {len(by_group)} groups "
          f"({n_val_tasks}/{total_tasks} distinct tasks), train: {len(ep_len) - len(val_eps)}")
    print(f"val episodes: {','.join(map(str, val_eps))}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    cmd = [
        sys.executable, str(Path(__file__).with_name("split_lerobot_train_val.py")),
        "--src", str(src), "--train", str(train), "--val", str(val),
        "--val-episodes", ",".join(map(str, val_eps)),
        "--mode", args.mode,
    ]
    if args.force:
        cmd.append("--force")
    print("\n+ " + " ".join(cmd) + "\n")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
