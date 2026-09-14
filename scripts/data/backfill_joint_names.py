"""Write joint names into the meta/info.json of LeRobot packs that name nothing.

Downstream tooling (scripts/viz/viz_episode_g1_sonic.py) reads the joint order from the
dataset rather than hardcoding it, which only works if `features[<column>]["names"]` is
filled in. This script fills it in from REFERENCE_STATE_NAMES below, concatenating the
joint groups in whatever order the target pack actually stores them:

    --joint-order hand arm leg waist neck    ->  the psix_sonic_v1 order (default)
    --joint-order leg waist arm hand neck    ->  the legacy g1_sonic_lerobot_0810 order

Because --joint-order takes a list, the dataset roots go *before* it (or after a `--`):

    python scripts/data/backfill_joint_names.py .data/g1_sonic_lerobot_0810_merged_val
    python scripts/data/backfill_joint_names.py <roots...> --write \
        --column observation.state --joint-order leg waist arm hand neck

Dry run by default. Every info.json it rewrites is copied to info.json.pre_names.bak first.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# The reference joint order: `.data/psix_sonic_v1_val`'s own
# features["observation.state"]["names"], verified 2026-08-26 by rendering that pack
# through g1_body29_hand14.urdf in viser. Held here rather than read back out of the pack
# so this script depends on no dataset but the ones it is rewriting. Only the names and
# their order *within* each group are used -- the order the groups are concatenated in
# comes from --joint-order.
REFERENCE_STATE_NAMES = (
    # hand [0:14] -- thumb, middle, index; the right hand mirrors the left exactly
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    # arm [14:28]
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    # leg [28:40]
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist [40:43]
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # neck [43:45]
    "neck_yaw_joint",
    "neck_pitch_joint",
)

# Joint groups, recognised by name so the reference order inside each group is carried
# over verbatim.
GROUP_PATTERNS = {
    "hand": re.compile(r"_hand_"),
    "arm": re.compile(r"_(shoulder|elbow|wrist)_"),
    "leg": re.compile(r"_(hip|knee|ankle)_"),
    "waist": re.compile(r"^waist_"),
    "neck": re.compile(r"^neck_"),
}


def group_names() -> dict[str, list[str]]:
    """REFERENCE_STATE_NAMES, split into joint groups, order preserved.

    One key per GROUP_PATTERNS entry, each holding that group's names in the order they
    appear in REFERENCE_STATE_NAMES -- the flat 45-D vector regrouped, nothing dropped
    or reordered:

        {
            "hand":  [left_hand_thumb_0_joint, ...],                          # 14
            "arm":   [left_shoulder_pitch_joint, ...],                        # 14
            "leg":   [left_hip_pitch_joint, left_hip_roll_joint, ...],        # 12
            "waist": [waist_yaw_joint, waist_roll_joint, waist_pitch_joint],  #  3
            "neck":  [neck_yaw_joint, neck_pitch_joint],                      #  2
        }

    main() concatenates the groups --joint-order asks for, in that order, to get the
    names for one column.
    """
    names = list(REFERENCE_STATE_NAMES)
    groups = {g: [n for n in names if p.search(n)] for g, p in GROUP_PATTERNS.items()}
    unmatched = [n for n in names if not any(p.search(n) for p in GROUP_PATTERNS.values())]
    if unmatched:
        raise SystemExit(f"unrecognised names in REFERENCE_STATE_NAMES: {unmatched}")
    return groups


def backfill(root: Path, column: str, names: list[str], write: bool) -> bool:
    info_path = root / "meta/info.json"
    if not info_path.exists():
        print(f"  {root}: no meta/info.json, skipped")
        return False
    info = json.loads(info_path.read_text())
    feature = info.get("features", {}).get(column)
    if feature is None:
        print(f"  {root}: '{column}' is not in info.json features, skipped")
        return False

    shape = feature.get("shape") or []
    if len(shape) != 1 or int(shape[0]) != len(names):
        raise SystemExit(
            f"{root}: '{column}' has shape {shape} but --joint-order gives {len(names)} names"
        )
    if feature.get("names") == names:
        print(f"  {root}: '{column}' already has these names, skipped")
        return False

    was = "null" if not isinstance(feature.get("names"), list) else "different names"
    feature["names"] = names
    print(f"  {root}: '{column}' {len(names)} names (was {was})")
    if write:
        shutil.copy2(info_path, info_path.with_suffix(".json.pre_names.bak"))
        info_path.write_text(json.dumps(info, indent=4) + "\n")
        print(f"    written (backup at {info_path.name}.pre_names.bak)")
    else:
        print("    dry run, nothing written")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--column", default="observation.state",
                        help="info.json feature to name (default: %(default)s)")
    parser.add_argument("--joint-order", nargs="+", metavar="GROUP",
                        default=list(GROUP_PATTERNS),
                        help=f"joint groups in column order, from {list(GROUP_PATTERNS)} "
                             "(default: the REFERENCE_STATE_NAMES order); pass the dataset "
                             "roots before this option, or after a `--`")
    parser.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = parser.parse_args()

    groups = group_names()
    unknown = [g for g in args.joint_order if g not in groups]
    if unknown:
        raise SystemExit(
            f"--joint-order got {unknown}, which is not in {list(groups)}. "
            "--joint-order takes a list, so it swallows anything after it -- put the "
            "dataset roots before the option, or separate them with `--`."
        )
    names = [n for g in args.joint_order for n in groups[g]]
    print(f"{args.column}: " + " | ".join(f"{g}[{len(groups[g])}]" for g in args.joint_order)
          + f" = {len(names)} names")
    for root in args.roots:
        backfill(root, args.column, names, args.write)


if __name__ == "__main__":
    main()
