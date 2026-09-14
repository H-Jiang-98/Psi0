"""Visualize G1 sonic-neck episodes in viser, driven entirely by dataset metadata.

Joint order is loaded from the dataset itself:

    meta/info.json      features["observation.state"]["names"]
                        -> one URDF joint name per state column
    meta/modality.json  the state/action blocks (qpos, hand_joints, neck, ...)
                        -> how those columns are grouped; used for the printout and
                           for --hand-source action

Usage:
    uv run --group viz scripts/viz/viz_episode_g1_sonic.py \
        --data-dirs /path/to/dataset_a /path/to/episode_pred.pkl --episode-idx 0 [--hand-source action]
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

STATE_KEY = "observation.state"

# Joint slider resolution. Passed explicitly: viser infers `step` from the initial value
# when it is not given, which rounds the readout to whole radians.
JOINT_DIGITS = 4
JOINT_STEP = 10**-JOINT_DIGITS

# modality.json block names that --hand-source action replays over the measured state.
# These are block names, not a column layout: where each block sits, and which joint
# each of its columns drives, still comes from the dataset meta.
ACTION_OVERRIDE_BLOCKS = ("hand_joints", "neck")

# Pickle field -> modality.json state block it fills, for --names-from.
PKL_FIELDS = {"qpos": ("qpos",), "hand_joints": ("left_hand_q", "right_hand_q"),
              "neck": ("neck_state",)}


# --------------------------------------------------------------------- dataset meta


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as e:
        raise FileNotFoundError(f"cannot read {path}: {e}") from e
    except ValueError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e


def _name_list(names: object, dim: int | None) -> list[str] | None:
    """Normalise a LeRobot `names` field (a list, or a single-key dict of lists)."""
    if isinstance(names, dict):
        values = list(names.values())
        names = values[0] if len(values) == 1 else None
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return None
    if dim is not None and len(names) != dim:
        return None
    return list(names)


def feature_names(root: Path, key: str) -> list[str] | None:
    """`features[key]["names"]` from meta/info.json; None when absent or unnamed."""
    feature = _load_json(root / "meta/info.json").get("features", {}).get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape") or []
    return _name_list(feature.get("names"), shape[0] if len(shape) == 1 else None)


def state_names(root: Path) -> list[str]:
    """The URDF joint name of every `observation.state` column, from meta/info.json."""
    names = feature_names(root, STATE_KEY)
    if names is None:
        raise ValueError(
            f'{root}/meta/info.json does not name features["{STATE_KEY}"], so its '
            "joint order is unknown. This script never guesses a layout: run "
            "`python scripts/data/backfill_joint_names.py --write <root>` to write the "
            "names declared by the pack's own modality.json, then rerun."
        )
    return names


@dataclasses.dataclass(frozen=True)
class Block:
    """One modality.json entry: a named span of one dataframe column."""

    name: str
    column: str
    start: int
    end: int


def modality_blocks(root: Path, section: str, column: str | None = None) -> list[Block]:
    """meta/modality.json[section] as spans, sorted; empty when the file is absent."""
    path = root / "meta/modality.json"
    if not path.exists():
        return []
    blocks = [
        Block(name, entry.get("original_key", section), int(entry["start"]),
              int(entry["end"]))
        for name, entry in (_load_json(path).get(section) or {}).items()
        if isinstance(entry, dict) and "start" in entry and "end" in entry
    ]
    if column is not None:
        blocks = [b for b in blocks if b.column == column]
    return sorted(blocks, key=lambda b: (b.column, b.start))


def urdf_joint_names(urdf_path: str) -> list[str]:
    """Actuated joints of the URDF, i.e. the joints a state column can drive."""
    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=False)
    return list(urdf.actuated_joint_names)


# ------------------------------------------------------------------------ episodes


@dataclasses.dataclass
class Episode:
    name: str
    states: np.ndarray
    """(N, D) in the pack's own column order -- never reordered."""
    names: list[str]
    """D URDF joint names, one per state column, from meta/info.json."""
    blocks: list[Block] = dataclasses.field(default_factory=list)
    """State grouping from meta/modality.json, for the printout only."""
    base_quat: np.ndarray | None = None
    """(N, 4) wxyz, or None when the source does not carry pelvis orientation."""

    def __len__(self) -> int:
        return int(self.states.shape[0])


def _stack(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32) for x in df[column].to_numpy()])


def _replay_action_blocks(
    states: np.ndarray, names: list[str], root: Path,
    values: dict[str, np.ndarray],
) -> np.ndarray:
    """Overwrite state columns with the action columns naming the same joints.

    `values` maps a dataframe/pickle column to its (N, K) array. Which blocks are
    replayed is ACTION_OVERRIDE_BLOCKS; where they sit and what they drive comes from
    meta/modality.json + meta/info.json, so a pack with the neck inside `action` and one
    with a separate `action.neck` column both work here unchanged.
    """
    index = {name: i for i, name in enumerate(names)}
    states = states.copy()
    for block in modality_blocks(root, "action"):
        if block.name not in ACTION_OVERRIDE_BLOCKS or block.column not in values:
            continue
        array = values[block.column]
        if array.shape[1] < block.end:
            raise ValueError(
                f"{root}: modality.json action block '{block.name}' needs "
                f"{block.column}[{block.start}:{block.end}] but the column is only "
                f"{array.shape[1]} wide"
            )
        block_names = feature_names(root, block.column)
        if block_names is None:
            raise ValueError(
                f'{root}/meta/info.json does not name features["{block.column}"], so '
                f"--hand-source action cannot tell which joint block '{block.name}' "
                "drives. Backfill the names (see scripts/data/backfill_joint_names.py)."
            )
        for offset, name in enumerate(block_names[block.start:block.end]):
            if name in index:
                states[:, index[name]] = array[:, block.start + offset]
    return states


def load_lerobot_episode(
    data_dir: str, episode_idx: int, hand_source: str = "state",
) -> Episode:
    root = Path(data_dir)
    chunk = episode_idx // 1000
    parquet = root / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    df = pd.read_parquet(parquet)
    names = state_names(root)
    states = _stack(df, STATE_KEY)
    if states.shape[1] != len(names):
        raise ValueError(
            f"{root}: {STATE_KEY} has {states.shape[1]} columns but meta/info.json "
            f"names {len(names)} of them"
        )
    if hand_source == "action":
        values = {c: _stack(df, c) for c in df.columns if str(c).startswith("action")}
        states = _replay_action_blocks(states, names, root, values)
    return Episode(
        name=root.name, states=states, names=names,
        blocks=modality_blocks(root, "state", STATE_KEY),
    )


# def load_replay_pickle(
#     pkl_path: str, names_from: str, hand_source: str = "state",
#     use_base_quat: bool = True,
# ) -> Episode:
#     """Load a record_sonic.py-style replay pickle (see replay_sonic.py).

#     The pickle has no meta of its own, so `names_from` supplies both the joint names and
#     the spans its fields occupy: that pack's modality.json must declare qpos,
#     hand_joints and neck over `observation.state`.
#     """
#     path = Path(pkl_path)
#     if not names_from:
#         raise ValueError(
#             f"{path} is a replay pickle and carries no meta/info.json, so its joint "
#             "order is unknown. Pass --names-from <lerobot root> whose modality.json "
#             "declares the qpos / hand_joints / neck split."
#         )
#     root = Path(names_from)
#     names = state_names(root)
#     blocks = {b.name: b for b in modality_blocks(root, "state", STATE_KEY)}
#     missing = [b for b in PKL_FIELDS if b not in blocks]
#     if missing:
#         raise ValueError(
#             f"--names-from {root} declares state blocks {sorted(blocks)}; a replay "
#             f"pickle needs {missing} to know where its fields land. Point it at a pack "
#             "with the qpos / hand_joints / neck layout."
#         )

#     with open(path, "rb") as f:
#         data = pickle.load(f)
#     ticks = data["ticks"]
#     n = len(ticks["t"])

#     absent = [k for fields in PKL_FIELDS.values() for k in fields if k not in ticks]
#     if "neck_state" in absent:
#         absent.remove("neck_state")  # optional; zero-filled below
#     if absent:
#         raise KeyError(
#             f"{path} has no measured-state fields {absent}; it was written without "
#             "them (see write_replay_pickle) and can only be streamed, not posed."
#         )

#     states = np.zeros((n, len(names)), dtype=np.float32)
#     for block_name, fields in PKL_FIELDS.items():
#         block = blocks[block_name]
#         width = block.end - block.start
#         parts = [
#             np.asarray(ticks[f], dtype=np.float32).reshape(n, -1)
#             for f in fields if f in ticks
#         ]
#         if not parts:
#             continue  # optional field (neck) missing -> stays zero
#         values = np.concatenate(parts, axis=1)
#         if values.shape[1] != width:
#             raise ValueError(
#                 f"{path}: {'+'.join(fields)} is {values.shape[1]} wide but "
#                 f"--names-from {root} declares '{block_name}' as {width} columns"
#             )
#         states[:, block.start:block.end] = values

#     if hand_source == "action":
#         action = np.asarray(ticks["action"], dtype=np.float32).reshape(n, -1)
#         states = _replay_action_blocks(states, names, root, {"action": action})

#     base_quat = None
#     if use_base_quat and "base_quat" in ticks:
#         q = np.asarray(ticks["base_quat"], dtype=np.float64).reshape(n, 4)
#         if not np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0])):
#             base_quat = q

#     return Episode(
#         name=path.stem, states=states, names=names,
#         blocks=list(blocks.values()), base_quat=base_quat,
#     )


def load_source(
    source: str, episode_idx: int, hand_source: str = "state",
    use_base_quat: bool = True, names_from: str = "",
) -> Episode:
    # if Path(source).suffix == ".pkl":
    #     return load_replay_pickle(source, names_from, hand_source, use_base_quat)
    return load_lerobot_episode(source, episode_idx, hand_source)


# ------------------------------------------------------------------------ rendering


def state_to_cfg(
    state: np.ndarray, names: list[str], joints: set[str]
) -> dict[str, float]:
    """{urdf joint name: angle} for the state columns this URDF actually has."""
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    return {
        name: float(value)
        for name, value in zip(names, values)
        if name in joints
    }


def unposed_names(names: list[str], joints: set[str]) -> list[str]:
    """State columns with no URDF joint -- the neck on g1_body29_hand14."""
    return [name for name in names if name not in joints]


def joint_slider_range(
    limits: dict[str, tuple[float | None, float | None]], name: str, values: np.ndarray
) -> tuple[float, float]:
    """Slider bounds: the URDF limit, widened to whatever the episode actually reaches.

    Recorded angles do overshoot the URDF limits slightly, and a slider clamps what it
    displays, so the readout would quietly lie at the extremes.
    """
    lower, upper = limits.get(name, (None, None))
    low = -np.pi if lower is None else float(lower)
    high = np.pi if upper is None else float(upper)
    low = min(low, float(np.min(values))) - 1e-3
    high = max(high, float(np.max(values))) + 1e-3
    return low, high


def describe_state(episode: Episode, joints: set[str], frame: int = 0) -> str:
    values = np.asarray(episode.states[frame], dtype=np.float64).reshape(-1)
    blocks = episode.blocks or [Block("state", STATE_KEY, 0, len(episode.names))]
    covered = {i for b in blocks for i in range(b.start, b.end)}
    leftover = [i for i in range(len(episode.names)) if i not in covered]
    lines = []
    for block in list(blocks) + ([Block("unlisted", STATE_KEY, -1, -1)] if leftover else []):
        indices = leftover if block.name == "unlisted" else range(block.start, block.end)
        lines.append(f"  [{block.name}]")
        for i in indices:
            mark = "" if episode.names[i] in joints else "   (not in URDF)"
            lines.append(f"    {episode.names[i]:<32s} {values[i]:+.4f}{mark}")
    return "\n".join(lines)


class RobotView:
    """One URDF instance driven by one episode."""

    def __init__(
        self,
        server: viser.ViserServer,
        urdf_path: str,
        episode: Episode,
        x_offset: float,
        index: int = 0,
    ) -> None:
        self.name = episode.name
        self.key = f"{index}_{episode.name}"
        self.episode = episode
        self.frames = episode.states
        self.x_offset = x_offset
        # Index-prefixed so two sources with the same basename cannot collide.
        root = f"/robots/{self.key}"
        self.root_frame = server.scene.add_frame(
            root, show_axes=False, position=(x_offset, 0.0, 0.0)
        )
        self.viser_urdf = ViserUrdf(
            server,
            urdf_or_path=Path(urdf_path),
            load_meshes=True,
            load_collision_meshes=False,
            root_node_name=root,
        )
        self.joints = set(self.viser_urdf.get_actuated_joint_names())
        self.limits = self.viser_urdf.get_actuated_joint_limits()
        self.set_frame(0)

    def __len__(self) -> int:
        return len(self.episode)

    def set_frame(self, idx: int) -> dict[str, float]:
        idx = int(np.clip(idx, 0, len(self) - 1))
        cfg = state_to_cfg(self.frames[idx], self.episode.names, self.joints)
        self.viser_urdf.update_cfg(cfg)  # type: ignore[arg-type]
        if self.episode.base_quat is not None:
            self.root_frame.wxyz = tuple(self.episode.base_quat[idx])
        return cfg

    def pose(self, cfg: dict[str, float]) -> None:
        """Pose the URDF directly, e.g. from a dragged joint slider."""
        self.viser_urdf.update_cfg(cfg)  # type: ignore[arg-type]


@dataclasses.dataclass
class Args:
    data_dirs: tuple[str, ...] = (
        "/home/songlin/hfm/data/g1_sonic_lerobot_0810_merged_train",
    )
    """One or more sources: LeRobot dataset roots and/or replay .pkl files.

    Robots are drawn side by side."""
    episode_idx: int = 0
    """Episode index, used for LeRobot dataset roots only (a pickle is one episode)."""
    names_from: str = ""
    """LeRobot root supplying joint names for replay .pkl sources (which carry no meta).

    Its modality.json must declare the pickle's qpos / hand_joints / neck split."""
    hand_source: Literal["state", "action"] = "state"
    """Fingers from the measured hand state, or from the hand action (pkl + LeRobot)."""
    use_base_quat: bool = True
    """Apply ticks["base_quat"] to the pelvis (replay pickles only)."""
    urdf: str = "real/assets/g1/g1_body29_hand14.urdf"
    spacing: float = 1.0
    """Lateral offset (m) between robots when several datasets are given."""
    host: str = "0.0.0.0"
    port: int = 9000
    print_only: bool = False
    """Print the initial frame state of each dataset and exit."""


def main(args: Args) -> None:
    urdf_path = args.urdf
    if not Path(urdf_path).is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = str(repo_root / urdf_path)
    joints = set(urdf_joint_names(urdf_path))

    episodes: list[Episode] = []
    for source in args.data_dirs:
        ep = load_source(
            source, args.episode_idx, args.hand_source, args.use_base_quat,
            args.names_from,
        )
        episodes.append(ep)
        skipped = unposed_names(ep.names, joints)
        print(f"[{ep.name}] {len(ep)} frames, state dim {ep.states.shape[1]}, "
              f"hands from {args.hand_source}"
              + ("" if ep.base_quat is None else ", base_quat applied")
              + ("" if not skipped else f", not in URDF: {', '.join(skipped)}"))
        print(describe_state(ep, joints))

    if args.print_only:
        return

    server = viser.ViserServer(args.host, args.port)
    server.scene.add_grid("/grid", width=4, height=4, position=(0.0, 0.0, 0.0))

    robots: list[RobotView] = []
    n = len(episodes)
    for i, ep in enumerate(episodes):
        x = (i - (n - 1) / 2.0) * args.spacing
        robots.append(RobotView(server, urdf_path, ep, x, index=i))

    num_frames = min(len(r) for r in robots)

    state = {"playing": False, "speed": 1}

    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Play/Pause")
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0
        )
        speed_slider = server.gui.add_slider(
            "Speed (frames/tick)", min=1, max=30, step=1, initial_value=1
        )
        info_text = server.gui.add_text("Not in URDF", initial_value="", disabled=True)

    # Live joint readouts, as real sliders. They must not be add_number(..., 0.0):
    # with no explicit `step`, viser infers it from the initial value, so 0.0 gives
    # step=1 and every joint angle rendered as a flat "0". An explicit step fixes the
    # precision, and a slider also shows the angle against its URDF limit at a glance.
    joint_sliders: dict[str, dict[str, viser.GuiInputHandle[float]]] = {}
    syncing = {"active": False}

    for robot in robots:
        sliders: dict[str, viser.GuiInputHandle[float]] = {}
        first = state_to_cfg(robot.frames[0], robot.episode.names, robot.joints)
        with server.gui.add_folder(f"Joints: {robot.name}", expand_by_default=False):
            for column, jname in enumerate(robot.episode.names):
                if jname not in robot.joints:
                    continue
                low, high = joint_slider_range(
                    robot.limits, jname, robot.frames[:, column]
                )
                sliders[jname] = server.gui.add_slider(
                    jname, min=low, max=high, step=JOINT_STEP,
                    initial_value=round(float(first[jname]), JOINT_DIGITS),
                )
        joint_sliders[robot.key] = sliders

        def on_drag(_, robot: RobotView = robot, sliders=sliders) -> None:
            """Dragging a slider poses that robot -- useful while paused."""
            if syncing["active"]:  # our own write-back, not the user
                return
            robot.pose({name: float(h.value) for name, h in sliders.items()})

        for handle in sliders.values():
            handle.on_update(on_drag)

    def refresh(idx: int) -> None:
        syncing["active"] = True
        try:
            info_bits = []
            for robot in robots:
                cfg = robot.set_frame(idx)
                sliders = joint_sliders[robot.key]
                for jname, value in cfg.items():
                    rounded = round(float(value), JOINT_DIGITS)
                    if sliders[jname].value != rounded:  # skip no-op websocket traffic
                        sliders[jname].value = rounded
                frame = robot.frames[min(idx, len(robot) - 1)]
                skipped = [
                    f"{name}={frame[i]:+.3f}"
                    for i, name in enumerate(robot.episode.names)
                    if name not in robot.joints
                ]
                if skipped:
                    info_bits.append(f"{robot.name}: {' '.join(skipped)}")
            info_text.value = " | ".join(info_bits)
        finally:
            syncing["active"] = False

    @frame_slider.on_update
    def _(_) -> None:
        refresh(int(frame_slider.value))

    @speed_slider.on_update
    def _(_) -> None:
        state["speed"] = int(speed_slider.value)

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    refresh(0)
    print(f"Serving on http://{args.host}:{args.port} ({num_frames} frames)")

    while True:
        if state["playing"] and num_frames > 0:
            frame_slider.value = (
                int(frame_slider.value) + int(state["speed"])
            ) % num_frames
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main(tyro.cli(Args))
