from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np

logger = logging.getLogger(__name__)
# To enable debug output: logging.getLogger("psi.data.gear.dataset.lerobot_sharded").setLevel(logging.DEBUG)
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import yaml
import copy
import sys
from psi.data.gear.video_utils import get_frames_by_timestamps, get_frames_by_image_paths

from .lerobot import LE_ROBOT_EPISODE_FILENAME, ExtendedModalityConfig, LeRobotMixtureDataset, LeRobotSingleDataset
from .mixin import ShardedDatasetMixin

class ShardedLeRobotSingleDataset(ShardedDatasetMixin, LeRobotSingleDataset):
    """
    A single dataset supports sharding. Should not be used independently; 
    use ShardedLeRobotMixtureDataset with a single dataset instead.
    """

    def __init__(
        self,
        *args,
        shard_size: int = int(1e4),
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        logger.debug(f"Initializing {self.__class__.__name__}")
        super().__init__(*args, **kwargs)
        self.shard_size = shard_size
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()
        self.frames_to_load = self.get_all_frames_to_load()

        # Set shard caching properties
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self.frame_indices_map: dict[int, dict[str, np.ndarray]] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "ShardedLeRobotSingleDataset should not be used independently. " \
            "Use ShardedLeRobotMixtureDataset with a single dataset instead."
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # ThreadPoolExecutor is not picklable; recreate it in the worker
        state["_executor"] = None
        state["_cache_job"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.sharded_trajectories)

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """Get the video paths for all trajectories and all views.

        Returns:
            dict[int, dict[str, Path]]: The video paths for all trajectories.
        """
        video_paths = {}
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        """Get the parquet paths for all trajectories.

        Returns:
            dict[int, Path]: The parquet paths for all trajectories.
        """
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self.trajectory_ids
        }

    def get_all_frames_to_load(self):
        """Generate a map of video frame indices to trajectory indices."""
        all_frames_to_load = {}
        for trajectory_id in self.trajectory_ids:
            all_frames_to_load[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                filtered_indices = self.step_filter[trajectory_id]
                if len(filtered_indices) > 0:
                    frames_to_load = np.unique(
                        np.concatenate(
                            [i + np.array(self.delta_indices[key]) for i in filtered_indices]
                        )
                    )
                    # Cap within the length of the trajectory and >= 0
                    frames_to_load = frames_to_load[
                        (frames_to_load < self.trajectory_lengths[trajectory_id])
                        & (frames_to_load >= 0)
                    ]
                else:
                    frames_to_load = np.array([])
                all_frames_to_load[trajectory_id][key] = frames_to_load
        return all_frames_to_load

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        frames_to_load: dict[int, dict[str, np.ndarray]],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
    ) -> tuple[
        dict[str, np.ndarray], dict[int, int], pd.DataFrame, dict[int, dict[str, np.ndarray]]
    ]:
        logger.debug("Caching shard")
        start_time = time.time()
        assert "video" in modality_keys, "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        frame_indices_map = {}
        curr_step_index = 0
        cached_df = None
        curr_frame_index = {key: 0 for key in modality_keys["video"]}
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            # Check timestamps are in sync
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            frame_indices_map[trajectory_id] = {}
            for key in modality_keys["video"]:
                # Only load the frames that are needed
                this_frames_to_load = frames_to_load[trajectory_id][key]
                if len(this_frames_to_load) == 0:
                    continue
                load_timestamps = parquet_timestamps[this_frames_to_load]
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                # Store a mapping that frame_indices_map[trajectory_id][key][frame_index] = index_in_concat_video_frames
                frame_indices_map[trajectory_id][key] = (
                    np.ones(len(parquet_timestamps), dtype=np.int32) * -1
                )
                frame_indices_map[trajectory_id][key][this_frames_to_load] = np.arange(
                    curr_frame_index[key],
                    curr_frame_index[key] + len(this_frames_to_load),
                    dtype=np.int32,
                )
                curr_frame_index[key] += len(this_frames_to_load)
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                    video_paths[trajectory_id][key].as_posix(),
                    timestamps=load_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs or {},
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        # Concatenate the frames
        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        logger.debug(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        # Add global "index" column if missing (some dataset formats omit it)
        if "index" not in cached_df.columns:
            cached_df = cached_df.reset_index(drop=True)
            cached_df["index"] = cached_df.index
        return cached_frames, trajectory_start_indices, cached_df, frame_indices_map

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.frames_to_load,
            self.video_backend,
            self.video_backend_kwargs,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df, self.frame_indices_map = (
            self._cache_job.result()
        )
        self._cache_job = None  # Clear the future to allow memory to be freed

    def delete_cached_shard(self):
        """Delete the cached shard."""
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_video(self, trajectory_id: int, key: str, step_indices: np.ndarray) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by a base index.

        Args:
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """

        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        # Calculate the absolute indices
        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
            and self.frame_indices_map is not None
            and trajectory_id in self.frame_indices_map
            and key in self.frame_indices_map[trajectory_id]
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.frame_indices_map[trajectory_id][key][step_indices]
        assert np.all(
            indices_in_shard != -1
        ), f"Indices in shard are not loaded for {trajectory_id=}, {key=}, {step_indices=}"
        return self.cached_shard[key][indices_in_shard]

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the trajectory data."""
        assert self.cached_df is not None, "Cached dataframe is None"
        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} {self.args} {self.kwargs}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        return traj_data

class ShardedLeRobotPsiX(ShardedDatasetMixin, LeRobotSingleDataset):
    """
    Psi-X lerobot dataset: 
    Supports subgoal;
    Should not be used independently; 
    Use ShardedLeRobotMixtureDataset with a single dataset instead.   
    """

    def __init__(
        self,
        *args,
        shard_size: int = int(1e4),
        simple_pad_freeze_action: bool = False,
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        self.simple_pad_freeze_action = simple_pad_freeze_action
        super().__init__(*args, **kwargs)
        self.shard_size = shard_size
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()

        # Set shard caching properties
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "ShardedLeRobotPsiX should not be used independently. Use ShardedLeRobotMixtureDataset with a single dataset instead."
        )
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_executor"] = None
        state["_cache_job"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self.sharded_trajectories)

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """Get the video paths for all trajectories and all views.

        Returns:
            dict[int, dict[str, Path]]: The video paths for all trajectories.
        """
        video_paths = {}
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                # Skip path-column video keys (e.g. video.sub_goal whose
                # original_key resolves to a parquet string column of
                # per-segment jpg paths). They have no mp4 dir; per-step
                # decode happens via the parent LeRobotSingleDataset.get_video.
                # if self._is_path_column_video_key(key):
                #     continue
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        """Get the parquet paths for all trajectories.

        Returns:
            dict[int, Path]: The parquet paths for all trajectories.
        """
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self.trajectory_ids
        }

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
        fps: float = None,
    ) -> tuple[dict[str, np.ndarray], dict[int, int], pd.DataFrame]:
        # Optional logging to avoid stdout overhead during tight loops
        # (controlled by instance-level verbose flag)
        # Using a staticmethod, we cannot read self.verbose; defer to caller to control prints
        logger.debug("Caching shard")
        start_time = time.time()
        assert "video" in modality_keys, "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        curr_step_index = 0
        cached_df = None
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            # Check timestamps are in sync
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            for key in modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                # Path-column video keys (e.g. video.sub_goal) have no
                # mp4 path and were skipped in get_all_video_paths. Skip
                # caching here too — per-step decode happens in get_video
                # via the parent LeRobotSingleDataset path-column dispatch.
                if key not in video_paths[trajectory_id]:
                    continue
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                    video_paths[trajectory_id][key].as_posix(),
                    timestamps=parquet_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs,
                    fps=fps,
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        # Concatenate the frames
        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        logger.debug(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        # Add global "index" column if missing (some dataset formats omit it)
        if "index" not in cached_df.columns:
            cached_df = cached_df.reset_index(drop=True)
            cached_df["index"] = cached_df.index
        return cached_frames, trajectory_start_indices, cached_df

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.video_backend,
            self.video_backend_kwargs,
            self.fps,
        )

    def finish_cache_shard(self):
        """Get the cached shard."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df = self._cache_job.result()
        self._cache_job = None  # Clear the future to allow memory to be freed
        # New shard => stale per-trajectory slices; drop the memo (see
        # get_trajectory_data). Without this, get_video/get_state/get_action
        # re-derive the same DataFrame slice ~18x per sample (full boolean
        # scan + .unique() each), the dominant data-layer CPU cost.
        self._traj_cache = {}

    def delete_cached_shard(self):
        """Delete the cached shard."""
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df
        self._traj_cache = {}

    def get_trajectories_in_shard(self) -> list[int]:
        """Get the trajectories in a shard."""
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray], current_step: int | None = None) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            indices (dict[str, np.ndarray]): The indices for each modality.
            current_step (int | None): The current step in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                # Only load the data if the key is in the indices
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key], current_step
                    )
                    # Skip this sample if state or action data is empty
                    if data[key] is not None and hasattr(data[key], '__len__') and len(data[key]) == 0:
                        return None
        return data

    def get_video(self, trajectory_id: int, key: str, step_indices: np.ndarray) -> np.ndarray:
        """Get the video frames from cached shards for a trajectory by uniformly sampling from language-consistent ranges.

        Args:
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the video.
            step_indices (np.ndarray): The step indices to retrieve data for.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]

        # Get trajectory data to access language annotations (reuse if already loaded)
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        traj_data = self.get_trajectory_data(trajectory_id)
        # print("trajectory id", trajectory_id, step_indices, trajectory_index)
        
        # Get language annotations for all steps in the trajectory
        # language_key = self.language_key
        for modality in self.modality_keys:
            for modality_key in self.modality_keys[modality]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    subkey_meta = annotation_meta[subkey]
                    language_key = subkey_meta.original_key
                    break
        assert language_key is not None, "Language key not found"
        if language_key in traj_data.columns:
            language_annotations = traj_data[language_key].values
        else:
            # Fallback to original behavior if language annotations are not available
            step_indices = np.maximum(step_indices, 0)
            step_indices = np.minimum(step_indices, trajectory_length - 1)
            return self._get_video_frames(trajectory_id, key, step_indices)

        # Find language-consistent ranges and uniformly sample from them
        sampled_indices = self._uniform_sample_from_language_ranges(
            step_indices, language_annotations, trajectory_length
        )

        # Ensure the sampled indices are within the valid range
        sampled_indices = np.maximum(sampled_indices, 0)
        sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)
        # print("sampled indices", sampled_indices)
        return self._get_video_frames(trajectory_id, key, sampled_indices)

    def _get_video_frames(self, trajectory_id: int, key: str, local_indices: np.ndarray) -> np.ndarray:
        """Read frames for local (per-trajectory) step indices from the dense in-RAM
        shard cache (frames pre-decoded sequentially in get_shard). The dense cache
        turns shuffled random-access video reads into cheap array indexing — far
        cheaper than per-sample on-demand keyframe-seek decode for this access pattern.
        """
        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.shard_start_indices[trajectory_id] + local_indices
        return self.cached_shard[key][indices_in_shard]

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
        current_step: int | None = None,
    ) -> np.ndarray | list[str] | None:
        """Get the data corresponding to the modality for a trajectory by step indices.

        This method dispatches to the appropriate specialized method based on the modality.
        For the language modality, empty strings are returned if no matching data is found.

        Args:
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data (video, state, action, language, etc.).
            key (str): The key of the data.
            step_indices (np.ndarray): The step indices of the trajectory.

        Returns:
            np.ndarray | list[str] | None: The data for the specified modality.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        elif modality == "meta":
            # `meta.*` keys resolve to episodes.jsonl fields, broadcast
            # across step_indices by the parent LeRobotSingleDataset.get_meta.
            return self.get_meta(trajectory_id, key, step_indices)
        elif modality == "state":
            return self.get_state(trajectory_id, modality, key, step_indices)
        elif modality == "action":
            return self.get_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        elif modality == "lapa_action":
            return self.get_lapa_action(trajectory_id, key, step_indices)
        elif modality == "dream_actions":
            return self.get_dream_actions(trajectory_id, key, step_indices)
        elif modality == "rl_info":
            return self.get_rl_info(trajectory_id, key, step_indices)
        elif modality == "subgoal":
            return self.get_subgoal_image(trajectory_id, key, step_indices, current_step)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def pin_val_goal_offset(self, trajectory_id: int, step_index: int) -> None:
        """Pin the future-frame goal at a fixed offset for validation.

        Training keeps its random future goal, but validation must not redraw a
        different horizon on every pass or the eval curve is not comparable
        across steps and runs. Only the future-frame branch is pinned; GT
        subgoal-file selection (when enabled) is unchanged.
        """
        sub = self.modality_configs.get("subgoal")
        kw = getattr(sub, "kwargs", None) if sub is not None else None
        if not kw:
            return
        # val_future_horizon=None is the legacy contract (psi0.1's HE recipes):
        # val redraws a random horizon every pass, exactly as before the merge.
        # The neck class pins 48; a psi0.1 recipe can opt in explicitly.
        dk = kw.get("val_future_horizon")
        if dk is None:
            return
        # Deliberately NOT gated on sample_from_gt_goal_prob < 1.0: a pack whose
        # annotation column is absent falls to the future branch regardless of the
        # configured probability, and would otherwise redraw the horizon on every
        # eval pass. Pinning when the branch is never taken costs nothing:
        # get_subgoal_image consumes and clears _val_goal_step on entry.
        dk = int(dk)
        traj_len = self.trajectory_lengths[self.get_trajectory_index(trajectory_id)]
        self._val_goal_step = step_index + int(min(dk, max(1, traj_len - 1 - step_index)))

    def _pack_goal_capability(self, subgoal_key: str) -> tuple[bool, bool]:
        """(annotation COLUMN exists, has a decodable WM goal track) for THIS pack.

        Every zedmini pack shares the embodiment tag psix_g1_sonic_neck, so the
        goal-source probabilities are configured once for the tag and cannot
        differ per pack -- but the packs genuinely differ: pick_place carries
        annotated subgoal jpgs, the other eleven do not, and the WM tracks are
        being generated pack by pack. Rather than force a launcher to run a
        separate job per capability, each pack reports what it actually has and
        the unavailable branches are skipped for it.

        The first element is COLUMN PRESENCE, deliberately not row content:
        psi0.1's HE packs mix annotated and unannotated EPISODES in one pack
        (commit 13026e4), so any per-pack verdict sniffed from one row would
        misroute the other kind -- caching row 0 of whichever trajectory a worker
        touched first could strip GT subgoals from a whole mixed pack for the
        run. Per-FRAME emptiness is instead handled sample-by-sample in
        get_subgoal_image (empty/nan path -> future frame, psi0.1's own
        fallback); only a pack whose column is entirely ABSENT must be forced
        off the GT branch, because that branch asserts the column exists.
        A pack lacking the WM track dies at shard decode -- hence the second
        element.
        """
        cached = getattr(self, "_goal_capability_cache", None)
        if cached is not None:
            return cached
        try:
            col_exists = subgoal_key in self.curr_traj_data.columns
            # row-0 sniff is for the LOG LINE only; routing must not depend on it
            _row0 = bool(str(self.curr_traj_data[subgoal_key].iloc[0]).strip()) if col_exists else False
        except Exception:
            col_exists, _row0 = False, False
        wm_key = self.modality_configs["subgoal"].kwargs.get("wm_goal_video_key")
        has_wm = bool(wm_key) and wm_key in self.modality_configs["video"].modality_keys
        self._goal_capability_cache = (col_exists, has_wm)
        logger.info(
            f"[goal sources] {os.path.basename(str(self.dataset_path).rstrip('/'))}: "
            f"annotation_column={'yes' if col_exists else 'NO'}"
            f"{' (first row annotated)' if _row0 else ' (first row empty -> per-sample fallback decides)' if col_exists else ''} "
            f"wm_track={'yes' if has_wm else 'NO'}"
        )
        return self._goal_capability_cache

    def get_subgoal_image(self, trajectory_id: int, key: str, step_indices: np.ndarray, current_step: int | None = None) -> np.ndarray:
        subkey = key.replace("subgoal.", "")
        original_key = self.lerobot_modality_meta.subgoal[subkey].original_key
        if original_key is None:
            original_key = subkey
        features = self._lerobot_info_meta.get("features", {})
        feat = features.get(original_key, {})
        assert feat.get("dtype") == "string", "only supports filepath for now"

        self.curr_traj_data = self.get_trajectory_data(trajectory_id)

        assert isinstance(self.modality_configs["subgoal"], ExtendedModalityConfig)
        # Consume the validation pin here, before any early return: a dropped
        # goal image must not leak this sample's offset onto the next one.
        pinned = getattr(self, "_val_goal_step", None)
        self._val_goal_step = None
        # Which source produced this sample's goal, and how far ahead it sits.
        # Read back by the iterators into step_data["goal_source"/"goal_offset"];
        # the model transform conditions the prompt on it.
        self._last_goal_source = None
        self._last_goal_offset = None

        subgoal_prob = self.modality_configs["subgoal"].kwargs.get("subgoal_prob", 1.0)
        # Randomly load a subgoal image with probability subgoal_prob; otherwise, no subgoal is provided.
        if random.random() < (1 - subgoal_prob):
            return None

        annot_col_exists, has_wm = self._pack_goal_capability(original_key)
        # wm_source_prob: fetch the chosen goal frame (GT or future) from the WM
        # track by the same index instead of the real source.
        _kw = self.modality_configs["subgoal"].kwargs
        _wm_key = _kw.get("wm_goal_video_key")
        wm_source_prob = float(_kw.get("wm_source_prob", 0.0))
        # pinned val: real track only (draw still consumed to keep streams aligned)
        use_wm_source = (
            bool(_wm_key) and has_wm and wm_source_prob > 0.0
            and random.random() < wm_source_prob
            and pinned is None
        )
        sample_from_gt_goal_prob = self.modality_configs["subgoal"].kwargs.get("sample_from_gt_goal_prob", 1.0)
        # Only a pack whose annotation COLUMN is entirely absent is forced off the
        # GT branch (that branch asserts the column exists). A pack whose column
        # exists but holds empty paths -- all-empty (zedmini) or per-episode-mixed
        # (psi0.1's HE packs) -- keeps the configured probability and the
        # per-sample check below redistributes empties to the future branch,
        # exactly psi0.1's behaviour. The RNG draw count is identical either way
        # (use_gt always consumes one draw), so resumed streams stay aligned.
        if not annot_col_exists:
            sample_from_gt_goal_prob = 0.0
        # Randomly sample the subgoal image from the specified subgoal images with probability sample_from_gt_goal_prob;
        # otherwise, randomly sample one future frame as the subgoal image.
        use_gt = random.random() < sample_from_gt_goal_prob
        # A pinned val offset forces the future branch: the eval goal must not
        # flip between GT/future across passes. The draw above is still consumed
        # so train/val streams stay RNG-aligned.
        if pinned is not None:
            use_gt = False
        rel_paths = None
        if use_gt:
            assert original_key in self.curr_traj_data.columns
            col = self.curr_traj_data[original_key]
            assert pd.api.types.is_string_dtype(col) or col.dtype == object
            rel_paths = col.iloc[step_indices].astype(str).tolist()
        # empty/null GT path (no-annotation data) -> fall back to future-frame sampling.
        # This per-FRAME check (psi0.1's own fallback) is the authority on which
        # samples use the GT jpg; the column-existence gate above only prevents the
        # assert from firing on packs with no annotation column at all.
        if use_gt and all(p.strip() and p.strip().lower() not in ("nan", "none") for p in rel_paths):
            gi = None
            if current_step is not None and "sub_goal_frame_index" in self.curr_traj_data.columns:
                gi = int(self.curr_traj_data["sub_goal_frame_index"].iloc[current_step])
            if use_wm_source and gi is not None and 0 <= gi < len(self.curr_traj_data):
                subgoal_image = self.get_video(trajectory_id, _wm_key, [gi])
                self._last_goal_source = "wm_subgoal"
            else:
                subgoal_image = get_frames_by_image_paths(self.dataset_path, rel_paths)
                self._last_goal_source = "subgoal_jpg"
            # GT goal's temporal offset, so variable-horizon consumers can clamp T
            if gi is not None and gi >= current_step:
                self._last_goal_offset = gi - current_step
        else:
            assert current_step is not None
            kwargs = self.modality_configs["subgoal"].kwargs
            traj_len = len(self.curr_traj_data)
            video_key = "video." + key.split(".", 1)[1]
            wm_key = kwargs.get("wm_goal_video_key")
            wm_prob = float(kwargs.get("wm_goal_prob", 0.0))
            if not has_wm:
                # pack's WM track not generated yet -> its share falls back to
                # the real future frame
                wm_prob = 0.0
            # `wm_prob > 0.0` is checked BEFORE drawing so that wm_goal_prob=0
            # consumes no randomness: a run with the key configured but the
            # probability zeroed stays bit-comparable to one without the key,
            # which is what makes it usable as a control arm.
            if wm_key and wm_prob > 0.0 and random.random() < wm_prob:
                # World-model goal track. It is timestamp-aligned with the real
                # one (wm[t] = WM(src[t - H])), so index current_step + H is the
                # WM's H-ahead prediction computed FROM the current frame --
                # exactly what the robot has at deploy time. H is the only
                # horizon the track was generated for, hence fixed, not sampled.
                # This branch is deliberately allowed during validation too: its
                # offset is constant, so the eval goal stays deterministic, and
                # the val curve would otherwise never score the WM condition
                # this recipe exists to test.
                # Tail caveat: for current_step > traj_len-1-H the clamp yields
                # wm[traj_len-1] = WM(src[traj_len-1-H]), i.e. a goal computed
                # from an EARLIER frame than current_step. That is still an
                # in-distribution goal (it matches the goal-age distribution
                # measured on the robot), just not a fresh one.
                horizon = int(kwargs.get("wm_goal_horizon", 48))
                future_step = int(min(current_step + horizon, traj_len - 1))
                video_key = wm_key
                self._last_goal_source = "wm_future"
            elif pinned is not None:
                # Validation: offset fixed by pin_val_goal_offset (consumed
                # above), read from the REAL track.
                future_step = int(min(pinned, traj_len - 1))
            else:
                future_horizon = kwargs.get("future_horizon", (0, 4*self.fps))
                future_step = np.random.randint(
                    min(current_step + future_horizon[0], traj_len - 1),
                    min(current_step + future_horizon[1], traj_len)
                )
            if use_wm_source and self._last_goal_source is None:
                video_key = wm_key
                self._last_goal_source = "wm_future"
            if self._last_goal_source is None:
                self._last_goal_source = "real_future"
            self._last_goal_offset = int(future_step - current_step)
            assert video_key in self.modality_configs["video"].modality_keys, (
                f"goal video key {video_key!r} is not a decoded video modality key "
                f"{self.modality_configs['video'].modality_keys}; declare it in the pack's "
                f"meta/modality.json and (for a WM track) set wm_goal_video_key"
            )
            subgoal_image = self.get_video(trajectory_id, video_key, [future_step])
        return subgoal_image

    def get_state(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the state data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        subkey = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]

        # Build sampled indices for state aligned with language and video sampling
        # For state, select only the anchor index per 30-frame chunk (stride 30):
        # [..., first_idx-30, first_idx, first_idx+30, ...]
        # Stop on language change at the step anchor, bounds, or when reaching 16 anchors (to match 16 chunks).
        # trajectory_index = self.get_trajectory_index(trajectory_id)
        # trajectory_length = self.trajectory_lengths[trajectory_index]
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        # language_key = self.language_key
        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = None
        for modality_name in self.modality_keys:
            for modality_key in self.modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    subkey_meta = annotation_meta[subkey]
                    language_key = subkey_meta.original_key
                    break
        if language_key is not None and language_key in traj_data.columns and len(step_indices) > 0:
            language_annotations = traj_data[language_key].values
            first_idx = max(0, min(int(step_indices[0]), max_length - 1))
            target_language = language_annotations[first_idx]
            
            # Get the number of chunks from video sampling to ensure alignment
            if hasattr(self, '_current_num_chunks') and first_idx in self._current_num_chunks:
                target_num_chunks: int = self._current_num_chunks[first_idx]
            else:
                target_num_chunks = 1

            max_frames = self.max_chunk_size or sys.maxsize
            sampled_indices: list[int] = step_indices.tolist()
            for _ in range(target_num_chunks - 1):
                next_start = sampled_indices[-1] + self.action_steps_per_chunk
                sampled_indices.append(next_start)
            
            sampled_indices = np.array(sampled_indices, dtype=int)
            valid = sampled_indices < max_length
            # valid = valid & (language_annotations[np.minimum(sampled_indices, max_length - 1)] == target_language)
            last_valid = int(sampled_indices[valid].max())
            sampled_indices = np.where(valid, sampled_indices, last_valid)
            assert sampled_indices.size <= max_frames, f"Sampled too many actions"
        else:
            # Fallback: use provided indices with bounds
            sampled_indices = np.maximum(step_indices, 0)
            sampled_indices = np.minimum(sampled_indices, max_length - 1)

        # print("sampled indices for state", sampled_indices)

        # Pad the data using the computed sampled indices
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Get the action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # Note [YL]: this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        subkey = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]

        # Build sampled indices for action aligned with language and video sampling
        # Action runs at 30fps, so for each ±30-frame step around first_idx,
        # collect a 30-length chunk with stride 1: [anchor ... anchor+29].
        # Stop on language change at the step anchor, bounds, or when reaching 480 frames (16 chunks * 30).
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        # traj_data = (
        #     self.curr_traj_data
        #     if getattr(self, "curr_traj_data", None) is not None
        #     else self.get_trajectory_data(trajectory_id)
        # )
        # language_key = self.language_key
        traj_data = self.get_trajectory_data(trajectory_id)
        language_key = None
        for modality_name in self.modality_keys:
            for modality_key in self.modality_keys[modality_name]:
                if modality_key.startswith("annotation."):
                    subkey = modality_key.replace("annotation.", "")
                    annotation_meta = self.lerobot_modality_meta.annotation
                    subkey_meta = annotation_meta[subkey]
                    language_key = subkey_meta.original_key
                    break
        
        # HACK songlin: skip chunkwise loading 
        chunkwise_loading = self.max_chunk_size is not None 
        if chunkwise_loading and language_key is not None and language_key in traj_data.columns and len(step_indices) > 0:
            language_annotations = traj_data[language_key].values
            first_idx = max(0, min(int(step_indices[0]), trajectory_length - 1))
            target_language = language_annotations[first_idx]
            
            # Get the number of chunks from video sampling to ensure alignment
            if hasattr(self, '_current_num_chunks') and first_idx in self._current_num_chunks:
                target_num_chunks: int = self._current_num_chunks[first_idx]
            else:
                target_num_chunks = 1

            max_frames = self.action_steps_per_chunk * self.max_chunk_size
            sampled_indices: list[int] = step_indices.tolist()
            for _ in range(target_num_chunks - 1):
                next_start = sampled_indices[-1] + 1
                sampled_indices.extend(range(next_start, next_start + self.action_steps_per_chunk))

            sampled_indices = np.array(sampled_indices, dtype=int)
            valid = sampled_indices < trajectory_length
            # valid = valid & (language_annotations[np.minimum(sampled_indices, trajectory_length - 1)] == target_language)
            last_valid = int(sampled_indices[valid].max())
            sampled_indices = np.where(valid, sampled_indices, last_valid)
            assert sampled_indices.size <= max_frames, f"Sampled too many actions"
        else:
            # Fallback: use provided indices with bounds
            sampled_indices = np.maximum(step_indices, 0)
            sampled_indices = np.minimum(sampled_indices, trajectory_length - 1)

        # print("sampled indices for action", first_idx, sampled_indices, trajectory_length)

        # Pad the data using the computed sampled indices
        action_data = self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

        # Calculate relative action on the fly if relative_action is enabled
        # Only apply to keys that are in relative_action_keys
        subkey = key.replace("action.", "")
        should_convert_to_relative = (
            (self.relative_action or self.relative_action_per_horizon)  
            and len(sampled_indices) > 0
            and (self.relative_action_keys is None or subkey in self.relative_action_keys)
        )
        if should_convert_to_relative:
            # print("action data before convert", action_data[0], action_data[-1], key)
            action_data = self._convert_to_relative_action(
                action_data=action_data,
                action_key=key,
                sampled_indices=sampled_indices,
                trajectory_id=trajectory_id,
                chunk_size=self.action_steps_per_chunk,
            )
            # print("action data after convert", action_data[0], action_data[-1], key)
        
        return action_data
    
    def _convert_to_relative_action(
        self,
        action_data: np.ndarray,
        action_key: str,
        sampled_indices: np.ndarray,
        trajectory_id: int,
        chunk_size: int = 24,
    ) -> np.ndarray:
        """Convert absolute action to relative action by subtracting reference state.
        
        Args:
            action_data: Absolute action data, shape (T, D)
            action_key: The action key (e.g., 'action.left_arm_joints')
            sampled_indices: The sampled indices for the action
            trajectory_id: The trajectory ID
            chunk_size: Size of each action chunk (default 24)
            
        Returns:
            np.ndarray: Relative action data, shape (T, D)
        """
        # Get corresponding state key (assume state key matches action key)
        state_key = action_key.replace("action.", "state.")
        subkey = action_key.replace("action.", "")
        
        # Get state data from trajectory
        traj_data = self.get_trajectory_data(trajectory_id)
        le_state_cfg = getattr(self.lerobot_modality_meta, "state", None)
        
        if le_state_cfg is None or subkey not in le_state_cfg:
            # If no corresponding state key, return original action data
            return action_data
        
        le_state_key = le_state_cfg[subkey].original_key
        if le_state_key is None:
            le_state_key = subkey
        
        if le_state_key not in traj_data.columns:
            # If state column doesn't exist, return original action data
            return action_data
        
        # Get state data array
        state_array: np.ndarray = np.stack(traj_data[le_state_key])
        if state_array.ndim == 1:
            state_array = state_array.reshape(-1, 1)
        
        # Apply same indices as action
        le_indices = np.arange(
            le_state_cfg[subkey].start,
            le_state_cfg[subkey].end,
        )
        state_array = state_array[:, le_indices]
        
        # Calculate relative action for each chunk
        relative_action_data = action_data.copy()
        num_chunks = len(sampled_indices) // chunk_size
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size
            
            # Get anchor index (first index of the chunk)
            anchor_idx = sampled_indices[chunk_start]
            
            # Get reference state at anchor index
            if anchor_idx < len(state_array):
                reference_state = state_array[anchor_idx]
                
                # Subtract reference state from all actions in this chunk
                relative_action_data[chunk_start:chunk_end] = (
                    action_data[chunk_start:chunk_end] - reference_state
                )
        
        return relative_action_data
    
    def _uniform_sample_from_language_ranges(
        self, 
        step_indices: np.ndarray, 
        language_annotations: np.ndarray, 
        trajectory_length: int
    ) -> np.ndarray:
        """Uniformly sample from language-consistent ranges based on the first index's language.
        
        Args:
            step_indices (np.ndarray): Original step indices to sample.
            language_annotations (np.ndarray): Language annotations for each step in the trajectory.
            trajectory_length (int): Total length of the trajectory.
            
        Returns:
            np.ndarray: New indices sampled uniformly from the language-consistent range of the first index.
        """
        if len(step_indices) == 0:
            return np.array([], dtype=int)

        # HACK songlin: if not chunk-wise loading as in dreamzero
        if not self.max_chunk_size:
            return step_indices
    
        # Use only the first index to determine the target language
        first_idx = max(0, min(step_indices[0], trajectory_length - 1))
        target_language = language_annotations[first_idx]
        
        video_frames_per_chunk = self.num_latent_per_block * 4
        assert self.action_steps_per_chunk % video_frames_per_chunk == 0

        video_stride = self.action_steps_per_chunk // video_frames_per_chunk
        assert np.all(np.diff(step_indices) == video_stride), "Step indices must be uniformly spaced"

        max_frames = video_frames_per_chunk * self.max_chunk_size + 1
        sampled_indices: list[int] = step_indices.tolist()
        for i in range(self.max_chunk_size-1):
            sampled_indices.extend(
                [
                    sampled_indices[-1] + i * video_stride
                    for i in range(1, video_frames_per_chunk + 1)
                ]
            )
            if sampled_indices[-1] >= trajectory_length:
                break

        # find max element in sampled_indices that out of bounds,
        # either because it exceeds trajectory_length or because language changes
        # replace all out-of-bounds indices with the last valid element
        sampled_indices = np.array(sampled_indices, dtype=int)
        valid = sampled_indices < trajectory_length
        # valid = valid & (language_annotations[np.minimum(sampled_indices, trajectory_length - 1)] == target_language)
        last_valid = int(sampled_indices[valid].max())
        sampled_indices = np.where(valid, sampled_indices, last_valid)
        assert sampled_indices.size <= max_frames, f"Sampled too many video frames"

        # Use first_idx as a key to track the current sample's chunk count
        if not hasattr(self, '_current_num_chunks'):
            self._current_num_chunks = {}
        self._current_num_chunks[first_idx] = (sampled_indices.size - 1) // video_frames_per_chunk
        return sampled_indices

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the trajectory data."""
        assert self.cached_df is not None, "Cached dataframe is None"

        # Fast path: the same trajectory's slice is requested ~18x per sample
        # (get_video + every get_state/get_action + subgoal). The slice is
        # immutable for the life of the cached shard, so memoize it. The memo
        # is reset in finish_cache_shard/delete_cached_shard.
        traj_cache = getattr(self, "_traj_cache", None)
        if traj_cache is None:
            traj_cache = self._traj_cache = {}
        cached = traj_cache.get(trajectory_id)
        if cached is not None:
            return cached

            # Quick verification
        if self.cached_df.empty:
            raise ValueError("cached_df is completely empty!")

        available_episodes = self.cached_df["episode_index"].unique()
        if trajectory_id not in available_episodes:
            raise ValueError(
                f"trajectory_id {trajectory_id} not found in cached_df. "
                f"Available episodes: {sorted(available_episodes)}"
            )

        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} {self.args} {self.kwargs}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        # Store in cache to avoid repeated filtering on subsequent calls.
        traj_cache[trajectory_id] = traj_data
        return traj_data

class ShardedLeRobotMixtureDataset(LeRobotMixtureDataset, IterableDataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: list[tuple[LeRobotSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
        allow_padding: bool = False,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[ShardedLeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __iter__ will yield different samples every epoch; if "val" or "test", __iter__ will yield the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            shard_sampling_rate (float): How much data per shard to sample, in a 0-1 scale.
            num_shards_to_sample (int): The number of shards to sample.
        """
        super().__init__(
            data_mixture=data_mixture,
            training=training,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
            allow_padding=allow_padding,
        )
        # Add type hint
        self.datasets: list[ShardedLeRobotSingleDataset] = self.datasets
        # Set properties
        self.shard_sampling_rate = shard_sampling_rate
        self.num_shards_to_sample = num_shards_to_sample

        # Calculate shard sampling weights
        all_shard_sampling_weights = []
        all_shards = []
        for dataset_id, (dataset, weight) in enumerate(
            zip(self.datasets, self._dataset_sampling_weights)
        ):
            shard_sampling_weights = dataset.shard_lengths / dataset.shard_lengths.sum()
            all_shard_sampling_weights.append(shard_sampling_weights * weight)
            all_shards.extend(
                [(dataset_id, shard_idx) for shard_idx in range(shard_sampling_weights.shape[0])]
            )
        all_shard_sampling_weights = np.concatenate(all_shard_sampling_weights)
        all_shard_sampling_weights /= all_shard_sampling_weights.sum()
        self._shard_sampling_weights = all_shard_sampling_weights
        self._all_shards = all_shards

        # Generate shards sample schedule for all ranks and workers
        self._shards_sample_schedule = self.generate_shards_sample_schedule()
        # Keep the full (unfiltered) schedule so filter_shards_sample_schedule
        # always partitions from the complete set, not from a previously filtered subset.
        self._full_shards_sample_schedule = self._shards_sample_schedule

        # Check shard sampling rate
        assert 0 <= shard_sampling_rate <= 1, "Shard sampling rate must be between 0 and 1"

        # Set properties for distributed training
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The dataset sampling weights."""
        return self._dataset_sampling_weights

    @property
    def shard_sampling_weights(self) -> list[np.ndarray]:
        """The weights of each shard."""
        return self._shard_sampling_weights

    @property
    def all_shards(self) -> list[tuple[int, int]]:
        """The shards to sample."""
        return self._all_shards

    @property
    def shards_sample_schedule(self) -> list[tuple[int, int]]:
        """The shards sample schedule.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        assert self._shards_sample_schedule is not None, "Shards sample schedule not set."
        return self._shards_sample_schedule

    @property
    def trajectory_sampling_weights(self):
        """The trajectory sampling weights."""
        raise ValueError("ShardedRobotMixtureDataset does not support trajectory sampling weights.")

    @property
    def primary_dataset_indices(self):
        """The primary dataset indices."""
        raise ValueError("ShardedRobotMixtureDataset does not support primary dataset indices.")

    def reset_seed(self, seed: int):
        self.seed = seed
        self._shards_sample_schedule = self.generate_shards_sample_schedule()
        self._full_shards_sample_schedule = self._shards_sample_schedule

    def generate_shards_sample_schedule(self):
        if self.training:
            rng = np.random.default_rng(self.seed)
            sampled_shard_ids = rng.choice(
                len(self.all_shards), size=self.num_shards_to_sample, p=self.shard_sampling_weights
            )
            shards_sample_schedule = [self.all_shards[i] for i in sampled_shard_ids]
            rng.shuffle(shards_sample_schedule)
            logger.debug(f"Generated shards sample schedule with {len(shards_sample_schedule)} shards .")
        else:
            from itertools import zip_longest
            per_dataset: dict[int, list] = {}
            for shard in self.all_shards:
                per_dataset.setdefault(shard[0], []).append(shard)
            shards_sample_schedule = [
                s for group in zip_longest(*per_dataset.values())
                for s in group if s is not None
            ]
        return shards_sample_schedule

    def filter_shards_sample_schedule(self):
        """Filter the shards sample schedule for each worker.

        Returns:
            list[tuple[int, int]]: The shards to sample, in (dataset_index, shard_index).
        """
        # Filter shards for each worker
        filtered_schedule = []
        worker_info = get_worker_info()
        # If we have multiple workers, further split shards among them
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert (
                self.worker_id == worker_id and self.num_workers == num_workers
            ), "Worker ID or number of workers has been changed since it was set. This is not allowed."

        num_slots = self.world_size * num_workers
        schedule = self._full_shards_sample_schedule
        # Repeat the shards so every (rank, worker) slot gets at least one shard.
        if 0 < len(schedule) < num_slots:
            reps = (num_slots + len(schedule) - 1) // len(schedule)
            schedule = schedule * reps

        for i, shard in enumerate(schedule):
            if i % num_slots == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        logger.debug(
            "Assigned %d shards for rank %d, worker %d",
            len(filtered_schedule), self.rank, worker_id,
        )
        return filtered_schedule

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            shard_lengths = dataset.shard_lengths
            assert len(shard_lengths.shape) == 1, "Shard lengths must be a 1D array"
            num_shards = shard_lengths.shape[0]
            max_shard_length = int(shard_lengths.max())
            min_shard_length = int(shard_lengths.min())
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
                "Num shards": num_shards,
                "Max shard length": max_shard_length,
                "Min shard length": min_shard_length,
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump(
            {
                "Mixture dataset": dataset_descriptions,
                "Rank": self.rank,
                "World size": self.world_size,
            }
        )

    def __iter__(self):
        """Iterate over the dataset."""

        # Not supported: balance_trajectory_weights=False
        if not self.balance_trajectory_weights:
            raise NotImplementedError(
                "balance_trajectory_weights=False is not supported. Please use balance_dataset_weights=True instead."
            )

        yield from self._iter_legacy()

    def _iter_legacy(self):
        """Original unconfigured stream path; intentionally stateful."""

        self._shards_sample_schedule = self.filter_shards_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed)
        for i, (dataset_index, shard_index) in enumerate(self.shards_sample_schedule):
            self.curr_shard_index += 1
            assert i == self.curr_shard_index, (
                f"Shard index mismatch: {i} != {self.curr_shard_index}"
            )
            dataset = self.datasets[dataset_index]
            wait_start = time.time()
            dataset.finish_cache_shard()
            wait_end = time.time()
            wait_s = wait_end - wait_start
            if wait_s > 0.1:
                logger.info(
                    f"rk={self.rank}, wk={self.worker_id}: waited {wait_s:.2f} sec "
                    f"for shard {shard_index} (ds={dataset_index})"
                )
            if self.curr_shard_index + 1 < len(self.shards_sample_schedule):
                self.cache_next_shard()
            all_steps: list[tuple[int, int]] = []
            for trajectory_id in dataset.get_trajectories_in_shard():
                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                if self.allow_padding:
                    allowed_length = dataset.trajectory_lengths[trajectory_index]
                else:
                    trajectory_length = dataset.trajectory_lengths[trajectory_index]
                    allowed_length = trajectory_length - dataset.max_delta_index - 1
                allowed_indices = dataset.step_filter[trajectory_id]
                allowed_indices = allowed_indices[allowed_indices <= allowed_length]
                for step_index in allowed_indices:
                    all_steps.append((trajectory_id, step_index))
            if self.training:
                rng.shuffle(all_steps)
            sampled_steps = all_steps[
                : int(dataset.shard_size * self.shard_sampling_rate)
            ]
            for trajectory_id, step_index in sampled_steps:
                indices = {
                    key: delta_indices + step_index
                    for key, delta_indices in dataset.delta_indices.items()
                }
                if not self.training:
                    dataset.pin_val_goal_offset(trajectory_id, step_index)
                step_data = dataset.get_step_data(
                    trajectory_id, indices, current_step=step_index
                )
                if step_data is not None:
                    step_data['episode_index'] = trajectory_id
                    step_data['frame_index'] = step_index
                    step_data['embodiment_tag'] = dataset.tag.value
                    step_data['goal_source'] = getattr(dataset, "_last_goal_source", None)
                    step_data['goal_offset'] = getattr(dataset, "_last_goal_offset", None)
                    if not self.training:
                        logger.debug(
                            f"rk={self.rank}, wk={self.worker_id} yielding: "
                            f"trajectory {trajectory_id}, step {step_index} from "
                            f"shard {shard_index} (ds={dataset_index})"
                        )
                    yield dataset.transforms(step_data)
            dataset.delete_cached_shard()

    def cache_next_shard(self):
        """Cache the next shard in a background thread."""
        try:
            next_dataset_idx, next_shard_idx = self.shards_sample_schedule[self.curr_shard_index + 1]
            self.datasets[next_dataset_idx].start_cache_shard(next_shard_idx)
        except IndexError:
            logger.debug(f"No more shard to cache. Rank {self.rank}, Worker {self.worker_id}: Next shard index {self.curr_shard_index + 1}")

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "__getitem__ is not supported for CachedRobotMixtureDataset. Please use __iter__ instead."
        )

    def __len__(self) -> int:
        """The length of the dataset."""
        total_length = 0
        for dataset_idx, _ in self.shards_sample_schedule:
            dataset = self.datasets[dataset_idx]
            total_length += int(dataset.shard_size * self.shard_sampling_rate)
        return total_length

    @property
    def total_episodes(self) -> int:
        """Total number of valid episodes across all datasets (discarded episodes excluded)."""
        return sum(
            sum(len(shard) for shard in dataset.sharded_trajectories)
            for dataset in self.datasets
        )

    @property
    def total_frames(self) -> int:
        """The total number of frames in the dataset.
            NOTE: this is different from the __len__ method which return an estimation
        """
        return sum(len(ds) for ds in self.datasets)
    
    def iter_episode(self, episode_id: int, dataset_index: int | None = None):
        """Iterate over every step of a single episode.

        Args:
            episode_id: The trajectory ID to iterate.
            dataset_index: Which dataset to search. If None, searches all datasets in order.
                           Provide this when episode IDs are not globally unique across datasets.

        Yields:
            dict: Same format as __iter__ (transforms applied, episode_index/frame_index/embodiment_tag set).

        Raises:
            ValueError: If the episode is not found.
        """
        # Locate the episode
        target_dataset = None
        target_shard_idx = None
        search = [dataset_index] if dataset_index is not None else range(len(self.datasets))
        for ds_idx in search:
            dataset = self.datasets[ds_idx]
            for shard_idx, shard in enumerate(dataset.sharded_trajectories):
                if episode_id in shard:
                    target_dataset = dataset
                    target_shard_idx = shard_idx
                    break
            if target_dataset is not None:
                break

        if target_dataset is None:
            raise ValueError(f"Episode {episode_id} not found in any dataset.")

        target_dataset.start_cache_shard(target_shard_idx)
        target_dataset.finish_cache_shard()
        try:
            trajectory_index = target_dataset.get_trajectory_index(episode_id)
            if self.allow_padding:
                allowed_length = target_dataset.trajectory_lengths[trajectory_index]
            else:
                trajectory_length = target_dataset.trajectory_lengths[trajectory_index]
                allowed_length = trajectory_length - target_dataset.max_delta_index - 1

            allowed_indices = target_dataset.step_filter[episode_id]
            allowed_indices = allowed_indices[allowed_indices <= allowed_length]

            for step_index in allowed_indices:
                indices = {
                    key: delta_indices + step_index
                    for key, delta_indices in target_dataset.delta_indices.items()
                }
                target_dataset.pin_val_goal_offset(episode_id, step_index)
                step_data = target_dataset.get_step_data(episode_id, indices, current_step=step_index)
                if step_data is not None:
                    step_data['episode_index'] = episode_id
                    step_data['frame_index'] = step_index
                    step_data['embodiment_tag'] = target_dataset.tag.value
                    step_data['goal_source'] = getattr(target_dataset, "_last_goal_source", None)
                    step_data['goal_offset'] = getattr(target_dataset, "_last_goal_offset", None)
                    yield target_dataset.transforms(step_data)
        finally:
            target_dataset.delete_cached_shard()
