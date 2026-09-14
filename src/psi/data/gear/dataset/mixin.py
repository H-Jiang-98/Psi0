import numpy as np
import logging
logger = logging.getLogger(__name__)

class ShardedDatasetMixin:
    """Mixin providing generate_shards for LeRobotSingleDataset subclasses."""

    def pin_val_goal_offset(self, trajectory_id: int, step_index: int) -> None:
        """No-op for datasets without a goal-image modality.

        The mixture iterators call this on every validation sample; only the
        PsiX dataset (which carries a subgoal modality) overrides it.
        """
        return None

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        """Generate shards of trajectories.
        """
        sharded_trajectories: list[list[int]] = [[]]
        shard_lengths: list[int] = []
        curr_num_steps = 0
        curr_shard_steps = 0
        discarded_episode_indices = []
        trajectory_ids = self.trajectory_ids
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get("discarded_episode_indices", [])
            trajectory_ids = [
                tid for tid in trajectory_ids
                if tid not in discarded_episode_indices
            ]

        assert (
            len(trajectory_ids) > 0
        ), f"No valid trajectories found for dataset {self.dataset_path}"

        total_steps = int(np.sum(
            [len(self.step_filter[tid]) for tid in trajectory_ids]
        ))

        for tid in trajectory_ids:
            traj_steps = len(self.step_filter[tid])
            # If adding this trajectory would push the shard well over shard_size
            # (and the shard already has some content), close the current shard
            # first to bound shard size variance and avoid wasted decode work.
            # A lone trajectory larger than shard_size is accepted as-is since
            # trajectories are the minimum indivisible unit.
            if curr_shard_steps > 0 and curr_shard_steps + traj_steps > 2 * self.shard_size:
                shard_lengths.append(curr_shard_steps)
                sharded_trajectories.append([])
                curr_shard_steps = 0
            sharded_trajectories[-1].append(tid)
            curr_num_steps += traj_steps
            curr_shard_steps += traj_steps
            if curr_shard_steps >= self.shard_size:
                shard_lengths.append(curr_shard_steps)
                sharded_trajectories.append([])
                curr_shard_steps = 0

        # Flush the last (possibly partial) shard.
        if sharded_trajectories[-1]:
            shard_lengths.append(curr_shard_steps)
        else:
            sharded_trajectories.pop()  # empty tail shard from exact boundary hit

        assert curr_num_steps == total_steps, "Total steps not equal to the sum of trajectory lengths"
        assert len(shard_lengths) == len(sharded_trajectories), "Shard count mismatch"
        logger.debug(
            f"Total steps: {total_steps}, Shard size: {self.shard_size}, "
            f"Num shards: {len(sharded_trajectories)}, "
            f"Avg shard size: {total_steps // max(len(sharded_trajectories), 1)}"
        )
        return sharded_trajectories, np.array(shard_lengths)
