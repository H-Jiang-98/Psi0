from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import logging

class _SuppressV21Warning(logging.Filter):
    """Drop lerobot's "your dataset is in v2.0 format" nag.

    Our datasets are deliberately v2.0 (global stats): nothing here reads
    meta/episodes_stats.jsonl, so converting them buys nothing and the message
    is pure noise on every dataset open. Match on the convert command, which is
    unique to that message and -- unlike the version string -- does not move
    when lerobot bumps its codebase version.
    """

    def filter(self, record):
        return "convert_dataset_v20_to_v21" not in record.getMessage()


_v21_filter = _SuppressV21Warning()
# lerobot emits it with a bare logging.warning(), i.e. through the root logger,
# so the root filter is what actually catches it. Handler-level filters are the
# fallback for records that reach root by propagation from a child logger --
# ancestor *logger* filters never run on those.
logging.getLogger().addFilter(_v21_filter)
for _h in logging.getLogger().handlers:
    _h.addFilter(_v21_filter)

import warnings
# lerobot's pyav decode path goes through torchvision.io.VideoReader, which emits a
# UserWarning on every frame decode about torchvision video io being deprecated in favour
# of torchcodec. We deliberately use the torchvision/pyav backend (torchcodec's shared libs
# are absent in this venv -- see LeRobotDatasetWrapper below), so silence that one warning.
# Registered at import time so it is inherited by forked DataLoader workers.
warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import torch
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform

class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self, 
        data_cfg: LerobotDataConfig, 
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        dataset_meta = LeRobotDatasetMetadata(first_repo, resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
        else:
            repo_ids = first_repo
            root_dir = resolve_path(f"{data_cfg.root_dir}/{first_repo}")
            lerobot_dataset_class = LeRobotDataset

        self.base_dataset = lerobot_dataset_class(
            repo_ids,# type: ignore
            root=root_dir,
            delta_timestamps=delta_timestamps, # type: ignore
            image_transforms=None,
            # torchcodec IMPORTS in this venv (so lerobot's get_safe_default_codec
            # selects it) but dies loading libtorchcodec at first decode -- the
            # FFmpeg shared libs are absent. The sharded loader already defaults
            # to pyav for the same reason (lerobot_sharded.py); mirror it here.
            video_backend="pyav",
        )
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        return self.base_dataset[idx]

    def __len__(self):
        return len(self.base_dataset)

    def frame_task_weights(self, rules: dict[str, float]) -> torch.Tensor:
        """Per-frame sampling weight from `rules` {task substring (lower) -> weight}.

        Reads task_index straight off the parquet-backed hf_dataset (no per-row decode),
        maps it through meta.tasks, and takes the max weight over matching rules
        (1.0 when none match). Works for a single pack and for MultiLeRobotDataset
        (per-pack indices concatenated in dataset order, matching __getitem__).
        """
        import numpy as np
        packs = getattr(self.base_dataset, "_datasets", None) or [self.base_dataset]
        chunks = []
        for ds in packs:
            task_index = np.asarray(ds.hf_dataset.with_format("numpy")["task_index"]).reshape(-1)
            n_tasks = int(task_index.max()) + 1
            per_task = np.ones(n_tasks, dtype=np.float64)
            for ti, task in ds.meta.tasks.items():
                t = str(task).lower()
                ws = [w for sub, w in rules.items() if sub in t]
                if ws:
                    per_task[int(ti)] = max(ws)
            chunks.append(per_task[task_index])
        weights = np.concatenate(chunks)
        assert len(weights) == len(self), f"weights {len(weights)} != frames {len(self)}"
        boosted = weights != 1.0
        if boosted.any():
            share_uniform = boosted.mean()
            share_weighted = weights[boosted].sum() / weights.sum()
            logging.getLogger(__name__).info(
                f"task_sample_weights {rules}: {int(boosted.sum())}/{len(weights)} frames boosted "
                f"({share_uniform:.1%} of frames -> {share_weighted:.1%} of draws)"
            )
        else:
            logging.getLogger(__name__).warning(
                f"task_sample_weights {rules} matched no task in "
                f"{[ds.repo_id for ds in packs]}; sampling stays uniform"
            )
        return torch.as_tensor(weights, dtype=torch.double)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore
