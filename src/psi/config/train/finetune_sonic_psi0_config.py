from typing import Any, Union, Annotated
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator

from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt
from psi.config import transform_psi0_sonic as ps

class DynamicDataTransform(DataTransform):
    repack: ps.SonicRepackTransform
    field: ps.SonicActionStateTransform
    model: pt.Psi0ModelTransform

class DynamicDataConfig(LerobotDataConfig):
    transform: DynamicDataTransform

class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0ModelConfig

    @model_validator(mode="after")
    def check_observation_dim(self, __context: Any) -> Self:
        repack, field = self.data.transform.repack, self.data.transform.field
        assert repack.pad_action_dim == field.pad_action_dim, (
            f"inconsistent action dim: --data.transform.repack.pad-action-dim="
            f"{repack.pad_action_dim} vs --data.transform.field.pad-action-dim="
            f"{field.pad_action_dim} (set both)"
        )
        assert repack.pad_state_dim == field.pad_state_dim, (
            f"inconsistent state dim: --data.transform.repack.pad-state-dim="
            f"{repack.pad_state_dim} vs --data.transform.field.pad-state-dim="
            f"{field.pad_state_dim} (set both)"
        )
        assert self.model.odim == repack.pad_state_dim, (
            f"inconsistent odim: --model.odim={self.model.odim} vs "
            f"--data.transform.repack.pad-state-dim={repack.pad_state_dim}"
        )
        assert self.model.action_chunk_size == repack.action_chunk_size, (
            f"inconsistent action chunk size: --model.action-chunk-size="
            f"{self.model.action_chunk_size} vs --data.transform.repack.action-chunk-size="
            f"{repack.action_chunk_size}"
        )
        if self.model.vlm_layer_indices is not None:
            assert self.model.num_blocks == len(self.model.vlm_layer_indices), (
                f"inconsistent number of blocks: num_blocks={self.model.num_blocks} but "
                f"{len(self.model.vlm_layer_indices)} vlm_layer_indices (need one VLM layer per block)"
            )
        return self
