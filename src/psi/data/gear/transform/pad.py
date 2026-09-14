from typing import Any

import numpy as np
import torch
from pydantic import Field

from psi.data.gear.transform.base import ModalityTransform


class PadToMaxChunkSize(ModalityTransform):
    """
    Zero-pad a sample to exactly max_chunk_size temporal chunks so that
    variable-length samples can be stacked into a batch.

    Convention: video has num_chunks + 1 frames (first frame is the anchor),
    and action has num_chunks * actions_per_chunk steps. Per-chunk extents are
    derived from the incoming data shape.

    Emits chunk_mask (bool numpy array of shape [max_chunk_size]): True for
    real chunks, False for zero-padded chunks.

    When enabled=False this transform is a strict no-op.
    """

    apply_to: list[str] = Field(
        default_factory=list,
        description="Not used; kept for compatibility with the transform pipeline.",
    )
    enabled: bool = Field(..., description="Apply padding when True; no-op when False.")
    max_chunk_size: int = Field(..., description="Target number of temporal chunks.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data

        video = data["video"]  # numpy [T_video, V, H, W, C]
        T_video = video.shape[0]
        current_num_chunks = T_video - 1  # first frame is anchor

        if current_num_chunks <= 0:
            return data

        chunk_mask = np.zeros(self.max_chunk_size, dtype=bool)
        chunk_mask[: min(current_num_chunks, self.max_chunk_size)] = True

        if current_num_chunks >= self.max_chunk_size:
            data["chunk_mask"] = chunk_mask
            return data

        n_pad = self.max_chunk_size - current_num_chunks

        # Pad video: [max_chunk_size + 1, V, H, W, C]
        pad_shape = (n_pad,) + video.shape[1:]
        data["video"] = np.concatenate(
            [video, np.zeros(pad_shape, dtype=video.dtype)], axis=0
        )

        # Pad action: [max_chunk_size * actions_per_chunk, D_action]
        if "action" in data:
            action = data["action"]  # torch tensor [T_action, D_action]
            T_action = action.shape[0]
            assert T_action % current_num_chunks == 0, (
                f"T_action={T_action} not divisible by current_num_chunks={current_num_chunks}"
            )
            actions_per_chunk = T_action // current_num_chunks
            n_pad_steps = n_pad * actions_per_chunk
            data["action"] = torch.cat(
                [action, torch.zeros(n_pad_steps, action.shape[1], dtype=action.dtype)],
                dim=0,
            )

        data["chunk_mask"] = chunk_mask
        return data
