"""Deferred Qwen3-VL video preprocessing.

``Qwen3VLVideoProcessor._preprocess`` (resize + rescale/normalize + patchify)
dominates ``PsixModelTransform.__call__`` (~110 ms/item, single-threaded per
worker). But the only thing the rest of the processor needs on CPU is
``video_grid_thw`` (pure ``smart_resize`` arithmetic, no pixels), which drives
``<video>`` token expansion. The pixel tensor is only consumed later on GPU.

So the CPU worker (:class:`DeferredVideoProcessor`) computes ``video_grid_thw``
and passes the raw frames through as ``raw_video_frames``; :func:`gpu_video_preprocess`
runs a batched ``_preprocess`` equivalent on GPU. Nothing in ``.venv-psi`` is
modified: transformers helpers are imported, the processor is swapped per-instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.video_utils import make_batched_metadata, make_batched_videos


def _size_edges(size: Any) -> tuple[int, int]:
    """Return (shortest_edge, longest_edge) from a SizeDict or plain dict."""
    if hasattr(size, "shortest_edge"):
        return size.shortest_edge, size.longest_edge
    return size["shortest_edge"], size["longest_edge"]


def _grid_thw(num_frames: int, height: int, width: int, *, patch_size: int,
              temporal_patch_size: int, merge_size: int,
              shortest_edge: int, longest_edge: int) -> list[int]:
    """Grid math from Qwen3VLVideoProcessor._preprocess (bit-identical so the GPU
    rows line up with the tokens expanded from this grid)."""
    resized_h, resized_w = smart_resize(
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_factor=temporal_patch_size,
        factor=patch_size * merge_size,
        min_pixels=shortest_edge,
        max_pixels=longest_edge,
    )
    # temporal dim padded up to a multiple of temporal_patch_size (last frame repeated)
    padded_t = math.ceil(num_frames / temporal_patch_size) * temporal_patch_size
    grid_t = padded_t // temporal_patch_size
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    return [grid_t, grid_h, grid_w]


class DeferredVideoProcessor:
    """Stand-in for a ``Qwen3VLVideoProcessor``: computes only ``video_grid_thw``
    and returns the raw frames as ``raw_video_frames`` (pixel work deferred to
    GPU). Installed via :func:`install_deferred_video`."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        # delegate config attrs (patch_size, merge_size, size, ...)
        return getattr(self._base, name)

    def __call__(self, videos, **kwargs) -> BatchFeature:
        base = self._base
        videos = make_batched_videos(videos)
        video_metadata = make_batched_metadata(videos, kwargs.get("video_metadata"))

        if kwargs.get("do_sample_frames"):
            raise NotImplementedError(
                "DeferredVideoProcessor expects pre-sampled frames (do_sample_frames=False)."
            )

        prepared = base._prepare_input_videos(
            videos, input_data_format=kwargs.get("input_data_format")
        )
        # PsiX vision-memory path: exactly one video per sample.
        if len(prepared) != 1:
            raise NotImplementedError(
                f"DeferredVideoProcessor handles one video per sample, got {len(prepared)}."
            )

        patch = base.patch_size
        tps = base.temporal_patch_size
        merge = base.merge_size
        shortest, longest = _size_edges(base.size)

        frames = prepared[0]  # (T, C, H, W)
        T, _, H, W = frames.shape
        grid = _grid_thw(
            T, H, W,
            patch_size=patch, temporal_patch_size=tps, merge_size=merge,
            shortest_edge=shortest, longest_edge=longest,
        )

        out = BatchFeature(
            data={
                "video_grid_thw": torch.tensor([grid], dtype=torch.long),
                "raw_video_frames": frames,  # as-is (uint8) — smaller to ship to GPU
            },
            tensor_type=None,
        )
        # Qwen3VLProcessor.__call__ pops video_metadata before tensorization.
        out["video_metadata"] = video_metadata
        return out


def install_deferred_video(vlm_processor) -> None:
    """Swap ``vlm_processor.video_processor`` for a :class:`DeferredVideoProcessor`
    (idempotent, mutates only the live instance)."""
    if getattr(vlm_processor, "_psix_video_deferred", False):
        return
    vlm_processor.video_processor = DeferredVideoProcessor(vlm_processor.video_processor)
    vlm_processor._psix_video_deferred = True


@dataclass
class VideoPreprocessParams:
    """Everything the GPU kernel needs to reproduce ``_preprocess``."""
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    shortest_edge: int
    longest_edge: int
    image_mean: tuple
    image_std: tuple
    rescale_factor: float = 1.0 / 255.0
    do_rescale: bool = True
    do_normalize: bool = True
    do_resize: bool = True


def params_from_processor(video_processor) -> VideoPreprocessParams:
    """Extract preprocess params from the real (unwrapped) Qwen3VL video processor."""
    shortest, longest = _size_edges(video_processor.size)
    return VideoPreprocessParams(
        patch_size=video_processor.patch_size,
        temporal_patch_size=video_processor.temporal_patch_size,
        merge_size=video_processor.merge_size,
        shortest_edge=shortest,
        longest_edge=longest,
        image_mean=tuple(video_processor.image_mean),
        image_std=tuple(video_processor.image_std),
        rescale_factor=getattr(video_processor, "rescale_factor", 1.0 / 255.0),
        do_rescale=getattr(video_processor, "do_rescale", True),
        do_normalize=getattr(video_processor, "do_normalize", True),
        do_resize=getattr(video_processor, "do_resize", True),
    )


def gpu_video_preprocess(
    raw_video_frames: torch.Tensor,
    video_grid_thw: torch.Tensor,
    params: VideoPreprocessParams,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """GPU batched equivalent of ``Qwen3VLVideoProcessor._preprocess``
    (resize -> rescale/normalize -> patchify) in one dense pass. Assumes a uniform
    ``(T,C,H,W)`` / grid across the batch (the PsiX vision-memory path).

    Args:
        raw_video_frames: ``(B, T, C, H, W)`` on GPU (typically uint8).
        video_grid_thw: ``(B, 3)`` ``[grid_t, grid_h, grid_w]`` from the CPU stage;
            the resize target is derived from it so it can't drift from the tokens.
        params: see :class:`VideoPreprocessParams`.
        out_dtype: dtype of the result (default float32; vision tower handles autocast).

    Returns:
        ``pixel_values_videos`` ``(B*grid_t*grid_h*grid_w, C*tps*patch**2)``, blocks
        concatenated in batch order, layout matching the reference patchify.
    """
    B, T, C, H, W = raw_video_frames.shape
    patch = params.patch_size
    tps = params.temporal_patch_size
    merge = params.merge_size

    # Uniform-batch: derive the resize target from the shared grid.
    grid = video_grid_thw
    if not bool((grid == grid[0]).all()):
        raise NotImplementedError(
            "gpu_video_preprocess assumes a uniform grid across the batch; "
            f"got mixed video_grid_thw=\n{grid}"
        )
    grid_t, grid_h, grid_w = (int(grid[0, 0]), int(grid[0, 1]), int(grid[0, 2]))
    resized_h, resized_w = grid_h * patch, grid_w * patch

    # 1. resize in native dtype before rescale (matches the reference's uint8
    #    round-trip, so outputs are bit-identical).
    x = raw_video_frames.reshape(B * T, C, H, W)
    if params.do_resize and (H != resized_h or W != resized_w):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as _TVF
        x = _TVF.resize(
            x, [resized_h, resized_w],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
    x = x.to(torch.float32)

    # 2. fused rescale + normalize
    if params.do_rescale:
        x = x * params.rescale_factor
    if params.do_normalize:
        mean = torch.tensor(params.image_mean, device=x.device, dtype=x.dtype).view(1, C, 1, 1)
        std = torch.tensor(params.image_std, device=x.device, dtype=x.dtype).view(1, C, 1, 1)
        x = (x - mean) / std

    x = x.reshape(B, T, C, resized_h, resized_w)

    # 3. temporal pad up to grid_t * tps by repeating the last frame
    target_t = grid_t * tps
    if T < target_t:
        x = torch.cat([x, x[:, -1:].repeat(1, target_t - T, 1, 1, 1)], dim=1)
    elif T > target_t:
        x = x[:, :target_t]

    # 4. patchify — identical layout to the reference _preprocess
    x = x.reshape(
        B, grid_t, tps, C,
        grid_h // merge, merge, patch,
        grid_w // merge, merge, patch,
    )
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten = x.reshape(B, grid_t * grid_h * grid_w, C * tps * patch * patch)

    # concat per-video blocks along dim 0 in batch order
    pixel_values_videos = flatten.reshape(B * grid_t * grid_h * grid_w, C * tps * patch * patch)
    return pixel_values_videos.to(out_dtype)


class DeferredQwen3VLProcessor:
    """Drop-in Qwen3-VL processor that defers video pixels to GPU. Two stages::

        proc = DeferredQwen3VLProcessor.from_pretrained(path)
        inputs = proc(text=..., videos=..., return_tensors="pt")  # grid + raw frames
        inputs = proc.finalize(inputs.to("cuda"))                 # fills pixel_values_videos
        model.generate(**inputs)

    ``__call__`` returns ``video_grid_thw`` (drives token expansion) +
    ``raw_video_frames``; :meth:`finalize` runs :func:`gpu_video_preprocess`.
    Other attrs/methods delegate to the wrapped processor.
    """

    def __init__(self, base_processor, out_dtype: torch.dtype = torch.float32):
        # capture real params BEFORE install_deferred_video swaps in place
        self.processor = base_processor
        self._params = params_from_processor(base_processor.video_processor)
        install_deferred_video(self.processor)
        self.out_dtype = out_dtype

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import AutoProcessor

        out_dtype = kwargs.pop("out_dtype", torch.float32)
        return cls(AutoProcessor.from_pretrained(*args, **kwargs), out_dtype=out_dtype)

    def __getattr__(self, name):
        if name == "processor":
            raise AttributeError(name)
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def finalize(self, inputs):
        """Fill ``pixel_values_videos`` from ``raw_video_frames`` on its device.
        No-op without videos. Accepts a single ``(T,C,H,W)`` or batched ``(B,T,C,H,W)``."""
        raw = inputs.get("raw_video_frames", None)
        if raw is None:
            return inputs
        grid = inputs["video_grid_thw"]
        if raw.ndim == 4:  # (T,C,H,W) -> (1,T,C,H,W)
            raw = raw.unsqueeze(0)
        inputs["pixel_values_videos"] = gpu_video_preprocess(
            raw, grid, self._params, out_dtype=self.out_dtype
        )
        inputs.pop("raw_video_frames", None)
        return inputs
