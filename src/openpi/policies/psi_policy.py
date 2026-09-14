import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_psi_policy(model_config: _model.BaseModelConfig):
    """Creates a random input example for the Droid policy."""
    return {
        "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        # TODO
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class HfmInputs(transforms.DataTransformFn):
    
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/image"])

        # Create inputs dict. Do not change the keys in the dict below.
        proprio_state = np.concatenate([
            # data["observation/hand_joints"], # (14,)
            # data["observation/arm_joints"], # (14,)
            # data["observation/leg_joints"], # (15,)
            # data["observation/torso_rpy"],# (3,)
            # data["observation/base_position"][2:3], # (1,) base height

            data["states"], #(32,)
        ], axis=0)
        
        inputs = {
            "state": proprio_state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            }
        }

        def nice(x):
            return " ".join([y.lower() for y in x.split("/")[-1].split("_")]) 
        
        if "actions" in data:
            # inputs["actions"] = np.concatenate([
            #     data["actions"][:, :4], # H + RPY (4): 1 dof base height + 3 torso rpy
            #     data["actions"][:, 4:18], # ARM (14): 7 dof left arm + 7 dof right arm
            #     data["actions"][:, 18:32], # HAND (14): 7 dof left hand + 7 dof right hand
            #     # data["actions"][:, 32:], # LEG (15): 6 dof left leg + 6 dof right leg + 3 dof waist
            # ], axis=1)
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = nice(data["prompt"]) # "pick up dumpling toy and squat to put on the chair" # data["prompt"] # FIXME use natual language prompt later
        
        return inputs

@dataclasses.dataclass(frozen=True)
class HfmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # return {"actions": np.asarray(data["actions"][:, :8])}
        return data


# 80-D = hand(14) ++ body_token(64) ++ neck(2). This is the psix ordering. The psi0
# arm concatenates the same three groups as body_token ++ hand ++ neck, so anything
# slicing a checkpoint's output has to know which arm produced it.
ZEDMINI_ACTION_SLICES = {"hand": (0, 14), "body_token": (14, 78), "neck": (78, 80)}
ZEDMINI_ACTION_DIM = ZEDMINI_ACTION_SLICES["neck"][1]
# The hand DoF sit at the head of the pack's 36-D `action` column; the remaining 22
# (arm / torso_rpy / base) are superseded by the 64-D body_token on this embodiment.
ZEDMINI_HAND_DIM = ZEDMINI_ACTION_SLICES["hand"][1]


@dataclasses.dataclass(frozen=True)
class ZedminiInputs(transforms.DataTransformFn):
    """ZED-mini G1 sonic+neck packs -> pi0/pi0.5 model inputs.

    The packs ship one egocentric camera. The two wrist slots pi0.5 carries are
    zero-filled and masked off, the same way HfmInputs handles single-view G1 data.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/image"])

        inputs = {
            "state": np.asarray(data["states"], dtype=np.float32),
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions/hand" in data:
            inputs["actions"] = np.concatenate(
                [
                    np.asarray(data["actions/hand"], dtype=np.float32)[..., :ZEDMINI_HAND_DIM],
                    np.asarray(data["actions/body_token"], dtype=np.float32),
                    np.asarray(data["actions/neck"], dtype=np.float32),
                ],
                axis=-1,
            )

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt if isinstance(prompt, str) else str(np.asarray(prompt).item())

        return inputs


@dataclasses.dataclass(frozen=True)
class ZedminiOutputs(transforms.DataTransformFn):
    """Trims the model's padded action back to the real dims."""

    action_dim: int = ZEDMINI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
