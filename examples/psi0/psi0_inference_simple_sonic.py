import os
from pathlib import Path

# Set working directory to project root (parent of examples/)
project_root = Path(__file__).resolve().parent
while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
    project_root = project_root.parent
os.chdir(project_root)

import dotenv
dotenv.load_dotenv()

import torch
import numpy as np
from pathlib import Path
from psi.utils import parse_args_to_tyro_config  #, seed_everything, move_to_device, batchify
from psi.config.config import LaunchConfig

ckpt_step = 40000
run_dir = Path(".runs/finetune/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223")
config_:LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
conf = (run_dir / "run_config.json").open("r").read()
launch_config = config_.model_validate_json(conf)

from psi.config.data_lerobot import LerobotDataConfig
data_cfg: LerobotDataConfig = launch_config.data # type: ignore

from psi.config.model_psi0 import Psi0ModelConfig
model_cfg: Psi0ModelConfig = launch_config.model # type: ignore

# Use GPU 0
DEVICE = "cuda:0"
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

from psi.models.psi0 import Psi0Model 
model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=DEVICE)
model.to(DEVICE)
model.eval()

maxmin = data_cfg.transform.field

import json
from psi.utils import resolve_path
from psi.config.transform_psi0_sonic import parse_modality_key


def action_modalities(data_cfg) -> list[tuple[str, int]]:
    """(label, dim) for each entry of data.transform.repack.action_keys.

    The repack transform concatenates these keys along the last axis in order,
    then pads to pad_action_dim -- so their widths are exactly the split points
    of the predicted/ground-truth action vector. Widths come from the run's
    stats file, the same source SonicActionStateTransform normalizes against.
    """
    repack = data_cfg.transform.repack
    with open(resolve_path(data_cfg.transform.field.stat_path), "r") as f:
        stats = json.load(f)

    modalities = []
    for key in repack.action_keys:
        base_key, idx = parse_modality_key(key)
        values = np.array(stats[base_key]["min"])
        if idx is not None:
            values = values[idx]
        modalities.append((key, int(np.atleast_1d(values).shape[0])))

    pad_dim = repack.pad_action_dim
    if pad_dim is not None and pad_dim > sum(dim for _, dim in modalities):
        modalities.append(("padding", pad_dim - sum(dim for _, dim in modalities)))
    return modalities


ACTION_MODALITIES = action_modalities(data_cfg)
labels_denormed = [label for label, _ in ACTION_MODALITIES]
action_dim = sum(dim for _, dim in ACTION_MODALITIES)
# cumulative widths, minus the trailing total -- np.split wants boundaries
action_splits = list(np.cumsum([dim for _, dim in ACTION_MODALITIES])[:-1])
print(f"action layout ({action_dim} dims): " + ", ".join(f"{l}={d}" for l, d in ACTION_MODALITIES))

vlm_processor = model.vlm_processor
transform_kwargs=dict(
    vlm_processor=vlm_processor,
    # Without this the run's img_aug=True keeps color-jittering frames at EVAL time.
    # Every reference client (psi/deploy/mock_psi0_client_*.py) passes no_aug=True.
    no_aug=True,
)
val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)
print(f"Validation dataset size: {len(val_dataset)}")

from PIL import Image
import numpy as np
np.set_printoptions(precision=4, suppress=True)

l2_xyz = []
num_eval=300

dataset = val_dataset
eps_idx = 18
skip = 30

np.random.seed(42)
random_indices = np.random.choice(len(dataset), size=min(num_eval, len(dataset)), replace=False)

start_frame_idx = val_dataset.raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
end_frame_idx = val_dataset.raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
print("number of frames: ", end_frame_idx - start_frame_idx)

avg_action_errors_denormed_list = []
# for i in random_indices:
from tqdm import tqdm
for i in tqdm(range(start_frame_idx, end_frame_idx, skip)):
    frame = val_dataset[i]
    images = frame["observations"] # List[PIL.Image.Image] # (0~255)
    batch_images = [images] # List[List[PIL.Image.Image]] batch size == 1

    instruction = frame['instruction']
    batch_instructions = [instruction] # List[str]

    states = frame['states'] # (1, 32)
    batch_states = torch.from_numpy(states).unsqueeze(0).to(DEVICE) # (B, H, D)

    pred_actions = model.predict_action(
        observations=batch_images, 
        states=batch_states, 
        instructions=batch_instructions, 
        num_inference_steps=10, 
        traj2ds=None)

    # Score in NORMALIZED space and scale by 0.5*(max-min), exactly as the trainer's
    # evaluate() does (denormalize_L1_action_err). Comparing denormalize(pred) against
    # raw_actions gives the same number on well-conditioned dims but disagrees on the
    # "ill" dims (max == min), which normalization leaves raw and denormalize() does not.
    gt_action = torch.from_numpy(frame["actions"]).unsqueeze(0).to(DEVICE)  # (1, Tp, Da) normalized
    error_l1 = (pred_actions - gt_action).detach().abs().cpu().numpy().reshape(-1, action_dim)
    error_l1_denormed = maxmin.denormalize_L1_action_err(error_l1)  # (Tp, Da)

    # action L1 errors
    avg_action_errors_denormed = error_l1_denormed.mean(0)  # (action_dim,) NOTE only if the error is L1 (linear)

    avg_lr_action_err_denormed = np.split(
        avg_action_errors_denormed, action_splits, axis=-1
    )
    avg_action_errors_denormed_list.append(avg_action_errors_denormed)

    # log metrics -- mean over the modality's dims, matching the trainer's
    # `modality_metrics[...] = mod_l1.mean(0)`. A norm() here would scale the number
    # by ~sqrt(n_dims) (8x for the 64-D latent) and is not comparable to training.
    for j in range(len(avg_lr_action_err_denormed)):
        tqdm.write(f"denormed_err_l1_{labels_denormed[j]}: {avg_lr_action_err_denormed[j].mean():.6f}")

avg_action_errors_denormed_list = np.stack(avg_action_errors_denormed_list, axis=0)
avg_action_errors_denormed = avg_action_errors_denormed_list.mean(axis=0)

avg_action_errors_denormed_split = np.split(
    avg_action_errors_denormed,
    action_splits,
    axis=-1
)

print("\n---------------------------\n")
for i in range(len(avg_action_errors_denormed_split)):
    print(f"denormed_err_l1_{labels_denormed[i]}: {avg_action_errors_denormed_split[i].mean():.6f}")