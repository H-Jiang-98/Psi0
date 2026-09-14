import os
import dotenv
from pathlib import Path
dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Locate project root and chdir into it so relative paths (.runs/, etc.) work
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent
while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
    project_root = project_root.parent
os.chdir(project_root)

import tyro
from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path
from psi.utils import parse_args_to_tyro_config, seed_everything  #, move_to_device, batchify
from psi.config.config import LaunchConfig


@dataclass
class Args:
    run_dir: Path                  # run directory (contains argv.txt and run_config.json)
    ckpt_step: int = 40000         # checkpoint step to load
    img_aug: bool = False          # enable image augmentation; off by default (no_aug=True)
    split: str = "val"             # dataset split to evaluate
    eps_idx: int = 0               # episode index to replay
    skip: int = 1                  # frame stride within the episode
    num_inference_steps: int = 8   # flow sampling steps for predict_action


args = tyro.cli(Args)
ckpt_step = args.ckpt_step
run_dir = args.run_dir

config_:LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
conf = (run_dir / "run_config.json").open("r").read()
launch_config = config_.model_validate_json(conf)
seed_everything(launch_config.seed or 42)

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

vlm_processor = model.vlm_processor
transform_kwargs=dict(
    vlm_processor=vlm_processor,
    no_aug=not args.img_aug
)
val_dataset = data_cfg(split=args.split, transform_kwargs=transform_kwargs)
print(f"Validation dataset size: {len(val_dataset)}")

from PIL import Image
import numpy as np
np.set_printoptions(precision=4, suppress=True)

l2_xyz = []
num_eval=300

dataset = val_dataset
eps_idx = args.eps_idx
skip = args.skip
action_dim = model_cfg.action_dim

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

    states = frame['states'] # (1, Ds)
    batch_states = torch.from_numpy(states).unsqueeze(0).to(DEVICE) # (B, H, D)

    pred_actions = model.predict_action(
        observations=batch_images, 
        states=batch_states, 
        instructions=batch_instructions, 
        num_inference_steps=args.num_inference_steps,
        traj2ds=None)

    gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(DEVICE) # (6, 7)
    denormalized_pred_actions = maxmin.denormalize(pred_actions)
    # print(pred_actions[0,0,:64].cpu().numpy())
    # print(denormalized_pred_actions[0,0,:64].cpu().numpy())
    error = denormalized_pred_actions - gt_action # (B, Tp, Da)
    error_l1 = error.detach().abs().cpu().numpy().reshape(-1, action_dim) # (B*Tp, Da)

    # action L1 errors
    avg_action_errors_denormed = error_l1.mean(0)  # (Da,) NOTE only if the error is L1 (linear)

    labels_denormed = [
        "latent_action",
        "hand_joints",
        "neck_joints"
    ]

    avg_lr_action_err_denormed = np.split(
        avg_action_errors_denormed, [64,78], axis=-1
    )
    avg_action_errors_denormed_list.append(avg_action_errors_denormed)

    # log metrics
    for i in range(len(avg_lr_action_err_denormed)):
        # tqdm.write(f"denormed_err_l1_{labels_denormed[i]}: {np.linalg.norm(avg_lr_action_err_denormed[i])}") # -- old behavior l2 norm
        tqdm.write(f"denormed_err_l1_{labels_denormed[i]}: {avg_lr_action_err_denormed[i].mean()}") # -- new behavior l1 error

    # break

avg_action_errors_denormed_list = np.stack(avg_action_errors_denormed_list, axis=0)
avg_action_errors_denormed = avg_action_errors_denormed_list.mean(axis=0)

labels_denormed = [
   "latent_action",
    "hand_joints",
    "neck_joints"
]

avg_action_errors_denormed_split = np.split(
    avg_action_errors_denormed, 
    [64,78],
    axis=-1
)

print("\n---------------------------\n")
for i in range(len(avg_action_errors_denormed_split)):
    print(f"denormed_err_l1_{labels_denormed[i]}: {avg_action_errors_denormed_split[i].mean()}")