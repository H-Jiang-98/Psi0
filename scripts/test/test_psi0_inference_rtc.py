import os
from pathlib import Path

project_root = Path.cwd().resolve()
while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
    project_root = project_root.parent

os.chdir(project_root)
print(f"project root dir changed to {project_root}.")

import dotenv
assert dotenv.load_dotenv(), ".env not loaded"

import torch
import numpy as np
from pathlib import Path
from psi.utils import parse_args_to_tyro_config, seed_everything, move_to_device, batchify
from psi.config.config import LaunchConfig
from psi.utils import seed_everything

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime

ckpt_step = 40000
run_dir = Path(".runs/finetune/ffclear.g1soni.flow1000.cosine.lr1.0e-04.b128.gpus8.2608181732") # non-rtc 
# run_dir = Path(".runs/finetune/neck-realsense-mpnp.neckle.flow1000.cosine.lr1.0e-04.b128.gpus4.2607131521") # rtc 
g = os.environ.get("G", "test_rtc") # "train_rtc" # "test_rtc_inpainting" # "test_rtc"
alpha = float(os.environ.get("ALPHA", 0.9)) # test_rtc guidance strength

config_:LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
conf = (run_dir / "run_config.json").open("r").read()
launch_config = config_.model_validate_json(conf)
seed_everything(launch_config.seed or 42)

# from we.config.data import SimpleDataConfig
data_cfg = launch_config.data # type: ignore

# from we.config.model import Together_ModelConfig
model_cfg = launch_config.model # type: ignore

# Use GPU 0
DEVICE = "cuda:0"
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# from we.learn.pipelines.vla_pipeline import VLAPipeline
# pipeline = VLAPipeline.from_pretrained(run_dir, ckpt_step, launch_config).to(DEVICE)
# from we.learn.models.hfm import HumanFoundationTogetherModel
from psi.models.psi0 import Psi0Model
model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=DEVICE)
model = model.to(DEVICE)

#print(model mode train or eval)
# model.eval()

maxmin = data_cfg.transform.field
model.eval()

vlm_processor = model.vlm_processor
transform_kwargs=dict(
    vlm_processor=vlm_processor,
)
val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)
print(f"Validation dataset size: {len(val_dataset)}")

from PIL import Image
import numpy as np
np.set_printoptions(precision=4, suppress=True)

l2_xyz = []
num_eval=300

dataset = val_dataset
eps_idx = 0

# 随机采样
np.random.seed(42)  # 设置随机种子以确保可复现性
random_indices = np.random.choice(len(dataset), size=min(num_eval, len(dataset)), replace=False)

# start_frame_idx = 30190 
# end_frame_idx = 31844 
start_frame_idx = val_dataset.raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
end_frame_idx = val_dataset.raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()

# NOTE the first frames of an episode are static (prev_actions == gt over the whole
# masked region), so they cannot separate the guidance variants -- START skips ahead.
start_frame_idx = int(os.environ.get("START", start_frame_idx))

print(start_frame_idx, end_frame_idx)
# tensor(16301)

avg_action_errors_denormed_list = []
# for i in random_indices:
for i in range(start_frame_idx, end_frame_idx, 16):
    frame = val_dataset[i]
    images = frame["observations"] # List[PIL.Image.Image] # (0~255)
    gt_actions = frame["actions"]
    # images[0].save("inf.png")
    # exit(0)

    batch_images = [images] # List[List[PIL.Image.Image]] batch size == 1

    instruction = frame['instruction']
    batch_instructions = [instruction] # List[str]

    states = frame['states'] # (1, 15)
    batch_states = torch.from_numpy(states).unsqueeze(0).to(DEVICE) # (B, H, D)

    pred_actions = model.predict_action(
        observations=batch_images, 
        states=batch_states, 
        instructions=batch_instructions, 
        num_inference_steps=10, 
        traj2ds=None)

    ### rtc ###
    gt_actions = gt_actions[np.newaxis, :, :]

    prev_actions = np.concatenate([gt_actions[:, 6:, :], np.zeros((1, 6, gt_actions.shape[-1]))], axis=1) # FIXME
    prev_actions = torch.from_numpy(prev_actions).to(DEVICE)

    guidance_method = {
        "train_rtc": (model.predict_action_with_training_rtc_flow, {}),
        "test_rtc_inpainting": (model.predict_action_with_rtc_flow_naive_inpaint, {}),
        # guidance_alpha = fraction of the masked error the correction closes per step
        "test_rtc": (model.predict_action_with_rtc_flow, {"guidance_alpha": alpha}),
    }
    guidance_fn, guidance_kwargs = guidance_method[g]
    pred_actions_rtc = guidance_fn(
        observations=batch_images,
        states=batch_states,
        instructions=batch_instructions,
        num_inference_steps=10,
        traj2ds=None,
        prev_actions=prev_actions,
        inference_delay=6,
        max_delay=8,
        execution_horizon=16,
        **guidance_kwargs
    )
    

    denorm_gt_actions = maxmin.denormalize(gt_actions)[0]
    denorm_prev_actions = maxmin.denormalize(prev_actions).cpu().numpy()[0]
    # denorm_pred_actions_rtc = maxmin.denormalize(pred_actions_rtc.float()).cpu().numpy()[0]
    denorm_pred_actions_rtc = maxmin.denormalize(pred_actions_rtc.float()).cpu().numpy()[0]
    denorm_pred_actions = maxmin.denormalize(pred_actions.float()).cpu().numpy()[0]

    # 可视化三个动作数组的对比
    H, D = denorm_prev_actions.shape  # 应该是 (30, 36)
    
    # 计算子图的行列数
    n_cols = 6  # 每行6个子图
    n_rows = int(np.ceil(D / n_cols))  # 根据维度数计算需要多少行
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
    fig.suptitle(f'Actions Comparison - Iteration {i}', fontsize=16, y=0.995)
    
    # 展平axes数组以便索引
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # 为每个维度绘制子图
    for dim in range(D):
        ax = axes_flat[dim]
        
        # 横坐标是时间步 H
        time_steps = np.arange(H)
        
        # 绘制三条线
        ax.plot(time_steps, denorm_gt_actions[:, dim], 'k-', label='gt_actions', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps, denorm_prev_actions[:, dim], 'b-', label='prev_actions', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps, denorm_pred_actions_rtc[:, dim], 'r-', label='pred_actions_rtc', alpha=0.7, linewidth=1.5)
        ax.plot(time_steps, denorm_pred_actions[:, dim], 'g-', label='pred_actions', alpha=0.7, linewidth=1.5)
        
        # 设置标题和标签
        ax.set_title(f'Dim {dim}', fontsize=10)
        ax.set_xlabel('Time Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        ax.tick_params(labelsize=7)
    
    # 隐藏多余的子图
    for idx in range(D, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 使用时间戳命名文件
    vis_save_dir = Path(f".runs/visualize/{g}")
    vis_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"可视化图像将保存到: {vis_save_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = vis_save_dir / f"actions_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形释放内存
    print(f"已保存可视化图像到: {save_path}")
    ####
    gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(DEVICE) # (6, 7)
    denormalized_pred_actions = maxmin.denormalize(pred_actions)
    error = denormalized_pred_actions - gt_action # (B, 6, 7)
    error_l1 = error.detach().abs().cpu().numpy().reshape(-1, 80)

    # action L1 errors
    avg_action_errors_denormed = error_l1.mean(
        0
    )  # (7,) NOTE only if the error is L1 (linear)

    labels_denormed = [
        "val/denorm_err_l1_body",
        "val/denorm_err_l1_hand",
        "val/denorm_err_l1_neck",
    ]

    avg_lr_action_err_denormed = np.split(
        avg_action_errors_denormed, [64, 78], axis=-1
    )
    avg_action_errors_denormed_list.append(avg_action_errors_denormed)

    # log metrics
    for i in range(len(avg_lr_action_err_denormed)):
        print(f"denormed_err_l1_{labels_denormed[i]}: {avg_lr_action_err_denormed[i].mean()}")

    # break

avg_action_errors_denormed_list = np.stack(avg_action_errors_denormed_list, axis=0)
avg_action_errors_denormed = avg_action_errors_denormed_list.mean(axis=0)

labels_denormed = [
    "val/denorm_err_l1_body",
    "val/denorm_err_l1_hand",
    "val/denorm_err_l1_neck",
]

avg_action_errors_denormed_split = np.split(
    avg_action_errors_denormed, [64, 78], axis=-1
)

print("\n---------------------------\n")
for i in range(len(avg_action_errors_denormed_split)):
    print(f"denormed_err_l1_{labels_denormed[i]}: {avg_action_errors_denormed_split[i].mean()}")
