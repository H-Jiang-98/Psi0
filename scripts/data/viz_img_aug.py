"""Render a grid of real training frames through the recipe's image pipeline.

Rows: 4 frames from different tasks (2 trash-can). Columns: clean (resize+center_crop only),
3x view-aug only, 3x colour-jitter only, 3x full train pipeline (view aug + colour jitter).
"""
import os, sys, json
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.chdir("/mnt/beegfs/scratch/songlinwei/psi0")
np.random.seed(3); torch.manual_seed(3)
from torchvision.transforms import v2
from psi.config.transform import Psi0ModelTransform
from psi.config.augmentation import RandomViewPerturb
from psi.data.lerobot.compat import LeRobotDataset
from psi.utils import pt_to_pil

OUT = sys.argv[1] if len(sys.argv) > 1 else "assets/media/aug_grid_state0.2.png"
MIN_SCALE = float(os.environ.get("VIEW_AUG_MIN_SCALE", "0.85"))
PACK = ".data/g1_sonic_lerobot_0810_merged_train"
ds = LeRobotDataset("g1_sonic_lerobot_0810_merged_train", root=PACK, video_backend="pyav")
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{PACK}/meta/tasks.jsonl")}
ti = np.asarray(ds.hf_dataset.with_format("numpy")["task_index"]).reshape(-1)
ei = np.asarray(ds.hf_dataset.with_format("numpy")["episode_index"]).reshape(-1)
def pick(task_id, frac):
    idx = np.nonzero(ti == task_id)[0]
    return int(idx[int(len(idx) * frac)])
# trash can x2 (mid-episode, where the foot is near the bin), sweep, shoes
frames = [pick(39, 0.55), pick(41, 0.6), pick(13, 0.5), pick(6, 0.45)]

mt = Psi0ModelTransform(resize={"size": (270, 480)}, center_crop={"size": (270, 480)}, img_aug=True,
                        view_aug=True, view_aug_min_scale=MIN_SCALE, view_aug_prob=1.0)
base = v2.Compose([mt.resize(), mt.center_crop()])
view = RandomViewPerturb(size=(270, 480), min_scale=MIN_SCALE, prob=1.0)()
jit = mt.color_jitter()
cols = [("clean", lambda im: im)] + [(f"view #{i}", view) for i in range(1, 4)] \
     + [(f"jitter #{i}", jit) for i in range(1, 4)] + [(f"train #{i}", lambda im: jit(view(im))) for i in range(1, 4)]

W, H, PAD, TOP, LEFT = 240, 135, 6, 22, 0
grid = Image.new("RGB", (LEFT + len(cols) * (W + PAD), 40 + len(frames) * (H + PAD + TOP)), "white")
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception:
    font = ImageFont.load_default()
for c, (name, _) in enumerate(cols):
    draw.text((LEFT + c * (W + PAD) + 4, 4), name, fill="black", font=font)
for r, fi in enumerate(frames):
    raw = ds[fi]
    im = base(pt_to_pil(raw["observation.images.head"], normalized=False))
    y0 = 40 + r * (H + PAD + TOP)
    draw.text((4, y0), f"frame {fi} ep{int(ei[fi])}  {tasks[int(ti[fi])][:110]}", fill="black", font=font)
    for c, (name, fn) in enumerate(cols):
        out = fn(im)
        assert out.size == (480, 270), out.size
        grid.paste(out.resize((W, H), Image.BILINEAR), (LEFT + c * (W + PAD), y0 + TOP))
grid.save(OUT)
print(f"saved {OUT} {grid.size}; frames {frames}; view min_scale={MIN_SCALE}; jitter="
      f"b={mt.color_jitter.brightness} c={mt.color_jitter.contrast} s={mt.color_jitter.saturation} h={mt.color_jitter.hue}")
