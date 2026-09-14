"""Split a Psi0Model accelerate checkpoint (ckpt_N/) into the two things a finetune
run warm-starts from:

  <out>/model.safetensors + HF config/processor files -> --model.model-name-or-path
  <out>/action_header.safetensors                     -> --model.pretrained-action-header-path

FinetuneTrainer loads the VLM with `Qwen3VLForConditionalGeneration.from_pretrained()`
and the action header with `load_file(f"{path}/action_header.safetensors")`, so a run
checkpoint -- one flat state dict of `vlm_model.*` + `action_header.*` and nothing else
-- cannot be handed to either flag directly. This writes the split form.

The HF config/processor files `from_pretrained` needs are NOT in a run checkpoint, so
they are copied from the VLM directory the run itself started from: read out of the
run's argv.txt by default, or given with --hf-from.

This is a WEIGHTS-ONLY warm start, unlike --train.resume_from_checkpoint, which restores
the optimizer/scheduler as well and resumes the step counter where the source run ended.

Usage:
  python scripts/export_psi0_ckpt.py <ckpt_dir> <out_dir> [--hf-from DIR]
                                     [--dtype fp32|bf16] [--head-only]

  python scripts/export_psi0_ckpt.py \
      .runs/posttrain/<run>/checkpoints/ckpt_100000 \
      .runs/posttrain/<run>/pretrained/ckpt_100000
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import dotenv
import torch
from safetensors import safe_open
from safetensors.torch import save_file

dotenv.load_dotenv()  # PSI_HOME; does not override anything already exported

VLM_PREFIX = "vlm_model."
HEAD_PREFIX = "action_header."

# What Qwen3VLForConditionalGeneration.from_pretrained + AutoProcessor need next to the
# weights. config.json is mandatory; the rest are copied when present.
HF_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja",
]


def resolve_psi_home(path: str) -> Path:
    """Mirror Psi0ModelConfig._resolve_psi_home_paths: relative paths sit under $PSI_HOME.

    The launcher does not export PSI_HOME (scripts/train.py only picks it up inside
    load_dotenv), so read .env here too -- without it a checkpoint recorded as
    `cache/checkpoints/psi0/...` resolves under the default /psi and is not found.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    home = Path(os.environ.get("PSI_HOME", "/psi")) / path
    return home if home.exists() else candidate


def vlm_dir_from_run(ckpt_dir: Path) -> Path | None:
    """The --model.model_name_or_path the source run was launched with."""
    argv = ckpt_dir.parent.parent / "argv.txt"
    if not argv.exists():
        return None
    for token in argv.read_text().split():
        if token.startswith("--model.model_name_or_path="):
            return resolve_psi_home(token.split("=", 1)[1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir", type=Path, help="a run's checkpoints/ckpt_N directory")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--hf-from", type=Path, default=None,
                    help="VLM dir to copy config/processor files from "
                         "(default: the model_name_or_path in the run's argv.txt)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="VLM weight dtype; the header is always kept as stored")
    ap.add_argument("--head-only", action="store_true",
                    help="write only action_header.safetensors")
    args = ap.parse_args()

    weights = args.ckpt_dir / "model.safetensors"
    if not weights.exists():
        raise SystemExit(f"{weights} not found")

    hf_from = args.hf_from or vlm_dir_from_run(args.ckpt_dir)
    if not args.head_only:
        if hf_from is None:
            raise SystemExit(
                f"no --hf-from given and no --model.model_name_or_path in "
                f"{args.ckpt_dir.parent.parent / 'argv.txt'}"
            )
        if not (hf_from / "config.json").exists():
            raise SystemExit(f"{hf_from}/config.json not found (wrong --hf-from?)")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    # Assemble into a .partial directory and rename at the end: a launcher that waits
    # for these files must never see a half-written one.
    staging = args.out_dir.with_name(args.out_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    vlm, head, other = {}, {}, []
    with safe_open(weights, "pt") as f:
        for key in f.keys():
            if key.startswith(VLM_PREFIX):
                if not args.head_only:
                    vlm[key[len(VLM_PREFIX):]] = f.get_tensor(key).to(dtype).contiguous()
            elif key.startswith(HEAD_PREFIX):
                head[key[len(HEAD_PREFIX):]] = f.get_tensor(key).contiguous()
            else:
                other.append(key)
    if other:
        raise SystemExit(f"unexpected keys in {weights}: {other[:5]}")
    if not head:
        raise SystemExit(f"no {HEAD_PREFIX}* keys in {weights}")

    save_file(head, staging / "action_header.safetensors", metadata={"format": "pt"})
    copied = []
    if not args.head_only:
        save_file(vlm, staging / "model.safetensors", metadata={"format": "pt"})
        assert hf_from is not None
        for name in HF_FILES:
            # Prefer a copy the checkpoint carries itself; fall back to the source VLM.
            src = args.ckpt_dir / name
            if not src.exists():
                src = hf_from / name
            if src.exists():
                shutil.copy2(src, staging / name)
                copied.append(name)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(args.out_dir)

    print(f"vlm keys: {len(vlm)}  action_header keys: {len(head)}")
    if copied:
        print(f"hf files from {hf_from}: {', '.join(copied)}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
