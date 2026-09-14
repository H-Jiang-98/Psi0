"""Dump merged normalization stats into an existing run_dir as norm_stats.json.

For checkpoints trained BEFORE train.py started saving norm_stats.json. Run this
once on a machine that still has access to the training data (dataset_paths); it
rebuilds the exact same dataset mixture used in training and writes the *merged*
per-embodiment statistics to <run_dir>/norm_stats.json. After that you can rsync
just the checkpoint folder and serve with psi_serve_psix_rtc_sonic.py without the
data.

Usage:
    python scripts/deploy/dump_norm_stats.py --run-dir /path/to/run [--split train]
"""

import json
from pathlib import Path

import tyro

from psi.config.config import LaunchConfig
from psi.utils import parse_args_to_tyro_config


def main(run_dir: str, split: str = "train") -> None:
    run_path = Path(run_dir)
    assert (run_path / "run_config.json").exists(), f"no run_config.json in {run_path}"

    config_: LaunchConfig = parse_args_to_tyro_config(run_path / "argv.txt")  # type: ignore
    launch_config = config_.model_validate_json((run_path / "run_config.json").read_text())

    # Build the dataset mixture exactly like training. merged_metadata is populated
    # in LeRobotMixtureDataset.update_metadata (element-wise min/max merge per tag).
    dataset = launch_config.data(split=split)
    raw = getattr(dataset, "raw_dataset", None)
    merged = getattr(raw, "merged_metadata", None)
    assert merged, (
        "raw_dataset has no merged_metadata — is this a MixedFieldTransform (GR00T) run?"
    )

    norm = {tag: md.model_dump(mode="json")["statistics"] for tag, md in merged.items()}
    out = run_path / "norm_stats.json"
    out.write_text(json.dumps(norm, indent=2))
    print(f"Wrote {out} for tags {list(norm)}")


if __name__ == "__main__":
    tyro.cli(main)
