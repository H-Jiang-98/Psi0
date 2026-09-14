#!/usr/bin/env python3
"""Zip one dataset directory and push it to a ModelScope dataset repo.

The dataset counterpart of scripts/deploy/modelscope_run_dir.py: same token
handling, same retry loop, same DRY_RUN switch. Where that script pushes a run
directory tree to a *model* repo, this one archives

    LOCAL_BASE/DATASET_DIR  ->  MS_REPO_ID/PATH_PREFIX/<dataset name>.zip

as a single zip, because the datasets are thousands of small parquet and mp4
files and the hub is far happier with one large object than with many small
ones. The archive keeps the dataset directory as its root, so unzipping it
restores the original layout.

    upload_dataset_ms [ms-repo-id] [path-prefix] [dataset-dir] [local-base]

Set DRY_RUN=1 to print what would be uploaded without zipping or transferring,
MAX_RETRIES to change the retry budget, KEEP_ZIP=1 to leave the staged archive
behind, and ZIP_DIR to stage it somewhere other than beside the dataset.

The API token is read from the environment (MODELSCOPE_API_TOKEN, falling back
to MS_UPLOAD_TOKEN), loaded from the project .env if present. It is never
stored in this file.
"""

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# repo_root/scripts/deploy/modelscope_dataset_dir.py -> repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPO_ID = "uscpsix/psix-data"
DEFAULT_PATH_PREFIX = "finetune"
DEFAULT_DATASET_DIR = ".data/g1_sonic_lerobot_0810_merged_val"
DEFAULT_ENDPOINT = "https://www.modelscope.ai"

TOKEN_ENV_VAR = "MODELSCOPE_API_TOKEN"
FALLBACK_TOKEN_ENV_VAR = "MS_UPLOAD_TOKEN"

# Version-control and editor droppings; everything else in the tree is data.
EXCLUDED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def read_env_file(env_path):
    """Minimal .env reader, matching scripts/deploy/modelscope_run_dir.py."""
    env_values = {}
    if not env_path.exists():
        return env_values
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env_values[key] = value
    return env_values


def load_env():
    env_values = read_env_file(PROJECT_ROOT / ".env")
    for key, value in env_values.items():
        os.environ.setdefault(key, value)
    return env_values


def resolve_token():
    token = os.environ.get(TOKEN_ENV_VAR) or os.environ.get(FALLBACK_TOKEN_ENV_VAR)
    if not token:
        sys.exit(
            f"Neither {TOKEN_ENV_VAR} nor {FALLBACK_TOKEN_ENV_VAR} is set. "
            f"Add one of them to {PROJECT_ROOT / '.env'} before uploading."
        )
    return token


def build_zip(source, zip_path):
    """Archive `source` with its own name as the archive root.

    Stored, not deflated: the payload is parquet and mp4, already compressed,
    so deflating costs minutes of CPU to save a fraction of a percent.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in sorted(source.rglob("*")):
            if any(part in EXCLUDED_DIRS for part in path.relative_to(source).parts):
                continue
            if path.is_file() or path.is_dir():
                zf.write(path, Path(source.name) / path.relative_to(source))
    return zip_path


def main():
    argv = sys.argv[1:]

    def arg(index, default):
        # Matches bash ${N:-default}: an empty argument falls back to the default.
        return argv[index] if len(argv) > index and argv[index] else default

    repo_id = arg(0, DEFAULT_REPO_ID)
    path_prefix = arg(1, DEFAULT_PATH_PREFIX).strip("/")
    dataset_dir = arg(2, DEFAULT_DATASET_DIR)
    local_base = arg(3, os.getcwd())

    load_env()
    token = resolve_token()
    endpoint = os.environ.get("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")

    source = (Path(local_base) / dataset_dir).resolve()
    if not source.is_dir():
        sys.exit(f"dataset directory not found: {source}")

    name = f"{source.name}.zip"
    path_in_repo = f"{path_prefix}/{name}" if path_prefix else name
    # Stage beside the dataset by default: /tmp is often far too small for these.
    zip_dir = Path(os.environ.get("ZIP_DIR") or source.parent / ".ms_zip_staging")
    zip_path = zip_dir / name

    print(f"Zipping  {source}")
    print(f"      to {endpoint}/datasets/{repo_id} :: {path_in_repo}")

    dry_run = os.environ.get("DRY_RUN", "") not in ("", "0")
    if dry_run:
        print(f"[dry-run] would stage {zip_path}")
        print(f"[dry-run]     excluding {sorted(EXCLUDED_DIRS)}")
        print(f"[dry-run]    then upload -> {repo_id}:{path_in_repo}")
        print("dry run complete; nothing was uploaded.")
        return 0

    from modelscope.hub.api import HubApi

    api = HubApi(endpoint=endpoint, token=token)

    build_zip(source, zip_path)
    print(f"staged {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

    # Retry on failure (e.g. dropped connection) until the upload succeeds or we
    # run out of attempts, matching MAX_RETRIES in modelscope_run_dir.py.
    max_retries = int(os.environ.get("MAX_RETRIES", "50"))
    keep_zip = os.environ.get("KEEP_ZIP", "") not in ("", "0")

    try:
        attempt = 1
        while True:
            try:
                api.upload_file(
                    repo_id=repo_id,
                    path_or_fileobj=str(zip_path),
                    path_in_repo=path_in_repo,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Upload {name}",
                )
                print(f"upload of {name} completed successfully.")
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                if attempt >= max_retries:
                    print(
                        f"upload of {name} failed after {attempt} attempts; giving up: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"upload of {name} failed (attempt {attempt}/{max_retries}): {exc}; retrying in 5s...",
                    file=sys.stderr,
                )
                attempt += 1
                time.sleep(5)
    finally:
        if keep_zip:
            print(f"staged archive kept at {zip_path}")
        else:
            zip_path.unlink(missing_ok=True)
            # Only our own staging dir gets removed, and only once it is empty.
            if zip_dir.name == ".ms_zip_staging" and not any(zip_dir.iterdir()):
                shutil.rmtree(zip_dir, ignore_errors=True)

    print("upload completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
