#!/usr/bin/env python3
"""Pull one checkpoint of a run directory from ModelScope.

The exact inverse of scripts/deploy/modelscope_run_dir.py, and the ModelScope
counterpart of scripts/deploy/rsync_run_dir.sh: same positional arguments, same
checkpoint selection, same exclusion set, same retry loop. Where the upload
script pushes

    LOCAL_BASE/RUN_DIR  ->  MS_REPO_ID/PATH_PREFIX/SUBFOLDER/<run name>

this pulls

    MS_REPO_ID/PATH_PREFIX/SUBFOLDER/<run name>  ->  LOCAL_BASE/RUN_DIR

so a run uploaded with upload_ckpt_ms comes back to the same relative path.

    download_ckpt_ms [ms-repo-id] [path-prefix] [run-dir] [ckpt-step] [local-base]

Set DRY_RUN=1 to print what would be downloaded without transferring anything,
and MAX_RETRIES to change the retry budget (default 50, as in the rsync script).

The API token is read from the environment (MODELSCOPE_API_TOKEN, falling back
to MS_UPLOAD_TOKEN), loaded from the project .env if present. It is never
stored in this file.
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# repo_root/scripts/deploy/modelscope_download_run_dir.py -> repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPO_ID = "uscpsix/psix-model"
DEFAULT_PATH_PREFIX = "runs"
DEFAULT_RUN_DIR = ".runs/psix_finetune/mem.rss26r.flow1000.cosine.lr5.0e-05.b128.gpus8.2606090813"
DEFAULT_CKPT_STEP = "40000"
DEFAULT_ENDPOINT = "https://www.modelscope.ai"

TOKEN_ENV_VAR = "MODELSCOPE_API_TOKEN"
FALLBACK_TOKEN_ENV_VAR = "MS_UPLOAD_TOKEN"

# The same states the rsync script refuses to move: large and rebuildable.
IGNORE_PATTERNS = [
    "*random_states_*.pkl",
    "*optimizer.bin",
    "*scheduler.bin",
]


def read_env_file(env_path):
    """Minimal .env reader, matching scripts/deploy/upload_ckpt.py."""
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
            f"Add one of them to {PROJECT_ROOT / '.env'} before downloading."
        )
    return token


def list_repo_paths(api, repo_id):
    """Every file path in the repo, or None if the listing is unavailable."""
    try:
        files = api.get_model_files(model_id=repo_id, recursive=True, page_size=1000)
    except Exception:  # noqa: BLE001 - the listing is only used for diagnostics
        return None
    return {entry["Path"] for entry in files if entry.get("Path")}


def run_dirs_in(paths):
    """Remote directories that look like run dirs, i.e. contain checkpoints/."""
    return sorted(
        {
            path[: path.index("/checkpoints/")]
            for path in paths
            if "/checkpoints/" in path
        }
    )


def check_remote_layout(api, repo_id, remote_dir, ckpt_step):
    """Fail fast on a remote path that does not exist, with the near misses.

    snapshot_download reports success when a pattern matches nothing, so an
    argument slipped by one position (path-prefix vs run-dir) otherwise looks
    like two clean downloads followed by an empty staging dir.
    """
    paths = list_repo_paths(api, repo_id)
    if paths is None:
        return
    prefix = f"{remote_dir}/"
    if not any(path.startswith(prefix) for path in paths):
        candidates = run_dirs_in(paths)
        hint = "\n".join(f"  {c}" for c in candidates[:20]) or "  (none found)"
        sys.exit(
            f"{repo_id}:{remote_dir} does not exist.\n"
            f"Arguments are [ms-repo-id] [path-prefix] [run-dir] [ckpt-step] [local-base];\n"
            f"the remote path is <path-prefix>/<run-dir>, so a whole run path passed as\n"
            f"the prefix leaves the run-dir slot holding what should be the step.\n"
            f"Run directories in {repo_id}:\n{hint}"
        )
    ckpt_prefix = f"{remote_dir}/checkpoints/ckpt_{ckpt_step}/"
    if not any(path.startswith(ckpt_prefix) for path in paths):
        steps = sorted(
            {
                path[len(f"{remote_dir}/checkpoints/"):].split("/")[0]
                for path in paths
                if path.startswith(f"{remote_dir}/checkpoints/")
            }
        )
        sys.exit(
            f"checkpoint ckpt_{ckpt_step} not found in {repo_id}:{remote_dir}/checkpoints.\n"
            f"Available: {', '.join(steps) or '(none)'}"
        )


def merge_into(src, dst):
    """Move everything under src into dst, creating dirs and overwriting files."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            merge_into(item, target)
            item.rmdir()
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(item), str(target))


def main():
    argv = sys.argv[1:]

    def arg(index, default):
        # Matches bash ${N:-default}: an empty argument falls back to the default.
        return argv[index] if len(argv) > index and argv[index] else default

    repo_id = arg(0, DEFAULT_REPO_ID)
    path_prefix = arg(1, DEFAULT_PATH_PREFIX).strip("/")
    run_dir = arg(2, DEFAULT_RUN_DIR)
    ckpt_step = arg(3, DEFAULT_CKPT_STEP)
    local_base = arg(4, os.getcwd())

    load_env()
    token = resolve_token()
    endpoint = os.environ.get("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")

    # Mirror the upload layout: strip the leading .runs/ so the remote path is
    # <prefix>/<subfolder>/<run name>, e.g. runs/psix_finetune/mem.rss26r...
    rel = run_dir.strip("/")
    if rel.startswith(".runs/"):
        rel = rel[len(".runs/"):]
    remote_dir = f"{path_prefix}/{rel}" if path_prefix else rel

    dest = (Path(local_base) / run_dir).resolve()

    print(f"Downloading {endpoint}/models/{repo_id} :: {remote_dir}")
    print(f"        to  {dest}")
    print(f"Checkpoint: ckpt_{ckpt_step} (optimizer/scheduler/random states excluded)")

    # Two fetches instead of one, so "everything except the checkpoints, plus
    # exactly one checkpoint" is expressible: a single allow/ignore pair cannot
    # say "all checkpoints except this one".
    #
    # These are fnmatch patterns, not shell globs: `*` crosses `/`, so
    # `<dir>/*` means "everything underneath, recursively", and excluding the
    # other checkpoints has to be done with ignore_patterns. ignore wins over
    # allow (modelscope_hub/_download.py:728).
    jobs = [
        (
            [f"{remote_dir}/*"],
            [f"{remote_dir}/checkpoints/*", "*wandb/*"],
            "run metadata",
        ),
        (
            [f"{remote_dir}/checkpoints/ckpt_{ckpt_step}/*"],
            list(IGNORE_PATTERNS),
            f"ckpt_{ckpt_step}",
        ),
    ]

    max_retries = int(os.environ.get("MAX_RETRIES", "50"))
    dry_run = os.environ.get("DRY_RUN", "") not in ("", "0")

    if dry_run:
        for allow_patterns, ignore_patterns, label in jobs:
            print(f"[dry-run] would download {label}")
            print(f"[dry-run]        allowing {allow_patterns}")
            print(f"[dry-run]        ignoring {ignore_patterns}")
        print("dry run complete; nothing was downloaded.")
        return 0

    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import snapshot_download

    # Check the repo up front: a bad repo id (e.g. an ssh host name meant for
    # download_ckpt) 404s on every snapshot_download, and the retry loop would
    # otherwise spend 50 attempts on an error that will never become transient.
    api = HubApi(endpoint=endpoint, token=token)
    try:
        api.get_model(repo_id)
    except Exception as exc:  # noqa: BLE001 - surface auth/network problems up front
        sys.exit(
            f"Cannot access {endpoint}/models/{repo_id}: {exc}\n"
            f"Expected a ModelScope repo id like {DEFAULT_REPO_ID}; to pull from an\n"
            f"ssh host instead, use download_ckpt [remote-host] [remote-project-dir] ...\n"
            f"Otherwise check that {TOKEN_ENV_VAR} is valid and can read the repo."
        )

    check_remote_layout(api, repo_id, remote_dir, ckpt_step)

    # Stage beside the destination so the final move is a rename, not a copy.
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".download_ckpt_ms_", dir=str(dest.parent)))

    try:
        for allow_patterns, ignore_patterns, label in jobs:
            attempt = 1
            while True:
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="model",
                        local_dir=str(staging),
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                        token=token,
                        endpoint=endpoint,
                    )
                    print(f"download of {label} completed successfully.")
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - retry on anything transient
                    if attempt >= max_retries:
                        print(
                            f"download of {label} failed after {attempt} attempts; giving up: {exc}",
                            file=sys.stderr,
                        )
                        return 1
                    print(
                        f"download of {label} failed (attempt {attempt}/{max_retries}): {exc}; retrying in 5s...",
                        file=sys.stderr,
                    )
                    attempt += 1
                    time.sleep(5)

        fetched = staging / remote_dir
        if not fetched.is_dir():
            print(
                f"nothing was fetched for {remote_dir}; is the run present in {repo_id}?",
                file=sys.stderr,
            )
            return 1

        # snapshot_download reports success when a pattern matches nothing, so a
        # typo'd step would otherwise look like a clean run that fetched no
        # checkpoint. Mirror the upload script, which exits on a missing ckpt.
        if not (fetched / "checkpoints" / f"ckpt_{ckpt_step}").is_dir():
            print(
                f"checkpoint ckpt_{ckpt_step} not found in {repo_id}:{remote_dir}/checkpoints",
                file=sys.stderr,
            )
            return 1

        merge_into(fetched, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print("download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
