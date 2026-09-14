"""Console-script wrapper around scripts/deploy/modelscope_dataset_dir.py.

Exposed as the `upload_dataset_ms` command via [project.scripts] so it can
be run from the uv environment, alongside upload_ckpt_ms; the destination is a
ModelScope dataset repo rather than a model one, and the directory is zipped
before it is pushed:

    upload_dataset_ms [ms-repo-id] [path-prefix] [dataset-dir] [local-base]
"""

import os
import sys
from pathlib import Path

# repo_root/src/psi/deploy/upload_dataset_ms.py -> repo_root
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "deploy" / "modelscope_dataset_dir.py"


def main() -> None:
    if not _SCRIPT.is_file():
        sys.exit(f"upload script not found: {_SCRIPT}")
    # Replace the current process so signals (Ctrl-C) and the exit code pass through.
    os.execvp(sys.executable, [sys.executable, str(_SCRIPT), *sys.argv[1:]])


if __name__ == "__main__":
    main()
