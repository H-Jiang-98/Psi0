import argparse
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.constants import HF_TOKEN_PATH
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from urllib3.exceptions import MaxRetryError, NewConnectionError

try:
	from dotenv import load_dotenv
except ImportError:
	load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / ".runs"
FALLBACK_REPO_ID = "USC-PSI-Lab/psi-model"
FALLBACK_UPLOAD_ENDPOINT = "https://huggingface.co"
REMOTE_PREFIX = "runs"
EXCLUDED_STATE_PATTERNS = ("optimizer", "random_states")
UPLOAD_TOKEN_ENV_VAR = "HF_UPLOAD_TOKEN"
FALLBACK_TOKEN_ENV_VAR = "HF_TOKEN"
EXPECTED_ENV_VARS = (UPLOAD_TOKEN_ENV_VAR, FALLBACK_TOKEN_ENV_VAR, "HF_REPO_ID")


def read_env_file(env_path: Path) -> dict[str, str]:
	env_values: dict[str, str] = {}
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
		if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
			value = value[1:-1]
		env_values[key] = value
	return env_values


def warn_missing_env_vars(env_values: dict[str, str]) -> None:
	missing_vars = [name for name in EXPECTED_ENV_VARS if name not in env_values]
	if missing_vars:
		warnings.warn(
			"Missing uploader variables in .env: " + ", ".join(missing_vars),
			stacklevel=2,
		)


def load_env() -> dict[str, str]:
	env_path = PROJECT_ROOT / ".env"
	env_values = read_env_file(env_path)
	warn_missing_env_vars(env_values)
	if load_dotenv is not None and env_path.exists():
		load_dotenv(env_path)
	else:
		for key, value in env_values.items():
			os.environ.setdefault(key, value)
	return env_values


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Upload a training run from .runs to Hugging Face.")
	parser.add_argument("run_path", type=Path, help="Path to a run directory under .runs.")
	parser.add_argument(
		"--include-all-ckpts",
		action="store_true",
		help="Upload all checkpoint directories instead of only the latest checkpoint.",
	)
	parser.add_argument(
		"--include-wandb",
		action="store_true",
		help="Include the wandb directory in the upload.",
	)
	parser.add_argument(
		"--remote-prefix",
		default=REMOTE_PREFIX,
		help=f"Prefix the run's path under .runs is placed below in the repo (default: {REMOTE_PREFIX}).",
	)
	parser.add_argument(
		"--remote-dir",
		default=None,
		help="Full destination path inside the repo, overriding --remote-prefix and the derived run path.",
	)
	return parser.parse_args()


def resolve_run_path(run_path: Path) -> Path:
	resolved = run_path.expanduser().resolve()
	if not resolved.exists():
		raise FileNotFoundError(f"Run path does not exist: {resolved}")
	if not resolved.is_dir():
		raise NotADirectoryError(f"Run path is not a directory: {resolved}")
	try:
		resolved.relative_to(RUNS_ROOT)
	except ValueError as exc:
		raise ValueError(f"Run path must be inside {RUNS_ROOT}") from exc
	return resolved


def derive_remote_dir(run_path: Path, remote_prefix: str = REMOTE_PREFIX) -> str:
	rel_path = run_path.relative_to(RUNS_ROOT).as_posix()
	prefix = remote_prefix.strip("/")
	return f"{prefix}/{rel_path}" if prefix else rel_path


def get_upload_endpoint(env_values: dict[str, str]) -> str:
	return (
		env_values.get("HF_UPLOAD_ENDPOINT")
		or os.environ.get("HF_UPLOAD_ENDPOINT")
		or FALLBACK_UPLOAD_ENDPOINT
	).rstrip("/")


def raise_for_network_error(exc: BaseException, endpoint: str) -> None:
	message = str(exc)
	status = getattr(getattr(exc, "response", None), "status_code", None)
	if status in (401, 403):
		# The endpoint answered, so this is a credentials problem, not a routing one.
		raise PermissionError(
			f"{endpoint} rejected the upload token ({status}). "
			f"Set a token with write access to the repo in {UPLOAD_TOKEN_ENV_VAR} (or {FALLBACK_TOKEN_ENV_VAR}) "
			"in .env, or run `hf auth login` with a write token."
		) from exc
	if isinstance(exc, (RequestsConnectionError, RequestException, MaxRetryError, NewConnectionError)) or (
		"Network is unreachable" in message or "Failed to establish a new connection" in message
	):
		raise ConnectionError(
			f"Cannot reach upload endpoint {endpoint}. This machine does not currently have outbound network access to the Hugging Face write endpoint. "
			"Uploading from this node will fail until network access is available. "
			"If this cluster only allows mirror reads, run the upload from a machine that can reach https://huggingface.co."
		) from exc
	if isinstance(exc, OSError) and getattr(exc, "errno", None) == 101:
		raise ConnectionError(
			f"Cannot reach upload endpoint {endpoint}: network is unreachable from this machine."
		) from exc


def validate_upload_access(api: HfApi, repo_id: str, endpoint: str) -> None:
	try:
		api.repo_info(repo_id=repo_id, repo_type="model")
	except (RequestsConnectionError, RequestException, MaxRetryError, NewConnectionError, OSError) as exc:
		raise_for_network_error(exc, endpoint)
	except RepositoryNotFoundError as exc:
		message = str(exc)
		if "Invalid username or password" in message or "401" in message:
			raise PermissionError(
				f"Authentication failed for upload endpoint {endpoint}. "
				f"The repo {repo_id} may exist, but the upload token is not accepted there. "
				f"If you were using a mirror endpoint, set HF_UPLOAD_ENDPOINT={FALLBACK_UPLOAD_ENDPOINT} for uploads."
			) from exc
		raise FileNotFoundError(
			f"Could not access model repo {repo_id} on upload endpoint {endpoint}. "
			"Verify the repo id and that the token has permission to the model repo."
		) from exc
	except HfHubHTTPError as exc:
		raise RuntimeError(
			f"Failed to validate repo access for {repo_id} on {endpoint}: {exc}"
		) from exc


def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
	name = path.name
	if name.startswith("ckpt_"):
		suffix = name.removeprefix("ckpt_")
		if suffix.isdigit():
			return (1, int(suffix), name)
	return (0, path.stat().st_mtime, name)


def find_latest_checkpoint_dir(checkpoints_dir: Path) -> Path:
	candidates = [path for path in checkpoints_dir.iterdir() if path.is_dir()]
	if not candidates:
		raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
	return max(candidates, key=checkpoint_sort_key)


def should_skip_checkpoint_file(path: Path) -> bool:
	return any(path.name.startswith(prefix) for prefix in EXCLUDED_STATE_PATTERNS)


def copytree_filtered(src: Path, dst: Path, skip_wandb: bool) -> None:
	dst.mkdir(parents=True, exist_ok=True)
	for entry in src.iterdir():
		if skip_wandb and entry.name == "wandb":
			continue
		target = dst / entry.name
		if entry.is_dir():
			shutil.copytree(entry, target, dirs_exist_ok=True)
		else:
			shutil.copy2(entry, target)


def stage_latest_checkpoint_upload(run_path: Path, exclude_wandb: bool) -> Path:
	staging_dir = Path(tempfile.mkdtemp(prefix="upload_ckpt_"))
	for entry in run_path.iterdir():
		if entry.name == "checkpoints":
			continue
		if exclude_wandb and entry.name == "wandb":
			continue
		target = staging_dir / entry.name
		if entry.is_dir():
			shutil.copytree(entry, target, dirs_exist_ok=True)
		else:
			shutil.copy2(entry, target)

	checkpoints_dir = run_path / "checkpoints"
	if not checkpoints_dir.exists():
		raise FileNotFoundError(f"Run does not contain checkpoints/: {run_path}")

	latest_checkpoint_dir = find_latest_checkpoint_dir(checkpoints_dir)
	staged_checkpoint_dir = staging_dir / "checkpoints" / latest_checkpoint_dir.name
	staged_checkpoint_dir.mkdir(parents=True, exist_ok=True)
	for entry in latest_checkpoint_dir.iterdir():
		if entry.is_file() and should_skip_checkpoint_file(entry):
			continue
		target = staged_checkpoint_dir / entry.name
		if entry.is_dir():
			shutil.copytree(entry, target, dirs_exist_ok=True)
		else:
			shutil.copy2(entry, target)
	return staging_dir


def read_stored_token() -> str | None:
	"""Token written by `hf auth login`, read from disk rather than via get_token().

	get_token() prefers HF_TOKEN, which is exactly the value we may be falling
	back from, so it cannot serve as an independent candidate here.
	"""
	try:
		token = Path(HF_TOKEN_PATH).read_text().strip()
	except OSError:
		return None
	return token or None


def is_auth_rejection(exc: BaseException) -> bool:
	status = getattr(getattr(exc, "response", None), "status_code", None)
	return status in (401, 403) or "Invalid user token" in str(exc)


def resolve_token(env_values: dict[str, str], endpoint: str) -> tuple[str, str]:
	"""Return the first token that authenticates, with the source it came from.

	A stale HF_TOKEN in .env still reads public repos but fails at commit time, so
	each candidate is checked with whoami before it is used, and the token from a
	local `hf auth login` is kept as the last fallback.
	"""
	candidates = [
		(UPLOAD_TOKEN_ENV_VAR, env_values.get(UPLOAD_TOKEN_ENV_VAR) or os.environ.get(UPLOAD_TOKEN_ENV_VAR)),
		(FALLBACK_TOKEN_ENV_VAR, env_values.get(FALLBACK_TOKEN_ENV_VAR) or os.environ.get(FALLBACK_TOKEN_ENV_VAR)),
		("hf auth login", read_stored_token()),
	]
	candidates = [(source, token) for source, token in candidates if token]
	if not candidates:
		raise EnvironmentError(
			f"Neither {UPLOAD_TOKEN_ENV_VAR} nor {FALLBACK_TOKEN_ENV_VAR} is set and no token is stored by "
			"`hf auth login`. Add one of them to .env before uploading."
		)

	probe = HfApi(endpoint=endpoint)
	rejected = []
	for source, token in candidates:
		try:
			role = probe.whoami(token=token).get("auth", {}).get("accessToken", {}).get("role")
		except (RequestsConnectionError, RequestException, MaxRetryError, NewConnectionError, OSError) as exc:
			if is_auth_rejection(exc):
				rejected.append(f"{source} (rejected: invalid token)")
				continue
			raise_for_network_error(exc, endpoint)
			raise
		if role == "write" or role is None:
			# role is None for fine-grained tokens, whose scopes only show up on use.
			return token, source
		rejected.append(f"{source} (role: {role or 'unknown'})")
	raise PermissionError(
		"No token with write access was found. Rejected: " + ", ".join(rejected) + ". "
		f"Add a write token as {UPLOAD_TOKEN_ENV_VAR} in .env, or run `hf auth login`."
	)


def upload_run(
	run_path: Path,
	include_all_ckpts: bool,
	include_wandb: bool,
	remote_prefix: str = REMOTE_PREFIX,
	remote_dir: str | None = None,
) -> None:
	env_values = load_env()

	repo_id = os.environ.get("HF_REPO_ID", FALLBACK_REPO_ID)
	remote_dir = remote_dir.strip("/") if remote_dir else derive_remote_dir(run_path, remote_prefix)
	upload_endpoint = get_upload_endpoint(env_values)
	hf_token, token_source = resolve_token(env_values, upload_endpoint)
	api = HfApi(endpoint=upload_endpoint, token=hf_token)
	validate_upload_access(api, repo_id, upload_endpoint)

	staging_dir: Path | None = None
	folder_to_upload = run_path
	ignore_patterns: list[str] | None = None if include_wandb else ["wandb/**", "wandb"]
	if not include_all_ckpts:
		staging_dir = stage_latest_checkpoint_upload(run_path, exclude_wandb=not include_wandb)
		folder_to_upload = staging_dir
		ignore_patterns = None

	print(f"Uploading {folder_to_upload} -> {repo_id}/{remote_dir}")
	print(f"Upload endpoint: {upload_endpoint}")
	if not include_all_ckpts:
		print("Mode: latest checkpoint only (optimizer/random states excluded)")
	else:
		print("Mode: all checkpoints included")
	if include_wandb:
		print("Mode: wandb included")
	else:
		print("Mode: wandb excluded")
	print(f"Token source: {token_source}")

	try:
		api.upload_folder(
			repo_id=repo_id,
			repo_type="model",
			folder_path=str(folder_to_upload),
			path_in_repo=remote_dir,
			ignore_patterns=ignore_patterns,
			commit_message=f"Upload run {run_path.name}",
		)
	except (RequestsConnectionError, RequestException, MaxRetryError, NewConnectionError, OSError) as exc:
		raise_for_network_error(exc, upload_endpoint)
	finally:
		if staging_dir is not None:
			shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
	args = parse_args()
	run_path = resolve_run_path(args.run_path)
	upload_run(
		run_path,
		include_all_ckpts=args.include_all_ckpts,
		include_wandb=args.include_wandb,
		remote_prefix=args.remote_prefix,
		remote_dir=args.remote_dir,
	)


if __name__ == "__main__":
	main()
