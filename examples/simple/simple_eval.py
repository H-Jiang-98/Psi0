#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from simple.evals.api import EvalConfig, EvalRunner


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pick_free_port(start: int = 22085, end: int = 22999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _wait_for_port(host: str, port: int, tries: int, sleep_s: float) -> None:
    for _ in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(sleep_s)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _build_server_cmd(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "psi.deploy.serve_psi0_simple",
        "--host",
        args.server_host,
        "--port",
        str(port),
        "--policy",
        args.policy,
        "--run-dir",
        args.run_dir,
        "--ckpt-step",
        str(args.ckpt_step),
        "--device",
        args.device,
    ]
    if args.action_exec_horizon is not None:
        cmd.extend(["--action-exec-horizon", str(args.action_exec_horizon)])
    if args.rtc:
        cmd.append("--rtc")
    if args.rtc_inference_delay is not None:
        cmd.extend(["--rtc-inference-delay", str(args.rtc_inference_delay)])
    if args.min_exec_horizon is not None:
        cmd.extend(["--min-exec-horizon", str(args.min_exec_horizon)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PSI checkpoint with SIMPLE.")
    parser.add_argument("--run-dir", required=True, help="PSI run directory.")
    parser.add_argument("--ckpt-step", required=True, help="Checkpoint step or 'latest'.")
    parser.add_argument("--env-id", default="simple/G1WholebodyBendPick-v0")
    parser.add_argument("--policy", default="psi0")
    parser.add_argument("--data-dir", required=True, help="LeRobot dataset root for SIMPLE eval.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--host", default="localhost", help="Host used by SIMPLE eval client.")
    parser.add_argument("--port", type=int, help="Policy server port. Defaults to a free port.")
    parser.add_argument("--sim-mode", default="mujoco_isaac")
    parser.add_argument("--data-format", default="lerobot")
    parser.add_argument("--eval-dir", default="data/evals")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=360)
    parser.add_argument("--success-criteria", type=float, default=0.9)
    parser.add_argument("--action-exec-horizon", type=int)
    parser.add_argument("--wait-tries", type=int, default=1200)
    parser.add_argument("--wait-sleep-s", type=float, default=0.1)
    parser.add_argument("--server-log", help="Optional path for server stdout/stderr.")
    parser.add_argument("--rtc", action="store_true")
    parser.add_argument("--rtc-inference-delay", type=int,
                        help="RTC 'd': rows of each new chunk reproduced exactly from "
                             "the previous plan. Required (>=1) for train-time RTC, "
                             "whose frozen prefix is its only conditioning channel.")
    parser.add_argument("--min-exec-horizon", type=int,
                        help="RTC 's': the guidance mask's free-region boundary (H-s). "
                             "Must be >= --action-exec-horizon. Raise it to pull the "
                             "ramp back off the chunk seam.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    port = args.port or _pick_free_port()
    server_cmd = _build_server_cmd(args, port)
    eval_config = EvalConfig(
        env_id=args.env_id,
        policy=args.policy,
        split=args.split,
        host=args.host,
        port=port,
        data_format=args.data_format,
        sim_mode=args.sim_mode,
        headless=args.headless,
        eval_dir=args.eval_dir,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.num_episodes,
        episode_start=args.episode_start,
        data_dir=args.data_dir,
        success_criteria=args.success_criteria,
        save_video=args.save_video,
        num_workers=args.num_workers,
    )

    print(f"server_cmd={' '.join(server_cmd)}")
    print(f"eval_config={eval_config}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

    log_handle = None
    server = None
    try:
        if args.server_log:
            log_path = Path(args.server_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w")
            server = subprocess.Popen(
                server_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        else:
            server = subprocess.Popen(server_cmd, cwd=REPO_ROOT, env=env)

        _wait_for_port(args.host, port, args.wait_tries, args.wait_sleep_s)
        result = EvalRunner(eval_config).run()
        print(f"success_rate={result.success_rate:.6f}")
        print(f"log_path={result.log_path}")
        print(f"eval_dir={result.eval_dir}")
        return 0
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
