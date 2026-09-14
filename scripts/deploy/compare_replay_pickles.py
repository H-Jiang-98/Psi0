#!/usr/bin/env python3
"""Per-modality L1 between the actions of two replay pickles (mock_psi0_client_rtc dumps).

The pickles store the RAW sonic action layout -- hand(14) + neck(2) + token(64) -- see
write_replay_pickle() in src/psi/deploy/mock_psi0_client_rtc.py. The grouping and the
"mean |a - b| over frames, then mean over dims" reduction mirror the per-modality
denormalized-L1 block of that client (ACTION_LABELS / ACTION_SPLITS).

    python scripts/deploy/compare_replay_pickles.py A.pkl B.pkl [-n 30]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Raw-layout spans, from write_replay_pickle: HAND_DIM=14, NECK_DIM=2, TOKEN_DIM=64.
# "body" is the 64-D latent/token channel the WBC decodes into whole-body motion.
GROUPS = {"body": (16, 80), "hand": (0, 14), "neck": (14, 16)}


def load_actions(path: Path) -> tuple[np.ndarray, str]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["ticks"]["action"], dtype=np.float32), str(d.get("task", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("-n", "--num-actions", type=int, default=30, help="Frames to compare (default 30).")
    ap.add_argument("--plot", type=Path, default=None,
                    help="Save a per-dimension L1 plot of the body_tokens channel to this PNG.")
    args = ap.parse_args()

    a, task_a = load_actions(args.a)
    b, task_b = load_actions(args.b)
    n = min(args.num_actions, len(a), len(b))
    print(f"A: {args.a.name}  {a.shape}  | {task_a}")
    print(f"B: {args.b.name}  {b.shape}  | {task_b}")
    if task_a != task_b:
        print("WARNING: the two pickles are different episodes/tasks — this L1 is not an ablation.")
    print(f"comparing first {n} actions\n")

    err = np.abs(a[:n] - b[:n])                    # (n, 80)
    mean_err = err.mean(axis=0)                    # (80,) per-dim mean over frames
    print(f"{'group':<6} {'dims':>7} {'L1':>10} {'max|d|':>10}")
    for name, (lo, hi) in GROUPS.items():
        print(f"{name:<6} {f'{lo}:{hi}':>7} {mean_err[lo:hi].mean():>10.6f} {err[:, lo:hi].max():>10.6f}")
    print(f"{'all':<6} {'0:80':>7} {mean_err.mean():>10.6f} {err.max():>10.6f}")

    print("\nper-frame L1 (body / hand / neck):")
    for i in range(n):
        cols = "  ".join(f"{name} {err[i, lo:hi].mean():.6f}" for name, (lo, hi) in GROUPS.items())
        print(f"  t={i:>3}  {cols}")

    if args.plot is not None:
        plot_body_tokens(err, n, args.plot, args.a.name, args.b.name)


def plot_body_tokens(err: np.ndarray, n: int, out: Path, name_a: str, name_b: str) -> None:
    """Per-dimension L1 over the 64-D body/token channel, plus its frame x dim heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = GROUPS["body"]
    body = err[:, lo:hi]                 # (n, 64)
    per_dim = body.mean(axis=0)          # (64,)
    dims = np.arange(lo, hi)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[2, 1.4])
    ax.bar(dims, per_dim, color="tab:blue", width=0.8)
    ax.axhline(per_dim.mean(), color="tab:red", linestyle="--", linewidth=1.2,
               label=f"mean {per_dim.mean():.4f}")
    for d in np.argsort(per_dim)[-5:]:   # label the 5 worst dims
        ax.annotate(f"{lo + d}", (lo + d, per_dim[d]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean |A - B|")
    ax.set_title(f"body_tokens per-dimension L1 over first {n} actions\n{name_a}  vs  {name_b}",
                 fontsize=10)
    ax.legend()
    ax.margins(x=0.005)

    im = ax2.imshow(body.T, aspect="auto", origin="lower", cmap="magma",
                    extent=(0, n, lo - 0.5, hi - 0.5))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("action dim")
    fig.colorbar(im, ax=ax2, label="|A - B|", pad=0.01)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    worst = np.argsort(per_dim)[::-1][:5]
    print(f"\nworst body dims: {[(int(lo + d), round(float(per_dim[d]), 4)) for d in worst]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
