"""
Load one or more PE DSPG pickle files (same experiment tag) and plot mean ± 96% band vs VFI.
Pass --pattern to glob under results_dir (single run: one file; multiple: concatenated by shard_id if present, else sorted path order).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dspg.repo_paths import REPO_ROOT

# Two-sided 96% Gaussian multiplier (P in [0.02, 0.98])
Z_96 = 2.053748910641517


def set_static_styles():
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        }
    )


def load_and_stack_shards(
    results_dir: Path,
    pattern: str,
) -> tuple[np.ndarray, float, dict]:
    paths = sorted(glob.glob(str(results_dir / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched {results_dir / pattern}")

    shards: list[tuple[int, np.ndarray, dict]] = []
    vfi_gt = None
    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        cfg = obj.get("config", {})
        sid = int(cfg.get("shard_id", 0))
        rewards = np.asarray(obj["cumulative_reward_per_epoch"], dtype=np.float64)
        shards.append((sid, rewards, cfg))
        if vfi_gt is None:
            vfi_gt = float(obj["vfi_ground_truth_utility"])
        else:
            if abs(float(obj["vfi_ground_truth_utility"]) - vfi_gt) > 1e-6:
                raise ValueError("Inconsistent vfi_ground_truth_utility across shards")

    shards.sort(key=lambda x: x[0])
    stacked = np.concatenate([s[1] for s in shards], axis=0)
    meta = {
        "shard_paths": paths,
        "num_shards": len(shards),
        "repeats_stacked": stacked.shape[0],
        "epochs": stacked.shape[1],
    }
    return stacked, float(vfi_gt), meta


def resolve_pattern(results_dir: Path, pattern: str) -> str:
    """If default ``pe_dspg_*`` finds nothing, try legacy ``pe_uspg_*``."""
    if sorted(glob.glob(str(results_dir / pattern))):
        return pattern
    if "pe_dspg" in pattern:
        alt = pattern.replace("pe_dspg", "pe_uspg")
        if sorted(glob.glob(str(results_dir / alt))):
            print(
                "Note: using legacy pe_uspg_* pickles; rename to pe_dspg_* for consistency.",
                flush=True,
            )
            return alt
    return pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing shard pkls and output figure.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="pe_dspg_bs64_*_R10.pkl",
        help="Glob under results_dir (e.g. pe_dspg_bs64_*_R10.pkl; legacy pe_uspg_* fallback).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PDF path (default: results/pe_dspg_merged_training_curve.pdf).",
    )
    args = parser.parse_args()

    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cuda"

    results_dir = (REPO_ROOT / args.results_dir).resolve()
    pattern = resolve_pattern(results_dir, args.pattern)
    rewards, vfi_gt, meta = load_and_stack_shards(results_dir, pattern)

    n_rep, n_ep = rewards.shape
    epochs = np.arange(1, n_ep + 1, dtype=np.float64)
    mean_u = np.mean(rewards, axis=0)
    std_u = np.std(rewards, axis=0, ddof=0)
    lo = mean_u - Z_96 * std_u
    hi = mean_u + Z_96 * std_u

    set_static_styles()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.fill_between(
        epochs,
        lo,
        hi,
        color="red",
        alpha=0.3,
        label=f"96% band (±{Z_96:.3f} std across runs)",
    )
    ax.plot(epochs, mean_u, color="red", linewidth=2, label="DSPG (mean across runs)")
    ax.axhline(y=vfi_gt, color="gray", linestyle="--", linewidth=2, label="VFI ground truth")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Discounted cumulative utility (mean over batch)")
    ax.set_title("PE environment: DSPG vs VFI")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else (results_dir / "pe_dspg_merged_training_curve.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {meta['num_shards']} shards, {meta['repeats_stacked']} runs × {meta['epochs']} epochs.")
    print(f"VFI ground truth: {vfi_gt:.6f}")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
