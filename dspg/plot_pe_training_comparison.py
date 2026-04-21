"""
PE training comparison: VFI, DSPG, PPO, SAC, DDPG (when pkls are present).

DSPG: fixed-ergodic-g eval epochs. PPO / SAC / DDPG share pickle keys ``curve_log_update_indices``
and ``curve_ergodic_mean_discounted_u_at_log``. Bands: ±1.96·std across runs (Z_BAND).

Output PDF under figures/. Writes ``pe_table.tex`` (standalone table) and
``pe_table_layout.tex`` (left table + right training-curve PDF in one ``figure``).

Run: python plot_pe_training_comparison.py
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dspg.repo_paths import REPO_ROOT

Z_BAND = 1.00

# Distinct plot markers (DSPG / PPO / SAC / DDPG must all differ)
MARKER_DSPG = "D"
MARKER_PPO = "o"
MARKER_SAC = "s"
MARKER_DDPG = "^"


def pick_largest_experiment_family(paths: list[str], algo: str) -> list[str]:
    """
    If results/ mixes e.g. U500 vs U1000 or R1 vs R10, keep the largest (U,R) group by file count.
    Tie-break: higher total_updates U, then newer mtime.
    """
    if len(paths) <= 1:
        return paths
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for p in paths:
        name = Path(p).name
        um = re.search(r"_U(\d+)_", name)
        rm = re.search(r"_R(\d+)_rep", name)
        if not um or not rm:
            groups[(-1, -1)].append(p)
            continue
        groups[(int(um.group(1)), int(rm.group(1)))].append(p)
    if len(groups) == 1:
        return sorted(paths)
    def score(kv: tuple[tuple[int, int], list[str]]) -> tuple[int, int, float]:
        (u, r), lst = kv
        mt = max(Path(x).stat().st_mtime for x in lst)
        return (len(lst), u, mt)

    best_key, best_list = max(groups.items(), key=lambda kv: score(kv))
    if len(best_list) < len(paths):
        u0, r0 = best_key
        print(
            f"Note: {algo} — using {len(best_list)} pkls (U={u0}, R={r0}); "
            f"ignored {len(paths) - len(best_list)} other file(s).",
            flush=True,
        )
    return sorted(best_list)


def stack_ergodic_curves_at_log(paths: list[str], algo_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load pe_ppo / pe_sac style pkls; return x (update indices), Y (n_rep, n_points)."""
    x_ref = None
    rows: list[np.ndarray] = []
    for p in paths:
        with open(p, "rb") as f:
            o = pickle.load(f)
        ex = np.asarray(o["curve_log_update_indices"], dtype=np.int32)
        ey = np.asarray(o["curve_ergodic_mean_discounted_u_at_log"], dtype=np.float64)
        if x_ref is None:
            x_ref = ex
        elif not np.array_equal(x_ref, ex):
            raise ValueError(f"Inconsistent {algo_name} log indices: {p}")
        rows.append(ey)
    assert x_ref is not None
    return x_ref.astype(np.float64), np.stack(rows, axis=0)


def stats_across_runs(vals: list[float]) -> tuple[float, float]:
    """Return (mean, sample variance). Single run: variance 0."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    v = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    return m, v


def rl_algo_table_stats(paths: list[str]) -> dict[str, float] | None:
    """From PPO/SAC/DDPG pkls: last & best ergodic eval U per run. Keys: last_*, best_* means/vars."""
    if not paths:
        return None
    last_v: list[float] = []
    best_v: list[float] = []
    for p in paths:
        with open(p, "rb") as f:
            o = pickle.load(f)
        y = np.asarray(o["curve_ergodic_mean_discounted_u_at_log"], dtype=np.float64)
        if y.size == 0:
            continue
        last_v.append(float(y[-1]))
        best_v.append(float(np.max(y)))
    if not last_v:
        return None
    lm, lv = stats_across_runs(last_v)
    bm, bv = stats_across_runs(best_v)
    return {
        "last_mean": lm,
        "last_var": lv,
        "best_mean": bm,
        "best_var": bv,
    }


def dspg_table_stats(R_u: np.ndarray) -> dict[str, float] | None:
    """R_u shape (n_rep, n_epochs): last-epoch and best-per-run discounted U."""
    if R_u.size == 0:
        return None
    last_per = R_u[:, -1]
    best_per = np.max(R_u, axis=1)
    lm, lv = stats_across_runs(last_per.tolist())
    bm, bv = stats_across_runs(best_per.tolist())
    return {
        "last_mean": lm,
        "last_var": lv,
        "best_mean": bm,
        "best_var": bv,
    }


PE_TABLE_CAPTION = (
    "Partial-equilibrium methods: ergodic discounted utility (avg and sample variance "
    "across random seeds). Bold: best avg per column; best (lowest) var per column."
)

PE_FIGURE_CAPTION = (
    "Training curves: DSPG vs PPO vs SAC vs DDPG vs VFI (ergodic discounted utility)."
)


def _build_pe_table_tabular_and_masks(
    *,
    vfi_u: float,
    dspg_stats: dict[str, float] | None,
    ppo_stats: dict[str, float] | None,
    sac_stats: dict[str, float] | None,
    ddpg_stats: dict[str, float] | None,
) -> tuple[list[tuple[str, float, float, float, float]], list[list[bool]]]:
    tol = 1e-6
    table_rows: list[tuple[str, float, float, float, float]] = []
    for label, st in (
        ("DSPG", dspg_stats),
        ("PPO", ppo_stats),
        ("SAC", sac_stats),
        ("DDPG", ddpg_stats),
    ):
        if st is None:
            continue
        table_rows.append(
            (
                label,
                float(st["last_mean"]),
                float(st["last_var"]),
                float(st["best_mean"]),
                float(st["best_var"]),
            )
        )
    table_rows.append(("VFI", float(vfi_u), float("nan"), float(vfi_u), float("nan")))

    bold_mask: list[list[bool]] = [[False, False, False, False] for _ in table_rows]
    for j in range(4):
        col = [table_rows[i][j + 1] for i in range(len(table_rows))]
        finite = [x for x in col if np.isfinite(x)]
        if not finite:
            continue
        if j in (0, 2):
            target = max(finite)
            for i, x in enumerate(col):
                if np.isfinite(x) and abs(x - target) <= tol:
                    bold_mask[i][j] = True
        else:
            target = min(finite)
            for i, x in enumerate(col):
                if np.isfinite(x) and abs(x - target) <= tol:
                    bold_mask[i][j] = True
    return table_rows, bold_mask


def _pe_table_tabular_lines(
    table_rows: list[tuple[str, float, float, float, float]],
    bold_mask: list[list[bool]],
) -> list[str]:
    def fmt_mean(v: float, bold: bool) -> str:
        if not np.isfinite(v):
            return "---"
        s = f"{v:.4f}"
        return f"\\textbf{{{s}}}" if bold else s

    def fmt_var(v: float, bold: bool) -> str:
        if not np.isfinite(v):
            return "---"
        s = f"{v:.6f}"
        return f"\\textbf{{{s}}}" if bold else s

    lines = [
        r"{\setlength{\heavyrulewidth}{1.2pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multicolumn{2}{c}{Last eval} & \multicolumn{2}{c}{Best eval} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & avg & var & avg & var \\",
        r"\midrule",
    ]
    for i, (name, lm, lv, bm, bv) in enumerate(table_rows):
        if name == "VFI" and i > 0:
            lines.append(r"\specialrule{.1em}{0pt}{0pt}")
        b = bold_mask[i]
        lines.append(
            f"{name} & {fmt_mean(lm, b[0])} & {fmt_var(lv, b[1])} & "
            f"{fmt_mean(bm, b[2])} & {fmt_var(bv, b[3])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    return lines


def write_pe_table_tex(
    path: Path,
    *,
    vfi_u: float,
    dspg_stats: dict[str, float] | None,
    ppo_stats: dict[str, float] | None,
    sac_stats: dict[str, float] | None,
    ddpg_stats: dict[str, float] | None,
) -> None:
    """Standalone table only (backward compatible \\input)."""
    table_rows, bold_mask = _build_pe_table_tabular_and_masks(
        vfi_u=vfi_u,
        dspg_stats=dspg_stats,
        ppo_stats=ppo_stats,
        sac_stats=sac_stats,
        ddpg_stats=ddpg_stats,
    )
    tabular = _pe_table_tabular_lines(table_rows, bold_mask)
    lines = [
        r"% Auto-generated by plot_pe_training_comparison.py",
        r"% Requires: \usepackage{booktabs, multirow}",
        r"% Last/Best: avg and sample variance across random seeds (ergodic eval curve).",
        r"% Bold: max per avg column, min per var column (among finite entries).",
        r"\begin{table}[htbp]",
        r"\centering",
        *tabular,
        r"\caption{" + PE_TABLE_CAPTION + "}",
        r"\label{tab:pe_ergodic_summary}",
        r"\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pe_table_layout_tex(
    path: Path,
    *,
    figure_pdf_basename: str,
    vfi_u: float,
    dspg_stats: dict[str, float] | None,
    ppo_stats: dict[str, float] | None,
    sac_stats: dict[str, float] | None,
    ddpg_stats: dict[str, float] | None,
) -> None:
    """Left: same table as pe_table.tex inner part; right: training-curve PDF. Requires caption package."""
    table_rows, bold_mask = _build_pe_table_tabular_and_masks(
        vfi_u=vfi_u,
        dspg_stats=dspg_stats,
        ppo_stats=ppo_stats,
        sac_stats=sac_stats,
        ddpg_stats=ddpg_stats,
    )
    tabular = _pe_table_tabular_lines(table_rows, bold_mask)
    # Escape braces in captions for \captionof{table}{...} — captions are plain ASCII
    lines = [
        r"% Auto-generated by plot_pe_training_comparison.py",
        r"% Requires: \usepackage{caption, booktabs, multirow}",
        r"% Left table + right figure. In main doc: \input{figures/pe_table_layout.tex}",
        r"% or \graphicspath{{figures/}} so the PDF name resolves.",
        r"% \\noindent + \\vspace{0pt} fix vertical misalignment of side-by-side [t] minipages.",
        r"\begin{figure}[htbp]",
        r"\noindent",
        r"\begin{minipage}[t]{0.49\textwidth}",
        r"\vspace{0pt}",
        r"\centering",
        *tabular,
        r"\captionof{table}{" + PE_TABLE_CAPTION + "}",
        r"\label{tab:pe_ergodic_summary}",
        r"\end{minipage}",
        r"\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}",
        r"\vspace{0pt}",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{{figure_pdf_basename}}}",
        r"\captionof{figure}{" + PE_FIGURE_CAPTION + "}",
        r"\label{fig:pe_training_curves}",
        r"\end{minipage}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def dspg_eval_epoch_indices(n_ep: int, eval_every: int) -> np.ndarray:
    """1-based epoch indices where DSPG runs fixed-g curve eval (matches pe_dspg.curve_eval_this_epoch)."""
    if eval_every <= 0:
        eval_every = 1
    if eval_every == 1:
        return np.arange(1, n_ep + 1, dtype=np.float64)
    out: list[int] = []
    seen: set[int] = set()
    for ep in range(n_ep):
        if ep == 0 or ep == n_ep - 1 or (ep + 1) % eval_every == 0:
            e1 = ep + 1
            if e1 not in seen:
                seen.add(e1)
                out.append(e1)
    return np.asarray(out, dtype=np.float64)


def set_styles():
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "lines.linewidth": 3,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--dspg_glob",
        type=str,
        default="pe_dspg_bs64_*_R10.pkl",
        help="Glob for DSPG pickle runs under results_dir (legacy pe_uspg_* tried if empty).",
    )
    parser.add_argument(
        "--ppo_glob",
        type=str,
        default="pe_ppo_*.pkl",
        help="If multiple U*/R* runs exist, the largest family is kept automatically.",
    )
    parser.add_argument(
        "--sac_glob",
        type=str,
        default="pe_sac_*.pkl",
        help="If multiple U*/R* runs exist, the largest family is kept automatically.",
    )
    parser.add_argument(
        "--ddpg_glob",
        type=str,
        default="pe_ddpg_*.pkl",
        help="If multiple U*/R* runs exist, the largest family is kept automatically.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pe_DSPG_PPO_SAC_DDPG_VFI_training_curves.pdf",
        help="Filename written under figures/.",
    )
    parser.add_argument(
        "--table_out",
        type=str,
        default="pe_table.tex",
        help="Standalone LaTeX table under figures/.",
    )
    parser.add_argument(
        "--layout_out",
        type=str,
        default="pe_table_layout.tex",
        help="Left-table right-figure LaTeX snippet under figures/ (uses same PDF as --out).",
    )
    args = parser.parse_args()

    root = REPO_ROOT
    res = root / args.results_dir
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    dspg_paths = sorted(glob.glob(str(res / args.dspg_glob)))
    if not dspg_paths:
        legacy = sorted(glob.glob(str(res / "pe_uspg_bs64_*_R10.pkl")))
        if legacy:
            print(
                "Note: using legacy pe_uspg_* pickles; rename to pe_dspg_* for consistency.",
                flush=True,
            )
            dspg_paths = legacy
    if not dspg_paths:
        raise FileNotFoundError(f"No DSPG files for glob {res / args.dspg_glob}")
    with open(dspg_paths[0], "rb") as f:
        dspg_pkl = pickle.load(f)
    R_u = np.asarray(dspg_pkl["cumulative_reward_per_epoch"], dtype=np.float64)
    vfi_from_dspg_pkl = float(dspg_pkl["vfi_ground_truth_utility"])
    cfg_u = dspg_pkl.get("config") or {}
    eval_every_u = int(cfg_u.get("eval_every_epochs", 1))

    npz_path = res / "pe_vfi.npz"
    if npz_path.is_file():
        vfi_npz = float(np.asarray(np.load(npz_path)["mean_discounted_utility"]).reshape(()))
    else:
        vfi_npz = vfi_from_dspg_pkl

    ppo_paths = pick_largest_experiment_family(
        sorted(glob.glob(str(res / args.ppo_glob))), "PPO"
    )
    sac_paths = pick_largest_experiment_family(
        sorted(glob.glob(str(res / args.sac_glob))), "SAC"
    )
    ddpg_paths = pick_largest_experiment_family(
        sorted(glob.glob(str(res / args.ddpg_glob))), "DDPG"
    )
    if not ppo_paths:
        print(f"Warning: no PPO files for {res / args.ppo_glob} — skipping PPO curves.", flush=True)
    if not sac_paths:
        print(f"Warning: no SAC files for {res / args.sac_glob} — skipping SAC curves.", flush=True)
    if not ddpg_paths:
        print(f"Warning: no DDPG files for {res / args.ddpg_glob} — skipping DDPG curves.", flush=True)

    ppo_pack: tuple[np.ndarray, np.ndarray] | None = None
    if ppo_paths:
        x_p, P_p = stack_ergodic_curves_at_log(ppo_paths, "PPO")
        ppo_pack = (x_p, P_p)

    sac_pack: tuple[np.ndarray, np.ndarray] | None = None
    if sac_paths:
        x_s, P_s = stack_ergodic_curves_at_log(sac_paths, "SAC")
        sac_pack = (x_s, P_s)

    ddpg_pack: tuple[np.ndarray, np.ndarray] | None = None
    if ddpg_paths:
        x_d, P_d = stack_ergodic_curves_at_log(ddpg_paths, "DDPG")
        ddpg_pack = (x_d, P_d)

    dspg_stats = dspg_table_stats(R_u)
    ppo_stats = rl_algo_table_stats(ppo_paths) if ppo_paths else None
    sac_stats = rl_algo_table_stats(sac_paths) if sac_paths else None
    ddpg_stats = rl_algo_table_stats(ddpg_paths) if ddpg_paths else None

    n_ep_u = R_u.shape[1]
    x_u = dspg_eval_epoch_indices(n_ep_u, eval_every_u)
    idx_u = (x_u - 1).astype(np.int64)
    R_u_eval = R_u[:, idx_u]
    mean_u = np.nanmean(R_u_eval, axis=0)
    std_u = np.nanstd(R_u_eval, axis=0, ddof=0)
    lo_u = mean_u - Z_BAND * std_u
    hi_u = mean_u + Z_BAND * std_u

    set_styles()
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    ax.fill_between(x_u, lo_u, hi_u, color="C3", alpha=0.3)
    plot_u_kw: dict = {
        "color": "C3",
        "linewidth": 3,
        "label": "DSPG (ours)",
        "marker": MARKER_DSPG,
        "markersize": 5,
    }
    ax.plot(x_u, mean_u, **plot_u_kw)

    if ppo_pack is not None:
        x_erg, P_erg = ppo_pack
        mean_pe = np.mean(P_erg, axis=0)
        std_pe = np.std(P_erg, axis=0, ddof=0)
        lo_pe = mean_pe - Z_BAND * std_pe
        hi_pe = mean_pe + Z_BAND * std_pe
        ax.fill_between(x_erg, lo_pe, hi_pe, color="royalblue", alpha=0.3)
        ax.plot(
            x_erg,
            mean_pe,
            color="royalblue",
            linewidth=3,
            marker=MARKER_PPO,
            markersize=5,
            label="PPO",
        )

    if sac_pack is not None:
        x_sac, P_sac = sac_pack
        mean_sac = np.mean(P_sac, axis=0)
        std_sac = np.std(P_sac, axis=0, ddof=0)
        lo_s = mean_sac - Z_BAND * std_sac
        hi_s = mean_sac + Z_BAND * std_sac
        ax.fill_between(x_sac, lo_s, hi_s, color="orange", alpha=0.3)
        ax.plot(
            x_sac,
            mean_sac,
            color="orange",
            linewidth=3,
            marker=MARKER_SAC,
            markersize=5,
            label="SAC",
        )

    if ddpg_pack is not None:
        x_d, P_d = ddpg_pack
        mean_d = np.mean(P_d, axis=0)
        std_d = np.std(P_d, axis=0, ddof=0)
        lo_d = mean_d - Z_BAND * std_d
        hi_d = mean_d + Z_BAND * std_d
        ax.fill_between(x_d, lo_d, hi_d, color="green", alpha=0.3)
        ax.plot(
            x_d,
            mean_d,
            color="green",
            linewidth=3,
            marker=MARKER_DDPG,
            markersize=5,
            label="DDPG",
        )

    ax.axhline(
        y=vfi_npz,
        color="gray",
        linestyle="--",
        linewidth=3,
        label=f"VFI (ground truth)",
    )
    if abs(vfi_npz - vfi_from_dspg_pkl) > 1e-3:
        ax.axhline(
            y=vfi_from_dspg_pkl,
            color="0.5",
            linestyle=":",
            linewidth=1.5,
            label=f"DSPG pkl VFI ref ({vfi_from_dspg_pkl:.3f})",
        )

    ax.set_xlabel("Epoch (DSPG) / RL update (PPO, SAC, DDPG)")
    ax.set_ylabel("Average discounted utility")
    ax.grid(True)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()

    out_path = figdir / args.out
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    table_path = figdir / args.table_out
    write_pe_table_tex(
        table_path,
        vfi_u=float(vfi_npz),
        dspg_stats=dspg_stats,
        ppo_stats=ppo_stats,
        sac_stats=sac_stats,
        ddpg_stats=ddpg_stats,
    )
    print(f"Saved {table_path}")

    layout_path = figdir / args.layout_out
    write_pe_table_layout_tex(
        layout_path,
        figure_pdf_basename=Path(args.out).name,
        vfi_u=float(vfi_npz),
        dspg_stats=dspg_stats,
        ppo_stats=ppo_stats,
        sac_stats=sac_stats,
        ddpg_stats=ddpg_stats,
    )
    print(f"Saved {layout_path}")


if __name__ == "__main__":
    main()
