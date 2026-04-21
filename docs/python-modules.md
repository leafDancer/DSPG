# Python modules (`dspg/`)

Paths below are relative to the repository root. Entry pattern: `python -m dspg.<module> [options]` from the repo root.

---

## `dspg/__init__.py`

- **Role:** Package marker and docstring; documents importing via `dspg.pe_*` and running with `python -m dspg.<module>`.
- **Dependencies:** No runtime logic.

---

## `dspg/repo_paths.py`

| Item | Description |
|------|-------------|
| **Purpose** | Defines **`REPO_ROOT`** as `Path(__file__).resolve().parents[1]` for joining `results/` and `figures_tables/`. |
| **Used by** | All scripts that write paths: `pe_vfi`, `pe_dspg`, `pe_plot`, `plot_pe_training_comparison`, etc. |
| **Usage** | Not meant to be executed standalone; changing it affects output locations globally. |

---

## `dspg/pe_rl_env.py`

| Item | Description |
|------|-------------|
| **Purpose** | **Partial-equilibrium environment `PEEnv`:** Markov grids for productivity \(e\), interest rate \(r\), wage factor \(w\); continuous assets \(a\); action is a **consumption share** of cash-on-hand in \([0,1]\); reward is flow utility (e.g. log). Truncated horizon `T` is derived from `beta` and a tolerance. |
| **Structure** | Class API: `reset()` / `step(state, action)` in a Gym-like style (returns `trunc` when the horizon ends). |
| **Dependencies** | `numpy`. |
| **Usage** | Imported by all `pe_*` trainers and `pe_vfi`; optional smoke test via `python -m dspg.pe_rl_env`. |

---

## `dspg/pe_vfi.py`

| Item | Description |
|------|-------------|
| **Purpose** | **Value function iteration (VFI)** on the **PEEnv** state tensor; Monte Carlo discounted utility for benchmarking; writes **`results/pe_vfi.npz`** (plus a small `pe_vfi_meta.pkl`). |
| **Dependencies** | `PEEnv`, `jax` / `jax.numpy`, `repo_paths`. |
| **Prerequisites** | None (PE baseline source). |
| **Typical command** | `python -m dspg.pe_vfi --cuda 0` |
| **Common flags** | `--cuda` GPU id; `--burn_in` MC burn-in steps; `--seed`. |
| **Outputs** | `results/pe_vfi.npz`: **ground truth / bounds read by `pe_dspg` and RL baselines**. |

---

## `dspg/pe_dspg.py`

| Item | Description |
|------|-------------|
| **Purpose** | **DSPG** training on the PE environment: cross-section \(g\) with shape `(B, na, ne)`, Haiku policy, `optax` optimizer; epoch-wise curves vs VFI. |
| **Dependencies** | `PEEnv`, `pe_vfi.npz`, `jax`, `haiku`, `optax`, `matplotlib`. |
| **Prerequisites** | **`results/pe_vfi.npz`** must exist (override directory with `--results_dir` if needed). |
| **Typical command** | `python -m dspg.pe_dspg --cuda 0` |
| **Common flags** | `--batch_size`, `--epoch`, `--repeats`, `--lr`, `--cuda`, `--results_dir`, `--eval_every` (curve metric cadence), `--skip_plot`. |
| **Outputs** | `results/pe_dspg_bs*_lr*_ep*_H*_R*.pkl` (`cumulative_reward_per_epoch`, `vfi_ground_truth_utility`, `config`); optional `*_training_curve.pdf`. |

---

## `dspg/pe_ppo.py` / `dspg/pe_sac.py` / `dspg/pe_ddpg.py`

| Item | Description |
|------|-------------|
| **Purpose** | Standard RL baselines: **PPO** (Beta policy head), **SAC** (squashed Gaussian), **DDPG** (deterministic actor + twin Q), same `PEEnv`. |
| **Dependencies** | `PEEnv`, `jax`, `haiku`, `optax` (details vary per file). |
| **Prerequisites** | Usually **`pe_vfi.npz`** for evaluation / comparison (see each module docstring). |
| **Typical command** | `python -m dspg.pe_ppo --cuda 0` (same pattern for SAC, DDPG). |
| **Outputs** | Pickles whose keys include those expected by [`plot_pe_training_comparison`](../dspg/plot_pe_training_comparison.py) (e.g. `curve_log_update_indices`, `curve_ergodic_mean_discounted_u_at_log`). |

---

## `dspg/pe_plot.py`

| Item | Description |
|------|-------------|
| **Purpose** | Load **`pe_dspg`** shards from **`results/`** via glob (merge by `shard_id` when multiple files exist); plot training curves vs VFI into a PDF. |
| **Dependencies** | `matplotlib`, `repo_paths`; no training. |
| **Typical command** | `python -m dspg.pe_plot --pattern 'pe_dspg_bs64_*_R10.pkl'` |
| **Notes** | If no `pe_dspg_*` match, falls back to legacy `pe_uspg_*` with a console note. |

---

## `dspg/plot_pe_training_comparison.py`

| Item | Description |
|------|-------------|
| **Purpose** | **Multi-algorithm comparison:** overlays **DSPG, PPO, SAC, DDPG** vs **VFI** when pickles exist; writes comparison PDF and LaTeX snippets under **`figures_tables/`** (`pe_table.tex`, `pe_table_layout.tex`, etc.). |
| **Dependencies** | `repo_paths`, `matplotlib`, `numpy`; reads `results/*.pkl`. |
| **Typical command** | `python -m dspg.plot_pe_training_comparison` |
| **Flags** | Override globs with `--dspg_glob`, `--ppo_glob`, …; when multiple experiment families coexist, the script tends to pick the largest `(U,R)` group by file count. |

---

## `dspg/ablation_study.py`

| Item | Description |
|------|-------------|
| **Purpose** | **GE Huggett** trainer: bond supply, Markov TFP, bond-market clearing for \(r\); **U-Net-style** policy on cross-section \(g\); structural policy-gradient updates. |
| **Structure** | **`argparse` runs at top of file, before `import jax`**, so `CUDA_VISIBLE_DEVICES` is set first — do not move parsing below the JAX import without revisiting that design. |
| **Typical command** | `python -m dspg.ablation_study --cuda 0 --batch_size 64 --epoch 1000 --lr 2e-3` |
| **Outputs** | `results/DSPG_bs{batch}_lr{lr}_ep{epoch}.pkl` (training history, etc.). |
| **Notebook** | Same GE setup as [`notebooks/ablation_study.ipynb`](../dspg/notebooks/ablation_study.ipynb); the notebook focuses on loading saved `results/` for figures. |

---

## One-line summary

| File | One-liner |
|------|-----------|
| `repo_paths.py` | Repository root `REPO_ROOT`. |
| `pe_rl_env.py` | PE environment `PEEnv`. |
| `pe_vfi.py` | PE VFI → `pe_vfi.npz`. |
| `pe_dspg.py` | PE DSPG training. |
| `pe_ppo.py` / `pe_sac.py` / `pe_ddpg.py` | PE RL baselines. |
| `pe_plot.py` | Single-algorithm DSPG curves vs VFI. |
| `plot_pe_training_comparison.py` | Multi-algorithm + VFI comparison and paper tables. |
| `ablation_study.py` | GE Huggett batch training. |
