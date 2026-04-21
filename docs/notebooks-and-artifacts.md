# Notebooks & artifact directories

## `dspg/notebooks/main.ipynb`

| Item | Description |
|------|-------------|
| **Role** | Primary **GE Huggett** workflow: aligned with the GE illustration in the paper — DSPG from distributions, steady-state rate, simulation, etc. |
| **How to run** | Start Jupyter with the **repository root** as cwd so `results/` and `figures_tables/` resolve; set `CUDA_VISIBLE_DEVICES` in the first code cell if needed. |
| **Outputs** | Plotting cells write PDFs (and related files) under **`figures_tables/`**, as described in the root README. |

---

## `dspg/notebooks/ablation_study.ipynb`

| Item | Description |
|------|-------------|
| **Role** | **Batch-size** (and related) ablations: same GE setup as [`ablation_study.py`](../dspg/ablation_study.py); emphasizes loading **`results/DSPG_*.pkl`** for figures and SPG comparisons. |
| **Requirements** | Re-running training needs GPU + JAX; reproducing figures only requires matching `results/*.pkl` on disk. |

---

## `results/` (usually not committed)

| Example artifact | Produced by |
|------------------|-------------|
| `pe_vfi.npz` | [`pe_vfi.py`](../dspg/pe_vfi.py) |
| `pe_dspg_*.pkl` | [`pe_dspg.py`](../dspg/pe_dspg.py) |
| `pe_ppo_*` / `pe_sac_*` / `pe_ddpg_*` | respective baseline scripts |
| `DSPG_bs*_lr*_ep*.pkl` | [`ablation_study.py`](../dspg/ablation_study.py) |

After cloning, the folder may be missing until you train — scripts create it as needed. Do not commit large pickles (see root `.gitignore`).

---

## `figures_tables/` (tracked)

| Contents | Notes |
|----------|------|
| PDF figures | Training curves, ablations, comparisons vs SPG, etc. |
| `.tex` fragments | e.g. `pe_table.tex`, `\input` into the manuscript |

Producers: `plot_pe_training_comparison`, `pe_plot`, and plotting cells in the notebooks.

---

## nbformat note

For **GitHub** to render `.ipynb` correctly, **`stream` outputs** must include `name` (`stdout` / `stderr`), **`display_data`** must include `metadata`, and **`execute_result`** must include `execution_count`. Validate locally with `nbformat` to avoid an “Invalid Notebook” page.
