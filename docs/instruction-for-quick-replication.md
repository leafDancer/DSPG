# Quick replication guide for `figures_tables/`

This guide maps **artifacts under [`figures_tables/`](../figures_tables/)** to the **training runs and plotting steps** that produce them. Run everything from the **repository root** with GPU packages installed ([`requirements.txt`](../requirements.txt), JAX/CUDA as in the root [`README.md`](../README.md)).

**Dependency:** [`results/`](../README.md#results-directory) is gitignored. After cloning you must **train locally** (or copy pickles into `results/`) before plotting scripts can regenerate PDFs and LaTeX snippets.

---

## 1. Partial equilibrium (PE): comparison figure + LaTeX table

These files are produced by **`dspg/plot_pe_training_comparison.py`**:

| Output under `figures_tables/` | Producer |
|----------------------------------|----------|
| `pe_DSPG_PPO_SAC_DDPG_VFI_training_curves.pdf` | Default `--out` from `plot_pe_training_comparison` |
| `pe_table.tex` | Default `--table_out` |
| `pe_table_layout.tex` | Default `--layout_out` (includes the PDF above) |

The plotting script reads **pickles in `results/`**; it does not train models.

### 1.1 Required upstream experiments

1. **VFI baseline** (creates `results/pe_vfi.npz`):

   ```bash
   python -m dspg.pe_vfi --cuda 0
   ```

2. **DSPG on PE** (creates `results/pe_dspg_*.pkl`; default glob used by the plotter is `pe_dspg_bs64_*_R10.pkl` — match batch size / repeats / tag when you train):

   ```bash
   python -m dspg.pe_dspg --cuda 0
   ```

3. **Optional — full multi-algorithm figure:** train baselines so the comparison plot includes all curves:

   ```bash
   python -m dspg.pe_ppo --cuda 0
   python -m dspg.pe_sac --cuda 0
   python -m dspg.pe_ddpg --cuda 0
   ```

   If PPO/SAC/DDPG pickles are missing, the script **still runs** and emits warnings; it will plot DSPG + VFI only.

### 1.2 Generate the PDF and `.tex` files

```bash
python -m dspg.plot_pe_training_comparison
```

Override globs if your filenames differ, e.g. `--dspg_glob 'pe_dspg_bs64_*_R10.pkl'` (defaults are documented in `python -m dspg.plot_pe_training_comparison --help`).

### 1.3 Optional: DSPG-only training curve (not the paper comparison layout)

- **`dspg/pe_plot.py`** merges DSPG shards and writes a PDF (default under `results/` unless `--out` points into `figures_tables/`).
- **`dspg/pe_dspg.py`** can also save `*_training_curve.pdf` **under `results/`** next to the pickle (`--skip_plot` disables).

---

## 2. General equilibrium (GE): `main.ipynb` figures

Open **[`dspg/notebooks/main.ipynb`](../dspg/notebooks/main.ipynb)** with Jupyter, **cwd = repo root**, GPU set in the first code cell if needed. Execute the notebook through the plotting cells.

| Output under `figures_tables/` | Typical notebook cell |
|--------------------------------|------------------------|
| `steady_state_results.pdf` | Saves via `fig.savefig("./figures_tables/steady_state_results.pdf")` |
| `dynamic_model_results.pdf` | Saves via `fig.savefig("./figures_tables/dynamic_model_results.pdf")` |

Training/simulation cells must have run successfully before those figures reflect new runs.

---

## 3. General equilibrium (GE): ablation / SPG comparison figures

Workflow: **train** with **[`dspg/ablation_study.py`](../dspg/ablation_study.py)** (writes `results/DSPG_bs{batch}_lr{lr}_ep{epoch}.pkl`), then **plot** with **[`dspg/notebooks/ablation_study.ipynb`](../dspg/notebooks/ablation_study.ipynb)** using the same repo-root cwd.

| Output under `figures_tables/` | Source in notebook |
|--------------------------------|-------------------|
| `DSPG_ablation(bs).pdf` | Batch-size sensitivity plot |
| `DSPG_vs_SPG_ergodic_dist.pdf` | Ergodic distribution comparison vs SPG |
| `DSPG_vs_SPG_consumption.pdf` | Consumption comparison under fixed shocks |

The ablation notebook expects **`results/*.pkl`** from your training (and, for SPG panels, the **SPG baseline pickles** paths used in the notebook — adjust if your filenames differ). Re-execute training cells only if you need fresh checkpoints.

---

## 4. `hyper-params.tex` (appendix tables)

**[`figures_tables/hyper-params.tex`](../figures_tables/hyper-params.tex)** is a **LaTeX appendix fragment** for the paper (PE and GE hyper-parameters, baselines, etc.). It is **maintained alongside the code** and is **not** overwritten by the Python training or plotting scripts. After you change defaults in scripts, update this file if the paper should stay consistent.

---

## 5. Suggested order for a full local pass

| Step | Action |
|------|--------|
| A | `pe_vfi` → `pe_dspg` → (optional) `pe_ppo`, `pe_sac`, `pe_ddpg` |
| B | `plot_pe_training_comparison` → refresh `pe_*.tex` / comparison PDF |
| C | Run `main.ipynb` for GE steady/dynamic figures |
| D | Run `ablation_study.py` for GE pickles; run `ablation_study.ipynb` for GE ablation / SPG PDFs |

For architecture and per-module CLI details, see [**architecture.md**](architecture.md) and [**python-modules.md**](python-modules.md).
