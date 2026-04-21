# DSPG: Distribution-based Structural Policy Gradient

This repository is the **official implementation** accompanying the **DSPG** (Distribution-based Structural Policy Gradient) paper: the code here is what we use to reproduce the paper’s experiments and numerical results.

**Paper figures and tables:** The [`figures_tables/`](figures_tables/) folder contains the **same materials that appear in the paper** — the exported **figure PDFs** (e.g. training curves, ablation and comparison plots) and the **LaTeX table fragments** actually `\input` in the manuscript (`pe_table.tex`, `pe_table_layout.tex`, plus the hyper-parameter appendix in `hyper-params.tex`). Plotting scripts may write or refresh some of these files when you regenerate results locally. **To quickly reproduce those outputs**, read [`docs/instruction-for-quick-replication.md`](docs/instruction-for-quick-replication.md) first.

Abbreviation **DSPG** stands for **Distribution-based Structural Policy Gradient** (emphasizes cross-sectional distributions over agents, not “distributional RL” over return distributions).

## References

The **DSPG** paper does not appear in isolation: it **builds directly on** the two lines of work below. Taken together, they supply the **structural reinforcement learning** viewpoint and the **structural policy-gradient** machinery that DSPG extends to **distribution-based** updates on cross-sectional masses. Treat them as the **conceptual and methodological foundation** for this repository and the DSPG manuscript.

| Paper | How it relates to DSPG |
|--------|-------------------------|
| **[Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics](https://arxiv.org/abs/2512.18892)** (arXiv:2512.18892) | **Foundation for structural RL (SRL)** in heterogeneous-agent macro—equilibrium objects from simulation, HA environments with learned prices—on which GE experiments here are aligned. |
| **[Recurrent Structural Policy Gradient for Partially Observable Mean Field Games](https://arxiv.org/abs/2602.20141)** (arXiv:2602.20141) | **Foundation for structural policy-gradient methods** under **recurrent structural policy gradient (RSPG)** and related algorithms (see that paper for MFAX / PO-MFG); DSPG inherits this policy-gradient-through-simulation paradigm in a distribution-based form. |

When you cite prior structural RL or structural policy-gradient ideas alongside DSPG, point readers to **SRL** first for the HA macro setup, and to **RSPG** where comparisons or recurrent / gradient formalism matter.

---

All Python modules live under [`dspg/`](dspg/); notebooks are in [`dspg/notebooks/`](dspg/notebooks/). **Extended docs** (architecture, per-module reference, notebooks and artifacts): see [`docs/README.md`](docs/README.md).

| Layer | Contents |
|--------|-----------|
| **Partial equilibrium (PE)** | [`dspg/pe_rl_env.py`](dspg/pe_rl_env.py), [`dspg/pe_dspg.py`](dspg/pe_dspg.py), baselines [`dspg/pe_ppo.py`](dspg/pe_ppo.py), [`dspg/pe_sac.py`](dspg/pe_sac.py), [`dspg/pe_ddpg.py`](dspg/pe_ddpg.py), [`dspg/pe_vfi.py`](dspg/pe_vfi.py) |
| **General equilibrium (GE)** | Huggett-style bond economy with **bond supply \(B\)** and **market-clearing interest rate \(r\)** — [`dspg/notebooks/main.ipynb`](dspg/notebooks/main.ipynb), [`dspg/ablation_study.py`](dspg/ablation_study.py) |
| **Paper figures & tables** | [`figures_tables/`](figures_tables/) — PDFs + `.tex` snippets as in the DSPG paper |

## Requirements

- Python 3.10+ recommended.
- **JAX** with CUDA for GPU training (install the wheel matching your CUDA toolkit; see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html)).
- Other Python packages are listed in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
pip install "jax[cuda12]"   # example; pick the JAX extras that match your CUDA version
```

Run scripts from the **repository root** so `results/` and `figures_tables/` resolve correctly.

---

## Quick start — partial equilibrium (PE)

PE fixes **\((r,w)\)** exogenously (Markov over grids in [`PEEnv`](dspg/pe_rl_env.py)); use this block for [`pe_*`](dspg/) experiments and RL baselines on the PE environment.

1. **VFI** (generates `results/pe_vfi.npz` — bounds / ground truth for DSPG and baselines):

   ```bash
   python -m dspg.pe_vfi --cuda 0
   ```

2. **DSPG** on PE:

   ```bash
   python -m dspg.pe_dspg --cuda 0
   ```

   Outputs use the `pe_dspg_*` prefix under [`results/`](results/).

3. **Baselines** (examples):

   ```bash
   python -m dspg.pe_ppo --cuda 0
   python -m dspg.pe_sac --cuda 0
   python -m dspg.pe_ddpg --cuda 0
   ```

`--cuda` sets `CUDA_VISIBLE_DEVICES` to that GPU index.

---

## Quick start — general equilibrium (GE, “full” Huggett environment)

GE solves for the **equilibrium interest rate** each period via **bond market clearing** (total bond supply **\(B\)**), with **productivity \(z\)** following a Markov process — the same qualitative block as the Huggett illustration in the SRL paper ([arXiv:2512.18892](https://arxiv.org/abs/2512.18892)). In this repository you can run it in two ways:

### 1. Notebook (interactive)

Open [`dspg/notebooks/main.ipynb`](dspg/notebooks/main.ipynb), set `CUDA_VISIBLE_DEVICES` in the first code cell if needed, and run all cells **with the repository root as the Jupyter working directory** so paths such as `results/` resolve correctly.

### 2. Script (batch sizes / epochs from CLI)

[`dspg/ablation_study.py`](dspg/ablation_study.py) trains the GE Huggett DSPG setup and writes pickles under `results/`, e.g. `DSPG_bs{batch_size}_lr{lr}_ep{epoch}.pkl`:

```bash
python -m dspg.ablation_study --cuda 0 --batch_size 64 --epoch 1000 --lr 2e-3
```

Optional: [`dspg/notebooks/ablation_study.ipynb`](dspg/notebooks/ablation_study.ipynb) reproduces ablation-style figures using saved `results/DSPG_*.pkl` files.

---

## Plotting (mostly PE comparisons)

- [`dspg/plot_pe_training_comparison.py`](dspg/plot_pe_training_comparison.py): DSPG vs PPO / SAC / DDPG vs VFI on **PE** runs; writes PDFs and LaTeX under `figures_tables/`. Default DSPG glob: `pe_dspg_bs64_*_R10.pkl`; legacy `pe_uspg_*` pickles are detected if present.

- [`dspg/pe_plot.py`](dspg/pe_plot.py): DSPG training curve vs VFI (PE).

```bash
python -m dspg.plot_pe_training_comparison
python -m dspg.pe_plot --pattern 'pe_dspg_bs64_*_R10.pkl'
```

---

## Results directory

The [`results/`](results/) folder is **gitignored**: experiment outputs stay on your machine and are **not pushed to GitHub**. After cloning, create `results/` locally (the scripts write there automatically) or run PE/GE training to regenerate logs, pickles, and `.npz` artifacts.

## License

See [`LICENSE`](LICENSE).
