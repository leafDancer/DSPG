# DSPG: Distribution-based Structural Policy Gradient

Code for **DSPG**, a distribution-based extension of structural policy-gradient ideas (building on structural reinforcement learning), applied here to heterogeneous-agent style environments.

All Python modules live under [`dspg/`](dspg/); notebooks are in [`dspg/notebooks/`](dspg/notebooks/).

- **Partial equilibrium (PE)** experiments: [`dspg/pe_rl_env.py`](dspg/pe_rl_env.py), [`dspg/pe_dspg.py`](dspg/pe_dspg.py), baselines [`dspg/pe_ppo.py`](dspg/pe_ppo.py), [`dspg/pe_sac.py`](dspg/pe_sac.py), [`dspg/pe_ddpg.py`](dspg/pe_ddpg.py), and value-function iteration [`dspg/pe_vfi.py`](dspg/pe_vfi.py).
- **Notebooks**: [`dspg/notebooks/main.ipynb`](dspg/notebooks/main.ipynb), [`dspg/notebooks/ablation_study.ipynb`](dspg/notebooks/ablation_study.ipynb).
- **Figures / LaTeX snippets**: [`figures/`](figures/) — tables and hyper-parameter blocks for manuscripts.

Abbreviation **DSPG** stands for **Distribution-based Structural Policy Gradient** (emphasizes cross-sectional distributions over agents, not “distributional RL” over return distributions).

## Requirements

- Python 3.10+ recommended.
- **JAX** with CUDA for GPU training (install the wheel matching your CUDA toolkit; see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html)).
- Other Python packages are listed in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
pip install "jax[cuda12]"   # example; pick the JAX extras that match your CUDA version
```

Run scripts from the **repository root** so `results/` and `figures/` resolve correctly.

## Quick start (PE environment)

1. Run VFI to produce `results/pe_vfi.npz`:

   ```bash
   python -m dspg.pe_vfi --cuda 0
   ```

2. Train DSPG on the PE environment:

   ```bash
   python -m dspg.pe_dspg --cuda 0
   ```

   Outputs include pickles and PDFs under [`results/`](results/) with prefix `pe_dspg_`.

3. Baselines (examples):

   ```bash
   python -m dspg.pe_ppo --cuda 0
   python -m dspg.pe_sac --cuda 0
   python -m dspg.pe_ddpg --cuda 0
   ```

`--cuda` sets `CUDA_VISIBLE_DEVICES` to that GPU index.

## Plotting

- [`dspg/plot_pe_training_comparison.py`](dspg/plot_pe_training_comparison.py): compares DSPG, PPO, SAC, DDPG, and VFI; writes PDFs and LaTeX snippets under `figures/`. Default glob for DSPG pickles is `pe_dspg_bs64_*_R10.pkl`; legacy `pe_uspg_*` files are detected automatically if present.

- [`dspg/pe_plot.py`](dspg/pe_plot.py): DSPG-only training curve vs VFI with uncertainty band.

```bash
python -m dspg.plot_pe_training_comparison
python -m dspg.pe_plot --pattern 'pe_dspg_bs64_*_R10.pkl'
```

## Results directory

[`results/`](results/) may contain logs, pickles (`.pkl`), NumPy archives, and PDFs from training runs. Repositories cloned without large artifacts may need to re-run experiments locally to regenerate curves.

## Structural RL connection

DSPG extends ideas from structural reinforcement learning to operate explicitly on cross-sectional distributions; cite your preferred reference to the prior structural RL work when publishing.

## License

See [`LICENSE`](LICENSE).
