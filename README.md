# DSPG: Distribution-based Structural Policy Gradient

Code for **DSPG**, a distribution-based extension of structural policy-gradient ideas (building on structural reinforcement learning), applied here to heterogeneous-agent style environments.

This repository includes:

- **Partial equilibrium (PE)** experiments: [`pe_rl_env.py`](pe_rl_env.py), [`pe_dspg.py`](pe_dspg.py), baselines [`pe_ppo.py`](pe_ppo.py), [`pe_sac.py`](pe_sac.py), [`pe_ddpg.py`](pe_ddpg.py), and value-function iteration [`pe_vfi.py`](pe_vfi.py).
- **Notebooks**: [`main.ipynb`](main.ipynb) (DSPG / Huggett-style illustration), [`ablation_study.ipynb`](ablation_study.ipynb), [`validate_ks.ipynb`](validate_ks.ipynb) (Krusell–Smith validation).
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

## Quick start (PE environment)

1. Run VFI to produce `results/pe_vfi.npz` (used as bounds / ground truth for DSPG and some baselines):

   ```bash
   python pe_vfi.py --cuda 0
   ```

2. Train DSPG on the PE environment:

   ```bash
   python pe_dspg.py --cuda 0
   ```

   Outputs include pickles and PDFs under [`results/`](results/) with prefix `pe_dspg_`.

3. Baselines (examples):

   ```bash
   python pe_ppo.py --cuda 0
   python pe_sac.py --cuda 0
   python pe_ddpg.py --cuda 0
   ```

`--cuda` sets `CUDA_VISIBLE_DEVICES` to that GPU index.

## Plotting

- [`plot_pe_training_comparison.py`](plot_pe_training_comparison.py): compares DSPG, PPO, SAC, DDPG, and VFI; writes PDFs and LaTeX table snippets under `figures/`. Default glob for DSPG pickles is `pe_dspg_bs64_*_R10.pkl`; if only legacy `pe_uspg_*` files exist, they are detected automatically with a console note.
- [`pe_plot.py`](pe_plot.py): DSPG-only training curve vs VFI with uncertainty band.

```bash
python plot_pe_training_comparison.py
python pe_plot.py --pattern 'pe_dspg_bs64_*_R10.pkl'
```

## Results directory

[`results/`](results/) may contain logs, pickles (`.pkl`), NumPy archives, and PDFs from training runs. Repositories cloned without large artifacts may need to re-run experiments locally to regenerate curves.

## Structural RL connection

DSPG extends ideas from structural reinforcement learning to operate explicitly on cross-sectional distributions; cite your preferred reference to the prior structural RL work when publishing.

## License

See [`LICENSE`](LICENSE).
