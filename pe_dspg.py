"""
DSPG (distribution-based structural policy gradient) for the partial-equilibrium environment (pe_rl_env.py).
Distribution g: (B, na, ne). r and w are sampled each period (Markov); policy c is (B, na, ne) with
VFI bounds from mean(opt_c) over (r,w) (IID / no cross-section over r,w in the network head).

Training still uses adaptive g0 (ergodic during warmup, then previous epoch's final g). The saved
training curve uses a separate evaluation (same T-step discounted U under fixed ergodic g) only
every ``eval_every`` epochs (plus the first and last epoch); intervening entries are forward-filled
from the last evaluation so array shape stays (repeats, epochs).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

# JAX: use CUDA unless the user already set JAX_PLATFORMS (e.g. cpu for debugging).
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import optax

from pe_rl_env import PEEnv

jax.config.update("jax_enable_x64", True)

# Defaults: batch/epochs/warmup/CUDA follow ablation_study.py; PE DSPG uses INIT_LR=5e-4 (script default 2e-3).
NA = 200
INIT_LR = 2e-3
WARMUP_EPOCHS = 50
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH = 64
DEFAULT_CUDA = "5"
DEFAULT_EVAL_EVERY = 100
VFI_NPZ_NAME = "pe_vfi.npz"


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
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "Computer Modern Roman",
                "serif",
            ],
        }
    )


def plot_training_curves(
    rewards: np.ndarray,
    vfi_gt: float,
    out_path: Path,
    title: str = "DSPG training (PE environment)",
):
    set_static_styles()
    n_rep, n_ep = rewards.shape
    epochs = np.arange(1, n_ep + 1, dtype=np.float64)
    mean_u = np.nanmean(rewards, axis=0)
    std_u = np.nanstd(rewards, axis=0, ddof=0)
    lo = mean_u - 1.96 * std_u
    hi = mean_u + 1.96 * std_u

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.fill_between(epochs, lo, hi, color="red", alpha=0.3, label="95% band (±1.96 std across runs)")
    ax.plot(epochs, mean_u, color="red", linewidth=2, label="DSPG (mean across runs)")
    ax.axhline(y=vfi_gt, color="gray", linestyle="--", linewidth=2, label="VFI ground truth")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Disc. utility, fixed ergodic g₀ eval (mean over batch)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_train_step_profile(
    train_step,
    update_g,
    forward,
    params,
    opt_state,
    g0,
    key,
    *,
    batch_size: int,
    na: int,
    ne: int,
    horizon_t: int,
) -> None:
    print("=== PE DSPG timing profile ===", flush=True)
    print(
        f"  batch_size={batch_size}, NA={na}, ne={ne}, scan length T={horizon_t}; "
        f"g shape (B, na, ne)",
        flush=True,
    )

    t0 = time.perf_counter()
    out = train_step(params, opt_state, key, g0)
    jax.block_until_ready(out[2])
    print(
        f"  train_step, 1st call (compile + run): {(time.perf_counter() - t0) * 1000:.0f} ms",
        flush=True,
    )

    for i in range(3):
        t0 = time.perf_counter()
        out = train_step(params, opt_state, key, g0)
        jax.block_until_ready(out[2])
        print(
            f"  train_step, warm #{i + 1}: {(time.perf_counter() - t0) * 1000:.1f} ms",
            flush=True,
        )

    fwd_j = jax.jit(forward.apply)
    t0 = time.perf_counter()
    jax.block_until_ready(fwd_j(params, g0))
    print(
        f"  forward.apply, 1st call (compile + run): {(time.perf_counter() - t0) * 1000:.0f} ms",
        flush=True,
    )
    for i in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(fwd_j(params, g0))
        print(
            f"  forward.apply, warm #{i + 1}: {(time.perf_counter() - t0) * 1000:.1f} ms",
            flush=True,
        )

    ug = jax.jit(update_g)
    rng = jax.random.PRNGKey(0)
    wn = jax.random.uniform(rng, (batch_size, na, ne), minval=0.0, maxval=150.0)
    t0 = time.perf_counter()
    jax.block_until_ready(ug(wn, g0))
    print(
        f"  update_g, 1st call (compile + run): {(time.perf_counter() - t0) * 1000:.0f} ms",
        flush=True,
    )
    for i in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(ug(wn, g0))
        print(
            f"  update_g, warm #{i + 1}: {(time.perf_counter() - t0) * 1000:.1f} ms",
            flush=True,
        )

    print(
        "  Note: train_step cost is dominated by reverse-mode AD through scan(T); "
        "if 1st call >> warm calls, the gap is mostly XLA compile.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cuda", type=str, default=DEFAULT_CUDA)
    parser.add_argument("--lr", type=float, default=INIT_LR)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon for DSPG; default is PEEnv.T (same truncation as pe_rl_env).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
        help="Run fixed-g curve eval every N epochs (plus first and last epoch).",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print train_step / forward / update_g timings and exit (no training).",
    )
    args = parser.parse_args()

    cuda_id = str(args.cuda).strip()
    if not cuda_id:
        raise ValueError("Set GPU id via --cuda (e.g. --cuda 0 or --cuda 5).")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    env = PEEnv()
    a_min = float(env.a_min)
    a_max = float(env.a_max)
    beta = float(env.beta)
    min_c = float(env.c_min)
    sigma = float(env.sigma)

    ne, nr, nw = env.ne, env.nr, env.nw
    e_grid = jnp.asarray(env.e_grid, dtype=jnp.float64)
    r_grid = jnp.asarray(env.r_grid, dtype=jnp.float64)
    w_grid = jnp.asarray(env.w_grid, dtype=jnp.float64)
    e_trans = jnp.asarray(env.e_trans, dtype=jnp.float64)
    r_trans = jnp.asarray(env.r_trans, dtype=jnp.float64)
    w_trans = jnp.asarray(env.w_trans, dtype=jnp.float64)

    T = int(args.horizon) if args.horizon is not None else int(env.T)
    print(
        f"DSPG rollout length T={T} (default PEEnv.T={int(env.T)}; override with --horizon).",
        flush=True,
    )

    a_grid = jnp.geomspace(0.25, a_max - a_min, NA) + a_min - 0.25

    results_dir = Path(__file__).resolve().parent / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    vfi_path = results_dir / VFI_NPZ_NAME
    if not vfi_path.is_file():
        raise FileNotFoundError(
            f"Missing {vfi_path}. Run `python pe_vfi.py` first to generate VFI bounds and ground-truth utility."
        )
    vfi_npz = dict(np.load(vfi_path, allow_pickle=False))
    opt_c = jnp.asarray(vfi_npz["opt_c"], dtype=jnp.float64)
    if "mean_discounted_utility" in vfi_npz:
        vfi_gt = float(np.asarray(vfi_npz["mean_discounted_utility"]).reshape(()))
    else:
        vfi_gt = float(np.asarray(vfi_npz["mean_discounted_utility_1000"]).reshape(()))

    if opt_c.shape != (NA, ne, nr, nw):
        raise ValueError(f"opt_c shape {opt_c.shape} != ({NA}, {ne}, {nr}, {nw})")

    # Marginal reference policy over (r,w): mean for IID / no nr*nw output head
    opt_c_marg = jnp.mean(opt_c, axis=(2, 3))
    diffs = jnp.diff(opt_c_marg, axis=0)
    max_diff = jnp.zeros((NA, ne), dtype=jnp.float64)
    max_diff = max_diff.at[0, :].set(diffs[0, :])
    max_diff = max_diff.at[1:, :].set(diffs)
    max_diff *= 2.0
    e_off = jnp.asarray([0.05, 0.02, 0.1], dtype=jnp.float64)[:ne]
    csmp0 = jnp.maximum(opt_c_marg[0, :] - e_off, min_c)

    if "ergodic_g" in vfi_npz:
        g_erg_2d = np.asarray(vfi_npz["ergodic_g"], dtype=np.float64)
        if g_erg_2d.shape != (NA, ne):
            raise ValueError(
                f"ergodic_g in pe_vfi.npz has shape {g_erg_2d.shape}, expected ({NA}, {ne})"
            )
        g_erg_j = jnp.asarray(g_erg_2d)
        g_erg_j = g_erg_j / jnp.sum(g_erg_j)
    else:
        g_erg_j = None

    batch_size = args.batch_size
    n_epochs = args.epoch
    n_repeats = args.repeats
    init_lr = args.lr
    eval_every = args.eval_every
    if eval_every <= 0:
        eval_every = 1

    activation = jax.nn.leaky_relu
    pool = hk.AvgPool

    class DoubleConv(hk.Module):
        def __init__(self, out_channels, name=None):
            super().__init__(name=name)
            self.out_channels = out_channels

        def __call__(self, x):
            x = hk.Conv1D(
                output_channels=self.out_channels, kernel_shape=3, stride=1, padding="SAME"
            )(x)
            x = activation(x)
            x = hk.Conv1D(
                output_channels=self.out_channels, kernel_shape=3, stride=1, padding="SAME"
            )(x)
            x = activation(x)
            return x

    def forward_fn(g):
        # Same U-Net as ablation_study.forward_fn; head is (na, ne) only (r,w enter via wealth).
        g = g / jnp.sum(g, axis=(1, 2), keepdims=True)
        g = g * NA * ne
        x1 = DoubleConv(4)(g)
        x2 = pool(window_shape=2, strides=2, padding="SAME")(x1)
        x2 = DoubleConv(4)(x2)
        x3 = pool(window_shape=2, strides=2, padding="SAME")(x2)
        x3 = DoubleConv(4)(x3)
        y = x3
        y = hk.Conv1DTranspose(output_channels=4, kernel_shape=3, stride=2, padding="SAME")(y)
        y = y[:, : x2.shape[1], :]
        y = jnp.concatenate([y, x2], axis=-1)
        y = DoubleConv(4)(y)
        y = hk.Conv1DTranspose(output_channels=4, kernel_shape=3, stride=2, padding="SAME")(y)
        y = y[:, : x1.shape[1], :]
        y = jnp.concatenate([y, x1], axis=-1)
        y = DoubleConv(4)(y)
        y = hk.Conv1D(output_channels=ne, kernel_shape=1, stride=1, padding="SAME")(y)
        y = jax.nn.sigmoid(y)
        bsz = y.shape[0]
        c = jnp.zeros((bsz, NA + 1, ne))
        c = c.at[:, 0, :].set(csmp0[None, :] * jnp.ones((bsz, ne)))
        c = c.at[:, 1:, :].set(y * max_diff[None, ...])
        c = jnp.cumsum(c, axis=1)
        c = c[:, 1:, :]
        return c

    @jax.jit
    def iterpolate_nonuniform(x, grids):
        x = jnp.clip(x, grids[0], grids[-1])
        idx = jnp.searchsorted(grids, x) - 1
        idx = jnp.clip(idx, 0, len(grids) - 2)
        x0, x1 = grids[idx], grids[idx + 1]
        y0, y1 = idx.astype(jnp.int32), (idx + 1).astype(jnp.int32)
        w0 = (x1 - x) / (x1 - x0 + 1e-12)
        w1 = 1.0 - w0
        return y0, y1, w0, w1

    @jax.jit
    def utility(c):
        return jax.lax.cond(
            jnp.abs(sigma - 1.0) < 1e-12,
            lambda: jnp.log(c),
            lambda: c ** (1.0 - sigma) / (1.0 - sigma),
        )

    @jax.jit
    def update_g(wealth_next_batch, g_batch):
        """Same as ablation_study.update_g (mass over (a, e) only)."""

        def loop_B(wealth_next, g):
            def loop_a(aidx):
                def loop_e(eidx):
                    next_a = wealth_next[aidx, eidx]
                    y0, y1, w0, w1 = iterpolate_nonuniform(next_a, a_grid)
                    g_new = jnp.zeros((NA, ne))
                    enext_prob = e_trans[eidx]
                    g_new = g_new.at[y0].set(g[aidx, eidx] * w0 * enext_prob)
                    g_new = g_new.at[y1].set(g[aidx, eidx] * w1 * enext_prob)
                    return g_new

                return jax.vmap(loop_e)(jnp.arange(ne))

            return jax.vmap(loop_a)(jnp.arange(NA))

        return jax.vmap(loop_B)(wealth_next_batch, g_batch).sum(axis=(1, 2))

    all_rewards = np.full((n_repeats, n_epochs), np.nan, dtype=np.float64)

    def curve_eval_this_epoch(ep: int) -> bool:
        """True on first epoch, every eval_every-th epoch (1-based), and last epoch."""
        if ep == 0 or ep == n_epochs - 1:
            return True
        return (ep + 1) % eval_every == 0

    for rep in range(n_repeats):
        print(f"=== DSPG repeat {rep + 1}/{n_repeats} ===", flush=True)
        forward = hk.without_apply_rng(hk.transform(forward_fn))
        dummy_g = jnp.ones((2, NA, ne))
        rep_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), rep)
        k_init, key = jax.random.split(rep_key)
        params = forward.init(k_init, dummy_g)
        schedule = optax.exponential_decay(init_lr, n_epochs, 0.5)
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)

        def mean_wealth_under_g(g_b, ridx_b, widx_b):
            """E[wealth | r,w] under cross-section g_b; mean over batch rows."""
            r = r_grid[ridx_b][:, None, None]
            wv = w_grid[widx_b][:, None, None]
            wealth = (1.0 + r) * a_grid[None, :, None] + e_grid[None, None, :] * wv
            per_row = jnp.sum(g_b * wealth, axis=(1, 2))
            return jnp.mean(per_row)

        def forward_rollout(p, key_loss, g_init):
            key_a, key_b, key_c = jax.random.split(key_loss, 3)
            ridx0 = jax.random.choice(key_a, nr, (batch_size,))
            widx0 = jax.random.choice(key_b, nw, (batch_size,))
            discount0 = jnp.ones((batch_size,), dtype=jnp.float64)

            def step(pack, _):
                g_b, ridx_b, widx_b, discount, key_s = pack
                c = forward.apply(p, g_b)

                r = r_grid[ridx_b][:, None, None]
                wv = w_grid[widx_b][:, None, None]
                wealth = (1.0 + r) * a_grid[None, :, None] + e_grid[None, None, :] * wv

                c = jnp.minimum(c, wealth - a_min - min_c)
                wealth_next = wealth - c
                g_b = update_g(wealth_next, g_b)
                util = jnp.sum(utility(c) * g_b, axis=(1, 2)) * discount
                discount = discount * beta

                key_s, kr, kw = jax.random.split(key_s, 3)
                keys_r = jax.random.split(kr, batch_size)
                keys_w = jax.random.split(kw, batch_size)

                def step_r(i, k):
                    return jax.random.choice(k, nr, p=r_trans[i])

                def step_w(i, k):
                    return jax.random.choice(k, nw, p=w_trans[i])

                ridx_b = jax.vmap(step_r)(ridx_b, keys_r)
                widx_b = jax.vmap(step_w)(widx_b, keys_w)
                return (g_b, ridx_b, widx_b, discount, key_s), util

            (g_fin, rT, wT, _, _), utils = jax.lax.scan(
                step,
                (g_init, ridx0, widx0, discount0, key_c),
                None,
                length=T,
            )
            total_u = jnp.sum(utils, axis=0).mean()
            return total_u, g_fin, ridx0, widx0, rT, wT

        @jax.jit
        def train_step(p, ost, key_ts, g0):
            def lfn(pp):
                tu, gf, r0, w0, r_end, w_end = forward_rollout(pp, key_ts, g0)
                w_pre = mean_wealth_under_g(g0, r0, w0)
                w_post = mean_wealth_under_g(gf, r_end, w_end)
                return -tu, (gf, w_pre, w_post)

            (loss, aux), grads = jax.value_and_grad(lfn, has_aux=True)(p)
            final_g, w_pre, w_post = aux
            upd, ost = optimizer.update(grads, ost)
            p = optax.apply_updates(p, upd)
            return p, ost, loss, final_g, w_pre, w_post

        t0 = time.time()
        if g_erg_j is not None:
            g_warmup = jnp.broadcast_to(g_erg_j[None, :, :], (batch_size, NA, ne))
        else:
            g_warmup = jnp.ones((batch_size, NA, ne)) / (NA * ne)
            print(
                "Warning: pe_vfi.npz has no ergodic_g; using uniform g for warmup and curve eval. Re-run pe_vfi.py with ergodic_g.",
                flush=True,
            )
        final_g = g_warmup

        @jax.jit
        def eval_training_curve_metric(p, key_e):
            """T-step mean discounted U with fixed g = ergodic (g_warmup); independent of train g0."""
            tu, gf, r0, w0, rT, wT = forward_rollout(p, key_e, g_warmup)
            w_pre = mean_wealth_under_g(g_warmup, r0, w0)
            w_post = mean_wealth_under_g(gf, rT, wT)
            return tu, w_pre, w_post

        if args.profile:
            run_train_step_profile(
                train_step,
                update_g,
                forward,
                params,
                opt_state,
                g_warmup,
                key,
                batch_size=batch_size,
                na=NA,
                ne=ne,
                horizon_t=T,
            )
            return

        u_carry = np.nan
        progress_every = max(1, n_epochs // 10)
        for ep in range(n_epochs):
            eval_now = curve_eval_this_epoch(ep)
            if eval_now:
                key, sk_train, sk_curve = jax.random.split(key, 3)
            else:
                key, sk_train = jax.random.split(key, 2)
            if ep < WARMUP_EPOCHS:
                g0 = g_warmup
            else:
                g0 = final_g
            params, opt_state, loss, final_g, _, _ = train_step(
                params, opt_state, sk_train, g0
            )
            if eval_now:
                u_curve, w_pre_c, w_post_c = eval_training_curve_metric(params, sk_curve)
                u_carry = float(u_curve)
            all_rewards[rep, ep] = u_carry
            if eval_now:
                print(
                    f"  epoch {ep + 1}/{n_epochs}  [curve eval] disc. U (fixed ergodic g₀, T={T}): {u_carry:.6f}  "
                    f"| wealth pre/post (same eval): {float(w_pre_c):.4f} -> {float(w_post_c):.4f}  "
                    f"| train -loss: {-float(loss):.6f}",
                    flush=True,
                )
            elif (ep + 1) % progress_every == 0:
                print(
                    f"  epoch {ep + 1}/{n_epochs}  train -loss: {-float(loss):.6f}  "
                    f"(curve eval every {eval_every} ep; last curve U {u_carry:.6f})",
                    flush=True,
                )
        print(f"  repeat {rep + 1} wall time: {time.time() - t0:.1f}s", flush=True)

    tag = f"pe_dspg_bs{batch_size}_lr{init_lr:.3E}_ep{n_epochs}_H{T}_R{n_repeats}"
    pkl_path = results_dir / f"{tag}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "cumulative_reward_per_epoch": all_rewards,
                "vfi_ground_truth_utility": vfi_gt,
                "config": {
                    "batch_size": batch_size,
                    "epochs": n_epochs,
                    "repeats": n_repeats,
                    "horizon_T": T,
                    "pe_env_T": int(env.T),
                    "lr": init_lr,
                    "na": NA,
                    "ne": ne,
                    "nr": nr,
                    "nw": nw,
                    "beta": beta,
                    "seed": args.seed,
                    "eval_every_epochs": eval_every,
                    "training_curve_metric": (
                        "After selected epoch updates: mean T-step discounted utility with initial g "
                        "fixed to VFI ergodic_g (broadcast to batch). Eval on epoch 1, every "
                        f"{eval_every} epochs (1-based), and last epoch; other entries forward-filled."
                    ),
                },
            },
            f,
        )
    print(f"Saved training curves and metadata to {pkl_path}", flush=True)

    fig_path = results_dir / f"{tag}_training_curve.pdf"
    if not args.skip_plot:
        plot_training_curves(all_rewards, vfi_gt, fig_path)
        print(f"Saved figure to {fig_path}", flush=True)


if __name__ == "__main__":
    main()
