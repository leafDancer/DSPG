"""
Value function iteration for the partial-equilibrium environment in pe_rl_env.py.
State: (a, e_idx, r_idx, w_idx); wealth = (1+r)*a + e*w; matches PEEnv.step dynamics.
CLI defaults align with ablation_study.py (CUDA device, etc.).
"""
import argparse
import os
import pickle
from pathlib import Path

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
import numpy as np

from pe_rl_env import PEEnv

# --- grids (align with Huggett VFI style: geometric asset grid) ---
NA = 200
N_MC_PATHS = 4096
N_C_GRID = 4096  # consumption grid points per state (accuracy vs speed)
VFI_TOL = 1e-7
VFI_MAX_ITER = 5000
# Monte Carlo: burn-in so the state is near ergodic before measuring discounted utility over T.
DEFAULT_MC_BURN_IN = 8000
# Print mean wealth along eval horizon every N steps (plus pre / post summary).
MC_WEALTH_DIAG_EVERY = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="5", help="CUDA device id (same default as ablation_study)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--burn_in",
        type=int,
        default=DEFAULT_MC_BURN_IN,
        help="MC paths: discard this many steps (ergodic warm-up) before summing utility over T.",
    )
    args = parser.parse_args()
    cuda_id = str(args.cuda).strip()
    if not cuda_id:
        raise ValueError("Set GPU id via --cuda (e.g. --cuda 0 or --cuda 5).")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    jax.config.update("jax_enable_x64", True)

    env = PEEnv()
    sim_horizon = int(env.T)
    burn_in = int(args.burn_in)
    if burn_in < 0:
        raise ValueError("--burn_in must be >= 0")
    seed = int(args.seed)
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

    a_grid = jnp.geomspace(0.25, a_max - a_min, NA) + a_min - 0.25

    @jax.jit
    def utility(c):
        return jax.lax.cond(
            jnp.abs(sigma - 1.0) < 1e-12,
            lambda: jnp.log(c),
            lambda: c ** (1.0 - sigma) / (1.0 - sigma),
        )

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
    def bellman_max_at_state(aidx, eidx, ridx, widx, V):
        a = a_grid[aidx]
        r = r_grid[ridx]
        e = e_grid[eidx]
        w = w_grid[widx]
        wealth = (1.0 + r) * a + e * w
        p_erf = (
            e_trans[eidx, :, None, None]
            * r_trans[ridx, None, :, None]
            * w_trans[widx, None, None, :]
        )

        c_low = min_c
        c_high = wealth - min_c
        c_high = jnp.maximum(c_high, c_low)
        cgrid = jnp.linspace(c_low, c_high, N_C_GRID)

        def q_for_c(c):
            next_a = wealth - c
            y0, y1, w0a, w1a = iterpolate_nonuniform(next_a, a_grid)
            v_mix = w0a * V[y0, :, :, :] + w1a * V[y1, :, :, :]
            ev = jnp.sum(p_erf * v_mix)
            return utility(c) + beta * ev

        qs = jax.vmap(q_for_c)(cgrid)
        best_q = jnp.max(qs)
        best_c = cgrid[jnp.argmax(qs)]
        return best_q, best_c

    aid = jnp.arange(NA)[:, None, None, None]
    eid = jnp.arange(ne)[None, :, None, None]
    rid = jnp.arange(nr)[None, None, :, None]
    wid = jnp.arange(nw)[None, None, None, :]
    A = jnp.broadcast_to(aid, (NA, ne, nr, nw))
    E = jnp.broadcast_to(eid, (NA, ne, nr, nw))
    R = jnp.broadcast_to(rid, (NA, ne, nr, nw))
    W = jnp.broadcast_to(wid, (NA, ne, nr, nw))

    @jax.jit
    def vfi_step(V):
        v_flat, c_flat = jax.vmap(
            lambda ai, ei, ri, wi: bellman_max_at_state(ai, ei, ri, wi, V)
        )(A.ravel(), E.ravel(), R.ravel(), W.ravel())
        return v_flat.reshape(NA, ne, nr, nw), c_flat.reshape(NA, ne, nr, nw)

    @jax.jit
    def run_vfi():
        """Single compiled loop: avoids one Python↔device sync per iteration."""

        def cond(carry):
            V, i, diff, _ = carry
            return jnp.logical_and(diff >= VFI_TOL, i < VFI_MAX_ITER)

        def body(carry):
            V, i, _, _oc = carry
            V_new, oc_new = vfi_step(V)
            diff = jnp.max(jnp.abs(V_new - V))
            return V_new, i + 1, diff, oc_new

        V0 = jnp.zeros((NA, ne, nr, nw), dtype=jnp.float64)
        oc0 = jnp.zeros((NA, ne, nr, nw), dtype=jnp.float64)
        init = (V0, jnp.int32(0), jnp.array(jnp.inf, jnp.float64), oc0)
        Vf, n_iter, diff, opt_c = jax.lax.while_loop(cond, body, init)
        return Vf, opt_c, n_iter, diff

    print("Compiling and running VFI (JAX while_loop)...", flush=True)
    V, opt_c, n_iter, diff = run_vfi()

    # --- policy interpolation in a (continuous asset) ---
    @jax.jit
    def policy_c(a, eidx, ridx, widx, oc):
        y0, y1, w0a, w1a = iterpolate_nonuniform(a, a_grid)
        eidx = jnp.int32(eidx)
        ridx = jnp.int32(ridx)
        widx = jnp.int32(widx)
        c0 = oc[y0, eidx, ridx, widx]
        c1 = oc[y1, eidx, ridx, widx]
        return w0a * c0 + w1a * c1

    @jax.jit
    def simulate_discounted_utility(key, oc):
        """Burn-in: full Markov on (a,e,r,w). Eval window T: e Markov, r and w IID uniform each step.
        Returns u_sum, a_post, e_post (for ergodic_g), wealth_pre (scalar, end of burn-in),
        wealth_eval_steps (T,) = cash-on-hand (1+r)a+ew at each eval step before c.
        """
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        a = jax.random.uniform(k1, (), minval=a_min, maxval=a_max)
        eidx = jax.random.choice(k2, ne)
        ridx = jax.random.choice(k3, nr)
        widx = jax.random.choice(k4, nw)

        def step_burnin(carry, _):
            a, eidx, ridx, widx, key = carry
            r = r_grid[ridx]
            e = e_grid[eidx]
            w = w_grid[widx]
            wealth = (1.0 + r) * a + e * w
            c = policy_c(a, eidx, ridx, widx, oc)
            c = jnp.clip(c, min_c, wealth - min_c)
            u = utility(c)
            next_a = wealth - c
            key, k1, k2, k3 = jax.random.split(key, 4)
            eidx = jax.random.choice(k1, ne, p=e_trans[eidx])
            ridx = jax.random.choice(k2, nr, p=r_trans[ridx])
            widx = jax.random.choice(k3, nw, p=w_trans[widx])
            return (next_a, eidx, ridx, widx, key), u

        def step_eval(carry, _):
            a, eidx, key = carry
            key, k_e, k_r, k_w = jax.random.split(key, 4)
            ridx_s = jax.random.choice(k_r, nr)
            widx_s = jax.random.choice(k_w, nw)
            r = r_grid[ridx_s]
            e = e_grid[eidx]
            w = w_grid[widx_s]
            wealth = (1.0 + r) * a + e * w
            c = policy_c(a, eidx, ridx_s, widx_s, oc)
            c = jnp.clip(c, min_c, wealth - min_c)
            u = utility(c)
            next_a = wealth - c
            eidx = jax.random.choice(k_e, ne, p=e_trans[eidx])
            return (next_a, eidx, key), (u, wealth)

        key, sub = jax.random.split(key)
        init = (a, eidx, ridx, widx, sub)
        if burn_in > 0:
            init, _ = jax.lax.scan(step_burnin, init, None, length=burn_in)
        a_post, e_post, r_post, w_post, key_e = init
        wealth_pre = (1.0 + r_grid[r_post]) * a_post + e_grid[e_post] * w_grid[w_post]
        init_eval = (a_post, e_post, key_e)
        _, (utilities, wealth_steps) = jax.lax.scan(
            step_eval, init_eval, None, length=sim_horizon
        )
        discounts = jnp.power(beta, jnp.arange(sim_horizon, dtype=jnp.float64))
        u_sum = jnp.sum(utilities * discounts)
        return (
            u_sum,
            a_post,
            jnp.asarray(e_post, dtype=jnp.int32),
            wealth_pre,
            wealth_steps,
        )

    @jax.jit
    def ergodic_g_histogram(a_batch, e_batch):
        """Map each MC agent's (a,e) to (NA, ne) with linear weights on a_grid (same as DSPG update_g)."""
        n = a_batch.shape[0]

        def weights(a):
            y0, y1, w0, w1 = iterpolate_nonuniform(a, a_grid)
            return y0, y1, w0, w1

        y0, y1, w0, w1 = jax.vmap(weights)(a_batch)
        scale = 1.0 / jnp.maximum(n, 1)
        flat0 = y0 * ne + e_batch
        flat1 = y1 * ne + e_batch
        g_flat = jnp.zeros(NA * ne, dtype=jnp.float64)
        g_flat = g_flat.at[flat0].add(scale * w0)
        g_flat = g_flat.at[flat1].add(scale * w1)
        g = g_flat.reshape(NA, ne)
        return g / (jnp.sum(g) + 1e-20)

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, N_MC_PATHS)
    sim_us, term_a, term_e, wealth_pre_paths, wealth_traj_paths = jax.vmap(
        lambda k: simulate_discounted_utility(k, opt_c)
    )(keys)
    mean_u = float(jnp.mean(sim_us))
    std_u = float(jnp.std(sim_us))
    g_ergodic = np.asarray(ergodic_g_histogram(term_a, term_e))

    mean_wealth_pre = float(jnp.mean(wealth_pre_paths))
    mean_wealth_traj = np.asarray(jnp.mean(wealth_traj_paths, axis=0), dtype=np.float64)
    mean_wealth_post = float(mean_wealth_traj[sim_horizon - 1])
    gap_pre_post = mean_wealth_post - mean_wealth_pre

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "pe_vfi.npz"
    jnp.savez(
        npz_path,
        a_grid=np.asarray(a_grid),
        opt_c=np.asarray(opt_c),
        V=np.asarray(V),
        e_grid=np.asarray(e_grid),
        r_grid=np.asarray(r_grid),
        w_grid=np.asarray(w_grid),
        e_trans=np.asarray(e_trans),
        r_trans=np.asarray(r_trans),
        w_trans=np.asarray(w_trans),
        beta=beta,
        min_c=min_c,
        sigma=sigma,
        n_iter=int(n_iter),
        diff=float(diff),
        sim_horizon=sim_horizon,
        mc_burn_in=burn_in,
        mc_eval_rw_uniform=np.array(True),
        n_mc_paths=N_MC_PATHS,
        ergodic_g=g_ergodic,
        mc_mean_wealth_pre=mean_wealth_pre,
        mc_mean_wealth_post=mean_wealth_post,
        mc_mean_wealth_pre_post_gap=gap_pre_post,
        mc_mean_wealth_eval_trajectory=mean_wealth_traj,
        mean_discounted_utility=mean_u,
        std_discounted_utility=std_u,
    )

    meta = {
        "na": NA,
        "ne": ne,
        "nr": nr,
        "nw": nw,
        "n_iter": int(n_iter),
        "final_sup_diff": float(diff),
        "sim_horizon": sim_horizon,
        "mc_burn_in": burn_in,
        "mc_eval_rw_uniform": True,
        "n_mc_paths": N_MC_PATHS,
        "ergodic_g_shape": list(g_ergodic.shape),
        "ergodic_g_sum": float(np.sum(g_ergodic)),
        "mc_mean_wealth_pre": mean_wealth_pre,
        "mc_mean_wealth_post": mean_wealth_post,
        "mc_mean_wealth_pre_post_gap": gap_pre_post,
        "mean_discounted_utility": mean_u,
        "std_discounted_utility": std_u,
        "npz": str(npz_path),
    }
    with open(out_dir / "pe_vfi_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("PE VFI finished.")
    print(f"  iterations: {int(n_iter)}, sup|V'-V|: {float(diff):.3e}")
    print(
        f"  Monte Carlo mean of sum_{{t=0}}^{{{sim_horizon - 1}}} beta^t u(c_t) "
        f"({N_MC_PATHS} paths; burn-in {burn_in} full-Markov; eval {sim_horizon} steps with "
        f"e Markov, r & w IID uniform): {mean_u:.6f} (std {std_u:.6f})"
    )
    print(
        f"  Mean wealth (cash-on-hand before c): pre (end burn-in)={mean_wealth_pre:.6f}, "
        f"post (after T eval)={mean_wealth_post:.6f}, gap={gap_pre_post:+.6f}",
        flush=True,
    )
    ev = MC_WEALTH_DIAG_EVERY
    print(
        f"  Mean wealth every {ev} eval steps (MC avg over {N_MC_PATHS} paths):",
        flush=True,
    )
    for s in range(ev, sim_horizon + 1, ev):
        print(f"    eval step {s:4d}: {mean_wealth_traj[s - 1]:.6f}", flush=True)
    if sim_horizon % ev != 0:
        print(
            f"    eval step {sim_horizon:4d} (post): {mean_wealth_post:.6f}",
            flush=True,
        )
    print(
        f"  Saved ergodic_g on (NA, ne)=({NA}, {ne}) from post-burn-in (a,e) "
        f"(mass sum={float(np.sum(g_ergodic)):.6f}) for DSPG warmup.",
        flush=True,
    )


if __name__ == "__main__":
    main()
