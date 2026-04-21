"""
Simple PPO on pe_rl_env.PEEnv.

Policy I/O (per user spec):
  - Input: low-dimensional continuous (a, e, r, w) — same 4 numbers as PEEnv._gen_obs
    (assets, productivity level, interest rate, wage factor), not one-hot indices.
  - Output: consumption share cshare in [0, 1], sampled from a Beta head (mean Beta used in post-train eval).

After training, evaluates mean discounted utility over T steps: initial (a,e) from VFI ergodic_g on
(NA, ne); initial (r,w) uniform on discrete states; then the same Markov transitions as PEEnv
(r_trans, w_trans, e_trans). This matches the environment dynamics (not the pe_vfi MC eval window
where r,w are IID uniform).

Training log metric `mean_disc_return_tail` (stored as curve_mean_discounted_return_per_update):
  For each parallel env, backward discounted sum over the *current rollout window* of length H
  (not necessarily a full episode of T; episodes reset on trunc). Initial states come from
  PEEnv.reset(): a ~ Uniform[a_min,a_max], e,r,w each uniform on discrete indices — NOT ergodic_g.
  Printed alongside an ergodic-based eval (fewer MC paths) when --log_ergodic_eval_paths > 0.

**Run:** ``python -m dspg.pe_ppo --cuda 0`` (requires ``results/pe_vfi.npz`` from ``dspg.pe_vfi``).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

from dspg.repo_paths import REPO_ROOT

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import optax

from dspg.pe_rl_env import PEEnv

jax.config.update("jax_enable_x64", True)

VFI_NPZ_NAME = "pe_vfi.npz"
DEFAULT_CUDA = "5"


def set_static_styles():
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        }
    )


def beta_log_prob(x: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    from jax.scipy.special import betaln

    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return (a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log(1.0 - x) - betaln(a, b)


def make_networks(obs_dim: int, hidden: int):
    def actor_critic(obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """obs (B,4) -> Beta alpha, Beta beta (both >0), value (B,)."""
        h = hk.nets.MLP([hidden, hidden], activation=jax.nn.tanh, name="shared")(obs)
        raw = hk.Linear(2, name="actor_head")(h)
        alpha = jax.nn.softplus(raw[:, 0]) + 1.0
        beta_p = jax.nn.softplus(raw[:, 1]) + 1.0
        v = hk.Linear(1, name="critic")(h)[:, 0]
        return alpha, beta_p, v

    return hk.transform(actor_critic)


def normalize_obs(obs: np.ndarray, env: PEEnv) -> np.ndarray:
    """Rough scale to O(1) for (a, e, r, w)."""
    scale = np.asarray(
        [env.a_max - env.a_min + 1e-12, 2.5, 0.035, 0.15], dtype=np.float64
    )
    off = np.asarray([env.a_min, 0.0, env.r_grid[0] - 0.01, env.w_grid[0] - 0.02], dtype=np.float64)
    return (obs - off) / scale


def utility_from_c(c: float, env: PEEnv) -> float:
    if abs(env.sigma - 1.0) < 1e-12:
        return float(np.log(c))
    return float(c ** (1.0 - env.sigma) / (1.0 - env.sigma))


def step_train(state, cshare: float, env: PEEnv, rng: np.random.Generator):
    """One PEEnv transition (Markov r, w). state = (ep, a, eidx, ridx, widx)."""
    ep, a, eidx, ridx, widx = state
    wealth = (1.0 + env.r_grid[ridx]) * a + env.e_grid[eidx] * env.w_grid[widx]
    c = np.clip(wealth * np.clip(cshare, 0.0, 1.0), env.c_min, wealth - env.c_min)
    u = utility_from_c(float(c), env)
    next_a = float(wealth - c)
    next_e = int(rng.choice(env.ne, p=env.e_trans[eidx]))
    next_r = int(rng.choice(env.nr, p=env.r_trans[ridx]))
    next_w = int(rng.choice(env.nw, p=env.w_trans[widx]))
    next_ep = ep + 1
    trunc = next_ep >= env.T
    obs = env._gen_obs(next_a, next_e, next_r, next_w)
    next_state = (next_ep, next_a, next_e, next_r, next_w)
    return next_state, obs, u, trunc


def sample_initial_from_ergodic(
    g: np.ndarray, a_grid: np.ndarray, rng: np.random.Generator
) -> tuple[float, int]:
    """g shape (NA, ne), flat categorical -> continuous a on a_grid, e index."""
    na, ne = g.shape
    flat = g.reshape(-1)
    flat = flat / (np.sum(flat) + 1e-20)
    idx = int(rng.choice(na * ne, p=flat))
    ia = idx // ne
    ie = idx % ne
    return float(a_grid[ia]), ie


def eval_ergodic_markov_prices(
    apply_fn,
    params,
    env: PEEnv,
    g_erg: np.ndarray,
    a_grid: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """(a,e) from VFI ergodic_g; (r,w) Markov via r_trans, w_trans (same as PEEnv.step)."""
    T = int(env.T)
    beta = float(env.beta)
    disc = np.power(beta, np.arange(T, dtype=np.float64))

    @jax.jit
    def policy_mean_cshare(obs_norm: jnp.ndarray) -> jnp.ndarray:
        a, b, _ = apply_fn(params, obs_norm[None, :])
        return a[0] / (a[0] + b[0])

    total = 0.0
    for _ in range(n_paths):
        a0, e0 = sample_initial_from_ergodic(g_erg, a_grid, rng)
        r0 = int(rng.integers(0, env.nr))
        w0 = int(rng.integers(0, env.nw))
        state = (0, a0, e0, r0, w0)
        u_path = np.zeros(T, dtype=np.float64)
        for t in range(T):
            ep, a, eidx, ridx, widx = state
            obs = env._gen_obs(a, eidx, ridx, widx)
            on = normalize_obs(obs.astype(np.float64), env)
            cshare = float(policy_mean_cshare(jnp.asarray(on)))
            state, _, u, trunc = step_train(state, cshare, env, rng)
            u_path[t] = u
            if trunc:
                break
        total += float(np.sum(u_path * disc))
    return total / n_paths


def collect_rollout(
    apply_fn,
    params,
    env: PEEnv,
    n_envs: int,
    horizon: int,
    rng: np.random.Generator,
    jax_key: jax.Array,
) -> dict:
    """Vectorized env rollout; horizon steps (may span trunc). Reset on trunc."""
    obs_l, act_l, rew_l, val_l, logp_l, done_l = [], [], [], [], [], []
    states = []
    obss = []
    for _ in range(n_envs):
        st, obs = env.reset()
        states.append(st)
        obss.append(np.asarray(obs, dtype=np.float64))

    key = jax_key

    for _ in range(horizon):
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, n_envs)
        obs_arr = np.stack(obss, axis=0)
        obs_n = normalize_obs(obs_arr, env)
        obs_j = jnp.asarray(obs_n)

        def sample_one(obs_i, k):
            a, b, v = apply_fn(params, obs_i[None, :])
            a0, b0 = a[0], b[0]
            x = jax.random.beta(k, a0, b0)
            lp = beta_log_prob(x, a0, b0)
            return x, lp, v[0]

        xs, lps, vs = jax.vmap(sample_one)(obs_j, keys)

        xs = np.asarray(xs)
        lps = np.asarray(lps)
        vs = np.asarray(vs)

        obs_l.append(obs_n.copy())
        act_l.append(xs.copy())
        rew_l.append([])
        val_l.append(vs.copy())
        logp_l.append(lps.copy())
        done_l.append([])

        new_obss = []
        rews_t = []
        dones_t = []
        for i in range(n_envs):
            ns, o, u, trunc = step_train(states[i], float(xs[i]), env, rng)
            rews_t.append(u)
            dones_t.append(trunc)
            if trunc:
                ns, o = env.reset()
            states[i] = ns
            new_obss.append(np.asarray(o, dtype=np.float64))
        rew_l[-1] = np.asarray(rews_t)
        done_l[-1] = np.asarray(dones_t, dtype=np.float64)
        obss = new_obss

    return {
        "obs": np.stack(obs_l, axis=0),
        "actions": np.stack(act_l, axis=0),
        "rewards": np.stack(rew_l, axis=0),
        "values": np.stack(val_l, axis=0),
        "log_probs": np.stack(logp_l, axis=0),
        "dones": np.stack(done_l, axis=0),
        "last_obs": normalize_obs(np.stack(obss), env),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """rewards, values, dones: (H, N). last_value: (N,) = V(s_first step after horizon)."""
    H, N = rewards.shape
    adv = np.zeros_like(rewards)
    last_gae = np.zeros(N, dtype=np.float64)
    for t in reversed(range(H)):
        v_sp1 = last_value if t == H - 1 else values[t + 1]
        m = 1.0 - dones[t]
        delta = rewards[t] + gamma * v_sp1 * m - values[t]
        last_gae = delta + gamma * lam * m * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


def beta_entropy(a_p: jnp.ndarray, b_p: jnp.ndarray) -> jnp.ndarray:
    """Mean differential entropy of Beta(a,b)."""
    from jax.scipy.special import betaln, digamma

    apb = a_p + b_p
    return jnp.mean(
        betaln(a_p, b_p)
        - (a_p - 1.0) * digamma(a_p)
        - (b_p - 1.0) * digamma(b_p)
        + (apb - 2.0) * digamma(apb)
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO on PEEnv (Beta policy head).")
    parser.add_argument("--cuda", type=str, default=DEFAULT_CUDA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent runs; each uses seed + repeat_index (separate pkl/pdf).",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--total_updates", type=int, default=500)
    parser.add_argument(
        "--log_every",
        type=int,
        default=None,
        help="Progress print + ergodic eval every N updates (default: total_updates//10).",
    )
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64, help="Steps per rollout (env may reset on trunc).")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=None, help="Default: env.beta")
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument(
        "--eval_paths",
        type=int,
        default=4096,
        help="MC paths for post-train eval (ergodic a,e + Markov r,w).",
    )
    parser.add_argument(
        "--log_ergodic_eval_paths",
        type=int,
        default=256,
        help="Each progress print: also run ergodic (a,e) eval with this many paths (0=skip).",
    )
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    cuda_id = str(args.cuda).strip()
    if not cuda_id:
        raise ValueError("Set GPU id via --cuda, e.g. --cuda 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    env = PEEnv()
    gamma = float(env.beta) if args.gamma is None else float(args.gamma)

    results_dir = REPO_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    vfi_path = results_dir / VFI_NPZ_NAME
    if not vfi_path.is_file():
        raise FileNotFoundError(f"Need {vfi_path} (ergodic_g, a_grid). run python -m dspg.pe_vfi first.")

    vfi = dict(np.load(vfi_path, allow_pickle=False))
    if "ergodic_g" not in vfi or "a_grid" not in vfi:
        raise KeyError("pe_vfi.npz must contain ergodic_g and a_grid")
    g_erg = np.asarray(vfi["ergodic_g"], dtype=np.float64)
    a_grid_vfi = np.asarray(vfi["a_grid"], dtype=np.float64)
    vfi_gt_u = float(np.asarray(vfi["mean_discounted_utility"]).reshape(()))

    obs_dim = 4
    net = make_networks(obs_dim, args.hidden)
    tx = optax.adam(args.lr)

    @jax.jit
    def forward(params, obs):
        return net.apply(params, None, obs)

    @jax.jit
    def loss_fn(params, batch):
        obs, act, old_lp, adv, ret = batch
        a_p, b_p, v = forward(params, obs)
        new_lp = beta_log_prob(act, a_p, b_p)
        ratio = jnp.exp(new_lp - old_lp)
        surr1 = ratio * adv
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        pi_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        v_loss = jnp.mean(jnp.square(v - ret))
        ent = beta_entropy(a_p, b_p)
        return pi_loss + vf_coef * v_loss - ent_coef * ent

    clip_eps = args.clip_eps
    vf_coef = args.vf_coef
    ent_coef = args.ent_coef

    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def update_step(params, opt_state, batch):
        loss, grads = grad_fn(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    log_interval = (
        args.log_every
        if args.log_every is not None
        else max(1, args.total_updates // 10)
    )

    def run_one_repeat(rep: int) -> None:
        rep_seed = int(args.seed + rep)
        rng = np.random.default_rng(rep_seed)
        key = jax.random.PRNGKey(rep_seed)
        k_init, key = jax.random.split(key)
        params = net.init(k_init, jnp.zeros((1, obs_dim)))
        opt_state = tx.init(params)
        print(
            f"=== PPO repeat {rep + 1}/{args.repeats} (seed={rep_seed}) ===",
            flush=True,
        )

        curve_disc_return: list[float] = []
        curve_mean_reward: list[float] = []
        curve_ergodic_mean_u_at_log: list[float] = []
        curve_log_update_idx: list[int] = []

        t0 = time.time()
        for upd in range(args.total_updates):
            key, sk = jax.random.split(key)
            data = collect_rollout(
                forward, params, env, args.n_envs, args.horizon, rng, sk
            )

            obs_f = data["obs"].reshape(-1, obs_dim)
            act_f = data["actions"].reshape(-1)
            rew_f = data["rewards"].reshape(-1)
            val_f = data["values"].reshape(-1)
            logp_f = data["log_probs"].reshape(-1)
            done_f = data["dones"].reshape(-1)

            last_obs = jnp.asarray(data["last_obs"])
            _, _, last_v = forward(params, last_obs)
            last_v = np.asarray(last_v)

            H, N = data["rewards"].shape
            adv, ret = compute_gae(
                data["rewards"],
                data["values"],
                data["dones"],
                last_v,
                gamma,
                args.lam,
            )
            adv_f = adv.reshape(-1)
            ret_f = ret.reshape(-1)
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

            batch_size = obs_f.shape[0]
            idx = np.arange(batch_size)

            for _ in range(args.ppo_epochs):
                rng.shuffle(idx)
                for start in range(0, batch_size, args.minibatch):
                    mb = idx[start : start + args.minibatch]
                    if mb.size == 0:
                        continue
                    batch = (
                        jnp.asarray(obs_f[mb]),
                        jnp.asarray(act_f[mb]),
                        jnp.asarray(logp_f[mb]),
                        jnp.asarray(adv_f[mb]),
                        jnp.asarray(ret_f[mb]),
                    )
                    params, opt_state, _ = update_step(params, opt_state, batch)

            disc_ret_ep = []
            for i in range(N):
                g = 0.0
                for t in range(H - 1, -1, -1):
                    g = data["rewards"][t, i] + gamma * g * (1.0 - data["dones"][t, i])
                disc_ret_ep.append(g)
            curve_disc_return.append(float(np.mean(disc_ret_ep)))
            curve_mean_reward.append(float(np.mean(rew_f)))

            log_now = (upd + 1) % log_interval == 0 or upd == 0
            if log_now:
                erg_line = ""
                if args.log_ergodic_eval_paths > 0:
                    u_erg = eval_ergodic_markov_prices(
                        forward,
                        params,
                        env,
                        g_erg,
                        a_grid_vfi,
                        args.log_ergodic_eval_paths,
                        np.random.default_rng(rep_seed + 500_000 + upd),
                    )
                    curve_ergodic_mean_u_at_log.append(float(u_erg))
                    curve_log_update_idx.append(upd + 1)
                    erg_line = f"  ergodic_eval_mean_u({args.log_ergodic_eval_paths}p) {u_erg:.4f}"
                print(
                    f"update {upd + 1}/{args.total_updates}  "
                    f"rollout_tail_disc (reset~uniform, H={args.horizon}): {curve_disc_return[-1]:.4f}  "
                    f"mean_step_u {curve_mean_reward[-1]:.4f}"
                    f"{erg_line}",
                    flush=True,
                )

        train_time = time.time() - t0
        print(f"PPO training wall time: {train_time:.1f}s", flush=True)

        eval_rng = np.random.default_rng(rep_seed + 999)
        mean_u_eval = eval_ergodic_markov_prices(
            forward,
            params,
            env,
            g_erg,
            a_grid_vfi,
            args.eval_paths,
            eval_rng,
        )
        print(
            f"Post-train eval (ergodic a,e + Markov r,w, same as PEEnv): "
            f"mean discounted utility (T={env.T}): {mean_u_eval:.6f}",
            flush=True,
        )
        print(
            f"VFI npz mean_discounted_utility (may use different r,w law in MC): {vfi_gt_u:.6f}",
            flush=True,
        )

        tag = (
            f"pe_ppo_H{args.horizon}_E{args.n_envs}_U{args.total_updates}_lr{args.lr:.2E}"
            f"_R{args.repeats}_rep{rep}_s{rep_seed}"
        )
        pkl_path = results_dir / f"{tag}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "curve_mean_discounted_return_per_update": np.asarray(
                        curve_disc_return, dtype=np.float64
                    ),
                    "curve_mean_utility_per_step": np.asarray(
                        curve_mean_reward, dtype=np.float64
                    ),
                    "curve_ergodic_mean_discounted_u_at_log": np.asarray(
                        curve_ergodic_mean_u_at_log, dtype=np.float64
                    ),
                    "curve_log_update_indices": np.asarray(
                        curve_log_update_idx, dtype=np.int32
                    ),
                    "eval_ergodic_markov_rw_mean_discounted_utility": mean_u_eval,
                    "vfi_mean_discounted_utility": vfi_gt_u,
                    "config": {
                        "seed": rep_seed,
                        "base_seed": args.seed,
                        "repeat_index": rep,
                        "repeats": args.repeats,
                        "total_updates": args.total_updates,
                        "n_envs": args.n_envs,
                        "rollout_horizon": args.horizon,
                        "lr": args.lr,
                        "gamma": gamma,
                        "gae_lambda": args.lam,
                        "clip_eps": args.clip_eps,
                        "ppo_epochs": args.ppo_epochs,
                        "minibatch": args.minibatch,
                        "hidden": args.hidden,
                        "vf_coef": args.vf_coef,
                        "ent_coef": args.ent_coef,
                        "eval_paths": args.eval_paths,
                        "log_every": log_interval,
                        "log_ergodic_eval_paths": args.log_ergodic_eval_paths,
                        "rollout_reset_init": (
                            "PEEnv.reset: a~U[a_min,a_max], e,r,w uniform on discrete states"
                        ),
                        "mean_disc_return_tail_definition": (
                            "per-env backward discounted sum over last rollout window H; "
                            "not full T unless episode fits without trunc"
                        ),
                        "post_train_eval_r_w": "markov (r_trans, w_trans) like PEEnv",
                        "pe_env_T": int(env.T),
                        "obs_space": "(a, e_level, r_level, w_level)",
                        "action": "cshare in [0,1] Beta policy",
                    },
                    "train_wall_time_s": train_time,
                },
                f,
            )
        print(f"Saved metadata + curves to {pkl_path}", flush=True)

        if not args.skip_plot:
            set_static_styles()
            fig, ax = plt.subplots(1, 1, figsize=(9, 5))
            u_all = np.arange(1, len(curve_disc_return) + 1)
            ax.plot(
                u_all,
                curve_disc_return,
                label="Rollout-tail disc. return (reset~uniform, window H)",
            )
            if len(curve_ergodic_mean_u_at_log) > 0:
                ax.plot(
                    np.asarray(curve_log_update_idx, dtype=np.float64),
                    np.asarray(curve_ergodic_mean_u_at_log, dtype=np.float64),
                    marker="o",
                    linestyle="-",
                    label="Ergodic (a,e) eval mean U (T steps, Markov r,w)",
                )
            ax.axhline(
                y=vfi_gt_u, color="gray", linestyle="--", label="VFI mean_discounted_utility"
            )
            ax.axhline(
                y=mean_u_eval,
                color="C1",
                linestyle=":",
                label=f"PPO final ergodic eval ({args.eval_paths} paths)",
            )
            ax.set_xlabel("PPO update")
            ax.set_ylabel("Utility / return")
            ax.set_title(
                f"PEEnv PPO repeat {rep + 1}/{args.repeats} (seed={rep_seed}) vs VFI"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            pdf_path = results_dir / f"{tag}_training_curve.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved figure to {pdf_path}", flush=True)

    for rep in range(args.repeats):
        run_one_repeat(rep)


if __name__ == "__main__":
    main()
