"""
Soft Actor-Critic (SAC) on pe_rl_env.PEEnv.

Mirrors pe_ppo.py structure: same obs (a, e, r, w), same step_train / ergodic Markov eval,
same logging keys for plot_pe_training_comparison.py.

Policy: squashed Gaussian to cshare in (0, 1) (tanh + affine) for stable SAC reparameterization;
PPO uses Beta — eval uses deterministic mean action (tanh(mu) mapped to [0,1]).

Training transitions use PEEnv.reset()-style starts (uniform), same as PPO.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

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

VFI_NPZ_NAME = "pe_vfi.npz"
DEFAULT_CUDA = "5"
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


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


def normalize_obs(obs: np.ndarray, env: PEEnv) -> np.ndarray:
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
    na, ne = g.shape
    flat = g.reshape(-1)
    flat = flat / (np.sum(flat) + 1e-20)
    idx = int(rng.choice(na * ne, p=flat))
    ia = idx // ne
    ie = idx % ne
    return float(a_grid[ia]), ie


def eval_ergodic_markov_prices(
    actor_apply,
    actor_params,
    env: PEEnv,
    g_erg: np.ndarray,
    a_grid: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """(a,e) from VFI ergodic_g; (r,w) Markov — deterministic mean cshare from actor."""
    T = int(env.T)
    beta = float(env.beta)
    disc = np.power(beta, np.arange(T, dtype=np.float64))

    @jax.jit
    def policy_mean_cshare(obs_norm: jnp.ndarray) -> jnp.ndarray:
        mu, _ = actor_apply(actor_params, obs_norm[None, :])
        z = mu[0]
        return 0.5 * (jnp.tanh(z) + 1.0)

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


def make_actor(obs_dim: int, hidden: int):
    def actor(obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = hk.nets.MLP([hidden, hidden], activation=jax.nn.relu, name="actor_body")(obs)
        mu = hk.Linear(1, name="mu")(h)[:, 0]
        log_std = hk.Linear(1, name="log_std")(h)[:, 0]
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    return hk.transform(actor)


def make_q_net(obs_dim: int, hidden: int):
    def qnet(obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, act[:, None]], axis=-1)
        h = hk.nets.MLP([hidden, hidden], activation=jax.nn.relu, name="q_body")(x)
        return hk.Linear(1, name="q")(h)[:, 0]

    return hk.transform(qnet)


def sample_squashed_action(
    key: jax.Array, mu: jnp.ndarray, log_std: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """z ~ N(mu, std), a = 0.5*(tanh(z)+1) in (0,1); returns (a, log_pi)."""
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mu.shape)
    z = mu + std * eps
    a = 0.5 * (jnp.tanh(z) + 1.0)
    a = jnp.clip(a, 1e-6, 1.0 - 1e-6)
    log_pz = -0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * ((z - mu) / std) ** 2
    log_det = jnp.log(0.5 + 1e-8) + jnp.log(1.0 - jnp.tanh(z) ** 2 + 1e-6)
    log_pi = log_pz - log_det
    return a, log_pi


def collect_rollout_sac(
    actor_apply,
    actor_params,
    env: PEEnv,
    n_envs: int,
    horizon: int,
    rng: np.random.Generator,
    jax_key: jax.Array,
) -> dict:
    """Same vectorized rollout as PPO; actions from SAC policy sample."""
    obs_l, act_l, rew_l, next_obs_n_l, done_l = [], [], [], [], []
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
            mu, log_std = actor_apply(actor_params, obs_i[None, :])
            a, lp = sample_squashed_action(k, mu[0], log_std[0])
            return a, lp

        xs, _lps = jax.vmap(sample_one)(obs_j, keys)
        xs = np.asarray(xs)

        rew_t = []
        done_t = []
        next_obs_n = []
        new_obss = []
        for i in range(n_envs):
            ns, o, u, trunc = step_train(states[i], float(xs[i]), env, rng)
            rew_t.append(u)
            done_t.append(float(trunc))
            no = normalize_obs(np.asarray(o, dtype=np.float64), env)
            if trunc:
                ns, o = env.reset()
                no = normalize_obs(np.asarray(o, dtype=np.float64), env)
            states[i] = ns
            new_obss.append(np.asarray(o, dtype=np.float64))
            next_obs_n.append(no)

        obs_l.append(obs_n.copy())
        act_l.append(xs.copy())
        rew_l.append(np.asarray(rew_t, dtype=np.float64))
        next_obs_n_l.append(np.stack(next_obs_n, axis=0))
        done_l.append(np.asarray(done_t, dtype=np.float64))
        obss = new_obss

    return {
        "obs": np.stack(obs_l, axis=0),
        "actions": np.stack(act_l, axis=0),
        "rewards": np.stack(rew_l, axis=0),
        "next_obs": np.stack(next_obs_n_l, axis=0),
        "dones": np.stack(done_l, axis=0),
    }


class ReplayBuffer:
    def __init__(self, cap: int, obs_dim: int):
        self.cap = int(cap)
        self.obs_dim = obs_dim
        self.obs = np.zeros((cap, obs_dim), dtype=np.float64)
        self.next_obs = np.zeros((cap, obs_dim), dtype=np.float64)
        self.act = np.zeros((cap,), dtype=np.float64)
        self.rew = np.zeros((cap,), dtype=np.float64)
        self.done = np.zeros((cap,), dtype=np.float64)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        o: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        no: np.ndarray,
        d: np.ndarray,
    ) -> None:
        n = o.shape[0]
        for i in range(n):
            j = self.ptr
            self.obs[j] = o[i]
            self.act[j] = a[i]
            self.rew[j] = r[i]
            self.next_obs[j] = no[i]
            self.done[j] = d[i]
            self.ptr = (self.ptr + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def sample(self, rng: np.random.Generator, batch: int) -> tuple[np.ndarray, ...]:
        idx = rng.integers(0, self.size, size=batch)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=None, help="Default: env.beta")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient for target Q networks.",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="SAC entropy temperature.")
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--grad_steps",
        type=int,
        default=80,
        help="Gradient steps per PPO-equivalent update (after buffer warm-up).",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=5000,
        help="Min transitions in buffer before first SAC gradient update.",
    )
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
        help="Each progress print: ergodic eval paths (0=skip).",
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

    results_dir = Path(__file__).resolve().parent / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    vfi_path = results_dir / VFI_NPZ_NAME
    if not vfi_path.is_file():
        raise FileNotFoundError(f"Need {vfi_path} (ergodic_g, a_grid). Run pe_vfi.py first.")

    vfi = dict(np.load(vfi_path, allow_pickle=False))
    if "ergodic_g" not in vfi or "a_grid" not in vfi:
        raise KeyError("pe_vfi.npz must contain ergodic_g and a_grid")
    g_erg = np.asarray(vfi["ergodic_g"], dtype=np.float64)
    a_grid_vfi = np.asarray(vfi["a_grid"], dtype=np.float64)
    vfi_gt_u = float(np.asarray(vfi["mean_discounted_utility"]).reshape(()))

    obs_dim = 4
    actor = make_actor(obs_dim, args.hidden)
    q1 = make_q_net(obs_dim, args.hidden)
    q2 = make_q_net(obs_dim, args.hidden)
    tx = optax.adam(args.lr)

    log_interval = (
        args.log_every
        if args.log_every is not None
        else max(1, args.total_updates // 10)
    )

    @jax.jit
    def actor_apply(p, obs):
        return actor.apply(p, None, obs)

    @jax.jit
    def q1_apply(p, obs, act):
        return q1.apply(p, None, obs, act)

    @jax.jit
    def q2_apply(p, obs, act):
        return q2.apply(p, None, obs, act)

    alpha_sac = float(args.alpha)
    tau_sac = float(args.tau)

    @jax.jit
    def sac_step(actor_p, q1_p, q2_p, tq1_p, tq2_p, opt_a, opt_q1, opt_q2, batch_o, rng_step):
        """Critic (TD on min twin Q target + entropy), then actor, then polyak target update."""
        o, a, r, no, d = batch_o
        k1, k2 = jax.random.split(rng_step)

        mu_n, log_std_n = actor_apply(actor_p, no)
        keys_n = jax.random.split(k1, o.shape[0])
        a_n, log_pi_n = jax.vmap(sample_squashed_action)(keys_n, mu_n, log_std_n)
        q1_nt = q1_apply(tq1_p, no, a_n)
        q2_nt = q2_apply(tq2_p, no, a_n)
        min_q_n = jnp.minimum(q1_nt, q2_nt)
        y = r + (1.0 - d) * gamma * jax.lax.stop_gradient(min_q_n - alpha_sac * log_pi_n)

        def loss_q_only(q1p, q2p):
            q1_c = q1_apply(q1p, o, a)
            q2_c = q2_apply(q2p, o, a)
            return jnp.mean((q1_c - y) ** 2 + (q2_c - y) ** 2)

        gq1, gq2 = jax.grad(loss_q_only, argnums=(0, 1))(q1_p, q2_p)
        u1, opt_q1_n = tx.update(gq1, opt_q1, q1_p)
        q1_p_n = optax.apply_updates(q1_p, u1)
        u2, opt_q2_n = tx.update(gq2, opt_q2, q2_p)
        q2_p_n = optax.apply_updates(q2_p, u2)

        def loss_pi_only(ap):
            mu, log_std = actor_apply(ap, o)
            keys_p = jax.random.split(k2, o.shape[0])
            a_pi, log_pi = jax.vmap(sample_squashed_action)(keys_p, mu, log_std)
            q1_pi = q1_apply(q1_p_n, o, a_pi)
            q2_pi = q2_apply(q2_p_n, o, a_pi)
            return jnp.mean(alpha_sac * log_pi - jnp.minimum(q1_pi, q2_pi))

        ga = jax.grad(loss_pi_only)(actor_p)
        ua, opt_a_n = tx.update(ga, opt_a, actor_p)
        actor_p_n = optax.apply_updates(actor_p, ua)

        tq1_n = jax.tree_util.tree_map(
            lambda t, p: tau_sac * p + (1.0 - tau_sac) * t, tq1_p, q1_p_n
        )
        tq2_n = jax.tree_util.tree_map(
            lambda t, p: tau_sac * p + (1.0 - tau_sac) * t, tq2_p, q2_p_n
        )
        return actor_p_n, q1_p_n, q2_p_n, tq1_n, tq2_n, opt_a_n, opt_q1_n, opt_q2_n

    def run_one_repeat(rep: int) -> None:
        rep_seed = int(args.seed + rep)
        rng = np.random.default_rng(rep_seed)
        key = jax.random.PRNGKey(rep_seed)
        k1, k2, k3, key = jax.random.split(key, 4)
        obs0 = jnp.zeros((1, obs_dim))
        actor_p = actor.init(k1, obs0)
        q1_p = q1.init(k2, obs0, jnp.zeros((1,)))
        q2_p = q2.init(k3, obs0, jnp.zeros((1,)))
        tq1_p = jax.tree_util.tree_map(lambda x: x + 0.0, q1_p)
        tq2_p = jax.tree_util.tree_map(lambda x: x + 0.0, q2_p)

        opt_a = tx.init(actor_p)
        opt_q1 = tx.init(q1_p)
        opt_q2 = tx.init(q2_p)

        buffer = ReplayBuffer(args.buffer_size, obs_dim)

        curve_disc_return: list[float] = []
        curve_mean_reward: list[float] = []
        curve_ergodic_mean_u_at_log: list[float] = []
        curve_log_update_idx: list[int] = []

        print(
            f"=== SAC repeat {rep + 1}/{args.repeats} (seed={rep_seed}) ===",
            flush=True,
        )

        t0 = time.time()
        for upd in range(args.total_updates):
            key, sk = jax.random.split(key)
            data = collect_rollout_sac(
                actor_apply, actor_p, env, args.n_envs, args.horizon, rng, sk
            )
            H, N = data["rewards"].shape
            o_flat = data["obs"].reshape(-1, obs_dim)
            a_flat = data["actions"].reshape(-1)
            r_flat = data["rewards"].reshape(-1)
            no_flat = data["next_obs"].reshape(-1, obs_dim)
            d_flat = data["dones"].reshape(-1)
            buffer.add_batch(o_flat, a_flat, r_flat, no_flat, d_flat)

            disc_ret_ep = []
            for i in range(N):
                g = 0.0
                for t in range(H - 1, -1, -1):
                    g = data["rewards"][t, i] + gamma * g * (1.0 - data["dones"][t, i])
                disc_ret_ep.append(g)
            curve_disc_return.append(float(np.mean(disc_ret_ep)))
            curve_mean_reward.append(float(np.mean(r_flat)))

            if buffer.size >= args.learning_starts:
                for _ in range(args.grad_steps):
                    bo, ba, br, bno, bd = buffer.sample(rng, args.batch_size)
                    batch_o = (
                        jnp.asarray(bo),
                        jnp.asarray(ba),
                        jnp.asarray(br),
                        jnp.asarray(bno),
                        jnp.asarray(bd),
                    )
                    key, skg = jax.random.split(key)
                    actor_p, q1_p, q2_p, tq1_p, tq2_p, opt_a, opt_q1, opt_q2 = sac_step(
                        actor_p,
                        q1_p,
                        q2_p,
                        tq1_p,
                        tq2_p,
                        opt_a,
                        opt_q1,
                        opt_q2,
                        batch_o,
                        skg,
                    )

            log_now = (upd + 1) % log_interval == 0 or upd == 0
            if log_now:
                erg_line = ""
                if args.log_ergodic_eval_paths > 0:
                    u_erg = eval_ergodic_markov_prices(
                        actor_apply,
                        actor_p,
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
        print(f"SAC training wall time: {train_time:.1f}s", flush=True)

        eval_rng = np.random.default_rng(rep_seed + 999)
        mean_u_eval = eval_ergodic_markov_prices(
            actor_apply,
            actor_p,
            env,
            g_erg,
            a_grid_vfi,
            args.eval_paths,
            eval_rng,
        )
        print(
            f"Post-train eval (ergodic a,e + Markov r,w): "
            f"mean discounted utility (T={env.T}): {mean_u_eval:.6f}",
            flush=True,
        )
        print(
            f"VFI npz mean_discounted_utility: {vfi_gt_u:.6f}",
            flush=True,
        )

        tag = (
            f"pe_sac_H{args.horizon}_E{args.n_envs}_U{args.total_updates}_lr{args.lr:.2E}"
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
                        "algorithm": "SAC",
                        "policy": "squashed_Gaussian_cshare",
                        "total_updates": args.total_updates,
                        "n_envs": args.n_envs,
                        "rollout_horizon": args.horizon,
                        "lr": args.lr,
                        "gamma": gamma,
                        "tau": args.tau,
                        "alpha_entropy": args.alpha,
                        "buffer_size": args.buffer_size,
                        "batch_size": args.batch_size,
                        "grad_steps_per_update": args.grad_steps,
                        "learning_starts": args.learning_starts,
                        "hidden": args.hidden,
                        "eval_paths": args.eval_paths,
                        "log_every": log_interval,
                        "log_ergodic_eval_paths": args.log_ergodic_eval_paths,
                        "rollout_reset_init": (
                            "PEEnv.reset: a~U[a_min,a_max], e,r,w uniform on discrete states"
                        ),
                        "mean_disc_return_tail_definition": (
                            "per-env backward discounted sum over last rollout window H"
                        ),
                        "post_train_eval_r_w": "markov (r_trans, w_trans) like PEEnv",
                        "pe_env_T": int(env.T),
                        "obs_space": "(a, e_level, r_level, w_level)",
                        "action": "cshare in (0,1) squashed Gaussian",
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
            ax.axhline(y=vfi_gt_u, color="gray", linestyle="--", label="VFI mean_discounted_utility")
            ax.axhline(
                y=mean_u_eval,
                color="C1",
                linestyle=":",
                label=f"SAC final ergodic eval ({args.eval_paths} paths)",
            )
            ax.set_xlabel("SAC update")
            ax.set_ylabel("Utility / return")
            ax.set_title(
                f"PEEnv SAC repeat {rep + 1}/{args.repeats} (seed={rep_seed}) vs VFI"
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
