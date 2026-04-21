"""
Deep Deterministic Policy Gradient (DDPG) on pe_rl_env.PEEnv.

Aligned with pe_ppo.py / pe_sac.py: same obs, step_train, ergodic Markov eval, replay + polyak
targets, and the same pickle keys for plot_pe_training_comparison.py.

Actor: deterministic cshare in (0, 1) via 0.5*(tanh(z)+1). Exploration: Gaussian noise on the
deterministic action (linear decay of std over training). Twin Q critics; TD target uses min
of the two target Qs; actor maximizes Q1(s, pi(s)).
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
    T = int(env.T)
    beta = float(env.beta)
    disc = np.power(beta, np.arange(T, dtype=np.float64))

    @jax.jit
    def policy_det_cshare(obs_norm: jnp.ndarray) -> jnp.ndarray:
        return actor_apply(actor_params, obs_norm[None, :])[0]

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
            cshare = float(policy_det_cshare(jnp.asarray(on)))
            state, _, u, trunc = step_train(state, cshare, env, rng)
            u_path[t] = u
            if trunc:
                break
        total += float(np.sum(u_path * disc))
    return total / n_paths


def make_actor(obs_dim: int, hidden: int):
    def actor(obs: jnp.ndarray) -> jnp.ndarray:
        h = hk.nets.MLP([hidden, hidden], activation=jax.nn.relu, name="actor_body")(obs)
        z = hk.Linear(1, name="pre_tanh")(h)[:, 0]
        return 0.5 * (jnp.tanh(z) + 1.0)

    return hk.transform(actor)


def make_q_net(obs_dim: int, hidden: int):
    def qnet(obs: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, act[:, None]], axis=-1)
        h = hk.nets.MLP([hidden, hidden], activation=jax.nn.relu, name="q_body")(x)
        return hk.Linear(1, name="q")(h)[:, 0]

    return hk.transform(qnet)


def collect_rollout_ddpg(
    actor_apply,
    actor_params,
    env: PEEnv,
    n_envs: int,
    horizon: int,
    rng: np.random.Generator,
    noise_std: float,
) -> dict:
    obs_l, act_l, rew_l, next_obs_n_l, done_l = [], [], [], [], []
    states = []
    obss = []
    for _ in range(n_envs):
        st, obs = env.reset()
        states.append(st)
        obss.append(np.asarray(obs, dtype=np.float64))

    for _ in range(horizon):
        obs_arr = np.stack(obss, axis=0)
        obs_n = normalize_obs(obs_arr, env)
        obs_j = jnp.asarray(obs_n)
        mu = np.asarray(actor_apply(actor_params, obs_j))
        noise = rng.normal(0.0, noise_std, size=mu.shape)
        xs = np.clip(mu + noise, 1e-6, 1.0 - 1e-6).astype(np.float64)

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
        help="Soft update for target actor and target Q networks.",
    )
    parser.add_argument(
        "--noise_sigma_init",
        type=float,
        default=0.1,
        help="Gaussian exploration noise std on cshare (linearly decays to noise_sigma_final).",
    )
    parser.add_argument(
        "--noise_sigma_final",
        type=float,
        default=0.02,
        help="Exploration noise std at end of training.",
    )
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--grad_steps",
        type=int,
        default=80,
        help="Gradient steps per update after buffer warm-up.",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=5000,
        help="Min transitions before first gradient update.",
    )
    parser.add_argument("--eval_paths", type=int, default=4096)
    parser.add_argument(
        "--log_ergodic_eval_paths",
        type=int,
        default=256,
        help="Ergodic eval paths at each log (0=skip).",
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

    tau_ddpg = float(args.tau)

    @jax.jit
    def ddpg_step(actor_p, q1_p, q2_p, ta_p, tq1_p, tq2_p, opt_a, opt_q1, opt_q2, batch_o):
        o, a, r, no, d = batch_o
        a_t = actor_apply(ta_p, no)
        q1_nt = q1_apply(tq1_p, no, a_t)
        q2_nt = q2_apply(tq2_p, no, a_t)
        y = r + (1.0 - d) * gamma * jnp.minimum(q1_nt, q2_nt)
        y = jax.lax.stop_gradient(y)

        def loss_q(q1p, q2p):
            q1_c = q1_apply(q1p, o, a)
            q2_c = q2_apply(q2p, o, a)
            return jnp.mean((q1_c - y) ** 2 + (q2_c - y) ** 2)

        gq1, gq2 = jax.grad(loss_q, argnums=(0, 1))(q1_p, q2_p)
        u1, opt_q1_n = tx.update(gq1, opt_q1, q1_p)
        q1_p_n = optax.apply_updates(q1_p, u1)
        u2, opt_q2_n = tx.update(gq2, opt_q2, q2_p)
        q2_p_n = optax.apply_updates(q2_p, u2)

        def loss_a(ap):
            mu_o = actor_apply(ap, o)
            return -jnp.mean(q1_apply(q1_p_n, o, mu_o))

        ga = jax.grad(loss_a)(actor_p)
        ua, opt_a_n = tx.update(ga, opt_a, actor_p)
        actor_p_n = optax.apply_updates(actor_p, ua)

        ta_n = jax.tree_util.tree_map(
            lambda t, p: tau_ddpg * p + (1.0 - tau_ddpg) * t, ta_p, actor_p_n
        )
        tq1_n = jax.tree_util.tree_map(
            lambda t, p: tau_ddpg * p + (1.0 - tau_ddpg) * t, tq1_p, q1_p_n
        )
        tq2_n = jax.tree_util.tree_map(
            lambda t, p: tau_ddpg * p + (1.0 - tau_ddpg) * t, tq2_p, q2_p_n
        )
        return actor_p_n, q1_p_n, q2_p_n, ta_n, tq1_n, tq2_n, opt_a_n, opt_q1_n, opt_q2_n

    def noise_std_for_update(upd: int) -> float:
        if args.total_updates <= 1:
            return float(args.noise_sigma_final)
        t = upd / float(args.total_updates - 1)
        return (1.0 - t) * args.noise_sigma_init + t * args.noise_sigma_final

    def run_one_repeat(rep: int) -> None:
        rep_seed = int(args.seed + rep)
        rng = np.random.default_rng(rep_seed)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(rep_seed), 3)
        obs0 = jnp.zeros((1, obs_dim))
        actor_p = actor.init(k1, obs0)
        q1_p = q1.init(k2, obs0, jnp.zeros((1,)))
        q2_p = q2.init(k3, obs0, jnp.zeros((1,)))
        ta_p = jax.tree_util.tree_map(lambda x: x + 0.0, actor_p)
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
            f"=== DDPG repeat {rep + 1}/{args.repeats} (seed={rep_seed}) ===",
            flush=True,
        )

        t0 = time.time()
        for upd in range(args.total_updates):
            sig = noise_std_for_update(upd)
            data = collect_rollout_ddpg(
                actor_apply, actor_p, env, args.n_envs, args.horizon, rng, sig
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
                    actor_p, q1_p, q2_p, ta_p, tq1_p, tq2_p, opt_a, opt_q1, opt_q2 = ddpg_step(
                        actor_p,
                        q1_p,
                        q2_p,
                        ta_p,
                        tq1_p,
                        tq2_p,
                        opt_a,
                        opt_q1,
                        opt_q2,
                        batch_o,
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
                    f"mean_step_u {curve_mean_reward[-1]:.4f}  noise_std {sig:.4f}"
                    f"{erg_line}",
                    flush=True,
                )

        train_time = time.time() - t0
        print(f"DDPG training wall time: {train_time:.1f}s", flush=True)

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
        print(f"VFI npz mean_discounted_utility: {vfi_gt_u:.6f}", flush=True)

        tag = (
            f"pe_ddpg_H{args.horizon}_E{args.n_envs}_U{args.total_updates}_lr{args.lr:.2E}"
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
                        "algorithm": "DDPG",
                        "policy": "deterministic_tanh_cshare",
                        "total_updates": args.total_updates,
                        "n_envs": args.n_envs,
                        "rollout_horizon": args.horizon,
                        "lr": args.lr,
                        "gamma": gamma,
                        "tau": args.tau,
                        "noise_sigma_init": args.noise_sigma_init,
                        "noise_sigma_final": args.noise_sigma_final,
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
                        "action": "cshare in (0,1) deterministic + Gaussian exploration",
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
                label=f"DDPG final ergodic eval ({args.eval_paths} paths)",
            )
            ax.set_xlabel("DDPG update")
            ax.set_ylabel("Utility / return")
            ax.set_title(
                f"PEEnv DDPG repeat {rep + 1}/{args.repeats} (seed={rep_seed}) vs VFI"
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
