"""Partial-equilibrium (PE) Gym-style environment used by ``pe_*`` RL scripts and ``pe_dspg``.

Markov states for productivity ``e``, interest rate ``r``, and wage factor ``w``; action is a
consumption *share* in ``[0, 1]``. Wealth evolves as in standard incomplete-markets setups;
horizon ``T`` is derived from ``beta`` and a truncation tolerance (see ``PEEnv.T``).
"""

from __future__ import annotations

import numpy as np


class PEEnv:
    """Discrete ``(e,r,w)`` shocks + continuous assets ``a``; returns per-period utility as reward."""

    def __init__(self) -> None:
        self.ne = 3
        self.e_grid = np.asarray([0.5343, 0.9735, 1.7739])
        self.e_trans = np.asarray(
            [
                [0.6460, 0.3539, 0.0001],
                [0.0304, 0.9392, 0.0304],
                [0.0001, 0.3539, 0.6460],
            ]
        )
        self.e_trans = self.e_trans / np.sum(self.e_trans, axis=1, keepdims=True)
        self.nr = 5
        self.r_grid = np.asarray([0.0193, 0.0218, 0.0247, 0.0279, 0.0315])
        self.r_trans = np.asarray(
            [
                [0.5655, 0.4172, 0.0173, 0.0000, 0.0000],
                [0.0920, 0.6079, 0.2966, 0.0035, 0.0000],
                [0.0034, 0.1423, 0.7032, 0.1507, 0.0004],
                [0.0000, 0.0056, 0.2287, 0.7131, 0.0526],
                [0.0000, 0.0000, 0.0081, 0.3438, 0.6480],
            ]
        )
        self.r_trans = self.r_trans / np.sum(self.r_trans, axis=1, keepdims=True)
        self.nw = 7
        self.w_grid = np.asarray([0.9418, 0.9608, 0.9802, 1.0000, 1.0202, 1.0408, 1.0618])
        self.w_trans = np.asarray(
            [
                [0.6657, 0.3312, 0.0031, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0541, 0.7002, 0.2442, 0.0014, 0.0000, 0.0000, 0.0000],
                [0.0001, 0.0842, 0.7363, 0.1787, 0.0007, 0.0000, 0.0000],
                [0.0000, 0.0003, 0.1254, 0.7487, 0.1254, 0.0003, 0.0000],
                [0.0000, 0.0000, 0.0007, 0.1787, 0.7363, 0.0842, 0.0001],
                [0.0000, 0.0000, 0.0000, 0.0014, 0.2442, 0.7002, 0.0541],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0031, 0.3312, 0.6657],
            ]
        )
        self.w_trans = self.w_trans / np.sum(self.w_trans, axis=1, keepdims=True)
        self.a_min = 0
        self.a_max = 100
        self.beta = 0.975
        self.c_min = 1e-3
        self.sigma = 1
        self.T = np.ceil(np.log(1e-2) / np.log(self.beta)).astype(int)

    def _gen_obs(self, a, eidx, ridx, widx):
        # TODO: normalize obs (a, e, r, w) using running bounds if useful for RL.
        obs = np.asarray([a, self.e_grid[eidx], self.r_grid[ridx], self.w_grid[widx]])
        return obs

    def reset(self):
        """Sample initial episode index 0, assets, and i.i.d. discrete shock indices."""
        ep = 0
        a = np.random.uniform(self.a_min, self.a_max)
        eidx = np.random.choice(self.ne)
        ridx = np.random.choice(self.nr)
        widx = np.random.choice(self.nw)
        obs = self._gen_obs(a, eidx, ridx, widx)
        state = (ep, a, eidx, ridx, widx)
        return state, obs

    def step(self, state, action):
        """One transition: apply share action, draw next ``(e,r,w)``, truncate at ``T``."""
        ep, a, eidx, ridx, widx = state
        consumption_share = np.clip(action, 0, 1)
        wealth = (1 + self.r_grid[ridx]) * a + self.e_grid[eidx] * self.w_grid[widx]
        consumption = np.clip(wealth * consumption_share, self.c_min, wealth - self.c_min)
        next_a = float(wealth - consumption)
        if self.sigma == 1:
            utility = np.log(consumption)
        else:
            utility = (consumption ** (1 - self.sigma)) / (1 - self.sigma)
        next_eidx = np.random.choice(self.ne, p=self.e_trans[eidx])
        next_ridx = np.random.choice(self.nr, p=self.r_trans[ridx])
        next_widx = np.random.choice(self.nw, p=self.w_trans[widx])
        obs = self._gen_obs(next_a, next_eidx, next_ridx, next_widx)
        state = (ep + 1, next_a, next_eidx, next_ridx, next_widx)
        done = False
        if ep + 1 >= self.T:
            trunc = True
        else:
            trunc = False
        return state, obs, utility, done, trunc


if __name__ == "__main__":
    env = PEEnv()
    state, obs = env.reset()
    print("Init State:", state)
    print("Init Obs:", obs)
    for i in range(5):
        print("\nStep:", i)
        action = np.random.uniform(0, 1)
        print("\tAction:", action)
        state, obs, reward, done, trunc = env.step(state, action)
        print("\tNext State:", state)
        print("\tNext Obs:", obs)
        print("\tReward:", reward)
        print("\tDone:", done)
        print("\tTrunc:", trunc)
