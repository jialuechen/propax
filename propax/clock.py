"""Business-time clocks and state-dependent gains (Section 5)."""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray


class BusinessClock(NamedTuple):
    """Piecewise-linear business time theta(t) = int_0^t rho(z_s) ds on a grid.

    Build with :meth:`from_rate`; call it on calendar times to get business times.
    Any strictly positive rate keeps the cost matrix positive definite (Proposition 3);
    a single clock also preserves buy-only optimality of pure-impact problems
    (Proposition 4).
    """
    grid: Array
    cumulative: Array

    @classmethod
    def from_rate(cls, grid, rate, normalize: bool = True):
        """Integrate a positive rate given at grid points (trapezoid rule).

        With ``normalize=True`` the clock has average rate one over the grid, which fixes
        the scale indeterminacy between the clock and the decay rates (Section 5.4).
        """
        grid = jnp.asarray(grid)
        rate = jnp.asarray(rate, grid.dtype)
        inc = 0.5 * (rate[1:] + rate[:-1]) * jnp.diff(grid)
        cum = jnp.concatenate([jnp.zeros(1, grid.dtype), jnp.cumsum(inc)])
        if normalize:
            cum = cum * (grid[-1] - grid[0]) / cum[-1]
        return cls(grid, cum + grid[0])

    def __call__(self, t):
        return jnp.interp(jnp.asarray(t), self.grid, self.cumulative)

    def rate(self):
        """Average rate on each grid interval."""
        return jnp.diff(self.cumulative) / jnp.diff(self.grid)


def log_linear_rate(features, coef):
    """rho(z) = exp(z @ coef): a positive clock rate from exogenous features (M, F)."""
    return jnp.exp(jnp.asarray(features) @ jnp.asarray(coef))


def log_linear_gains(features, coef):
    """a_k(z) = exp(z @ coef[:, k]): positive gains (N, K) for amplitude modulation."""
    return jnp.exp(jnp.asarray(features) @ jnp.asarray(coef))
