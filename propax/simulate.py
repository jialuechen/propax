"""Synthetic order flow and prices for testing and examples."""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from .calibrate import decay_features
from .clock import BusinessClock
from .kernels import ExpKernel


def random_order_flow(n: int, horizon: float = 1.0, *, persistence: float = 0.0,
                      mean_size: float = 1.0, seed: int = 0):
    """Random trade times on [0, horizon) and signed sizes with persistent signs.

    ``persistence`` in [0, 1) is the probability that a trade keeps the previous sign
    beyond chance, mimicking autocorrelated order flow (metaorder splitting).
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, horizon, n))
    signs = np.empty(n)
    signs[0] = rng.choice([-1.0, 1.0])
    for i in range(1, n):
        signs[i] = signs[i - 1] if rng.uniform() < persistence else rng.choice([-1.0, 1.0])
    sizes = rng.lognormal(mean=np.log(mean_size), sigma=0.5, size=n)
    return jnp.asarray(t), jnp.asarray(signs * sizes)


def simulate_prices(kernel: ExpKernel, trade_times, q, price_times, *, noise: float = 0.0,
                    seed: int = 0, clock: Optional[BusinessClock] = None, drift: float = 0.0):
    """Mid prices = transient impact of past trades + drift * t + noise * Brownian motion."""
    tt = jnp.asarray(trade_times)
    pt = jnp.asarray(price_times, tt.dtype)
    ut, up = (tt, pt) if clock is None else (clock(tt), clock(pt))
    impact = decay_features(ut, q, kernel.rates, up) @ kernel.weights
    rng = np.random.default_rng(seed)
    dt = np.diff(np.asarray(pt), prepend=float(pt[0]))
    bm = np.cumsum(rng.normal(size=pt.shape[0]) * np.sqrt(np.maximum(dt, 0.0)))
    return impact + drift * (pt - pt[0]) + noise * jnp.asarray(bm)
