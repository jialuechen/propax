"""Kernel calibration on a fixed decay-rate grid (Section 7).

With the rates fixed, impact is linear in the weights, so fitting the kernel is a convex
nonnegative least-squares problem. Regressing price changes on decayed signed order
flow deconvolves order-flow autocorrelation automatically (Section 7.2). Information
confounding (Section 7.3) is not solved by any regression; supply controls, and prefer
data with randomized execution speed where available.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._nnls import nnls
from .clock import BusinessClock
from .kernels import ExpKernel

Array = jnp.ndarray


def decay_features(trade_u, q, rates, eval_u, gains=None, strict: bool = True) -> Array:
    """phi_k(t) = sum_{u_j < u(t)} a_k(j) q_j exp(-lambda_k (u(t) - u_j)), shape (M, K).

    ``trade_u`` must be sorted. With ``strict=False`` trades at exactly ``eval_u`` count.
    Differentiable with respect to the times, which is what clock calibration uses.
    """
    trade_u = jnp.asarray(trade_u)
    q = jnp.asarray(q, trade_u.dtype)
    eval_u = jnp.asarray(eval_u, trade_u.dtype)
    rates = jnp.asarray(rates, trade_u.dtype)
    K = rates.shape[0]
    a = jnp.ones((trade_u.shape[0], K), trade_u.dtype) if gains is None else jnp.asarray(gains)
    dt = jnp.concatenate([jnp.zeros(1, trade_u.dtype), jnp.diff(trade_u)])
    dec = jnp.exp(-dt[:, None] * rates)

    def step(s, xs):
        d, x = xs
        s = d * s + x
        return s, s

    _, S = lax.scan(step, jnp.zeros(K, trade_u.dtype), (dec, a * q[:, None]))   # state right after trade j
    idx = jnp.searchsorted(trade_u, eval_u, side="left" if strict else "right") - 1
    has = idx >= 0
    idxc = jnp.clip(idx, 0)
    lag = eval_u - trade_u[idxc]
    return jnp.where(has[:, None], S[idxc] * jnp.exp(-lag[:, None] * rates), 0.0)


def regression_design(trade_times, q, price_times, prices, rates, *, horizon: int = 1,
                      clock: Optional[BusinessClock] = None):
    """Return (dPhi, dy): feature and price changes over ``horizon`` price steps."""
    tt = jnp.asarray(trade_times)
    pt = jnp.asarray(price_times, tt.dtype)
    ut, up = (tt, pt) if clock is None else (clock(tt), clock(pt))
    Phi = decay_features(ut, q, rates, up)
    p = jnp.asarray(prices, tt.dtype)
    return Phi[horizon:] - Phi[:-horizon], p[horizon:] - p[:-horizon]


class KernelFit(NamedTuple):
    kernel: ExpKernel
    controls: np.ndarray        # coefficients of the unconstrained control columns
    residual_std: float
    r2: float
    n_obs: int


def _finish(rates, A, y, w, v, F):
    pred = A @ w + (F @ v if F is not None else 0.0)
    res = y - pred
    ss = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return KernelFit(ExpKernel(jnp.asarray(rates), jnp.asarray(w)), v, float(np.std(res)),
                     1.0 - float(np.sum(res ** 2)) / ss, int(y.shape[0]))


def fit_kernel(trade_times, q, price_times, prices, rates, *, horizon: int = 1,
               smooth: float = 0.0, ridge: float = 0.0, controls=None,
               clock: Optional[BusinessClock] = None) -> KernelFit:
    """Fit nonnegative kernel weights on the fixed grid ``rates`` from one sample.

    ``controls`` (M - horizon, C) are unconstrained regressors aligned with the price
    changes, for example market returns or forecast variables.
    """
    A, y = regression_design(trade_times, q, price_times, prices, rates, horizon=horizon, clock=clock)
    A, y = np.asarray(A), np.asarray(y)
    F = None if controls is None else np.asarray(controls, float)
    w, v = nnls(A, y, free=F, ridge=ridge, smooth=smooth)
    return _finish(rates, A, y, w, v, F)


def _block_designs(blocks, rates, horizon, clock):
    out = []
    for blk in blocks:
        A, y = regression_design(blk[0], blk[1], blk[2], blk[3], rates, horizon=horizon, clock=clock)
        F = None if len(blk) < 5 or blk[4] is None else np.asarray(blk[4], float)
        out.append((np.asarray(A), np.asarray(y), F))
    return out


def fit_kernel_pooled(blocks: Sequence, rates, *, horizon: int = 1, smooth: float = 0.0,
                      ridge: float = 0.0, clock: Optional[BusinessClock] = None) -> KernelFit:
    """Fit one kernel from several independent samples (e.g. days or metaorders).

    Each block is ``(trade_times, q, price_times, prices[, controls])``. Features are
    built within each block, so no decay leaks across block boundaries.
    """
    designs = _block_designs(blocks, rates, horizon, clock)
    return _fit_designs(designs, rates, smooth, ridge)


def _fit_designs(designs, rates, smooth, ridge):
    A = np.vstack([d[0] for d in designs])
    y = np.concatenate([d[1] for d in designs])
    has_F = designs[0][2] is not None
    F = np.vstack([d[2] for d in designs]) if has_F else None
    w, v = nnls(A, y, free=F, ridge=ridge, smooth=smooth)
    return _finish(rates, A, y, w, v, F)


class BootstrapResult(NamedTuple):
    samples: np.ndarray     # (n_boot, K) weight samples
    mean: np.ndarray
    cov: np.ndarray


def bootstrap_kernel(blocks: Sequence, rates, *, n_boot: int = 200, seed: int = 0,
                     horizon: int = 1, smooth: float = 0.0, ridge: float = 0.0,
                     clock: Optional[BusinessClock] = None) -> BootstrapResult:
    """Block bootstrap of the kernel weights: resample whole blocks with replacement.

    Regression standard errors ignore dependence and misspecification; resampling days
    or metaorders does not (Section 7.5).
    """
    designs = _block_designs(blocks, rates, horizon, clock)
    rng = np.random.default_rng(seed)
    B = len(designs)
    samples = np.stack([
        np.asarray(_fit_designs([designs[i] for i in rng.integers(0, B, B)], rates, smooth, ridge).kernel.weights)
        for _ in range(n_boot)
    ])
    return BootstrapResult(samples, samples.mean(0), np.cov(samples, rowvar=False))


def delta_method(jacobian, cov) -> Array:
    """First-order covariance J Sigma J' of a function of the parameters (Section 7.5)."""
    J = jnp.asarray(jacobian)
    return J @ jnp.asarray(cov) @ J.T


def kernel_value_covariance(rates, weight_cov, taus) -> Array:
    """Covariance of the identifiable values G(tau) at the given lags from weight covariance."""
    E = jnp.exp(-jnp.outer(jnp.asarray(taus), jnp.asarray(rates)))
    return E @ jnp.asarray(weight_cov) @ E.T


class ClockFit(NamedTuple):
    kernel: ExpKernel
    coef: np.ndarray
    clock: BusinessClock
    loss_history: list


def fit_kernel_and_clock(trade_times, q, price_times, prices, rates, grid, grid_features, *,
                         horizon: int = 1, smooth: float = 0.0, ridge: float = 0.0,
                         n_iter: int = 40, step: float = 1.0, tol: float = 1e-10) -> ClockFit:
    """Jointly fit kernel weights and a log-linear business clock (Section 7.4).

    The clock rate is rho(z) = exp(z @ coef) on ``grid`` with exogenous features
    ``grid_features`` (M, F), normalized to average one. The procedure alternates an
    exact NNLS solve for the weights with a gradient step on the clock coefficients,
    accepting a step only if the profiled loss (weights refit) decreases.
    """
    grid = jnp.asarray(grid)
    Fz = jnp.asarray(grid_features, grid.dtype)
    tt = jnp.asarray(trade_times, grid.dtype)
    pt = jnp.asarray(price_times, grid.dtype)
    y = jnp.asarray(prices, grid.dtype)
    y = y[horizon:] - y[:-horizon]

    def design(coef):
        clk = BusinessClock.from_rate(grid, jnp.exp(Fz @ coef))
        Phi = decay_features(clk(tt), q, rates, clk(pt))
        return Phi[horizon:] - Phi[:-horizon]

    def loss(coef, w):
        r = y - design(coef) @ w
        return jnp.mean(r * r)

    grad_fn = jax.jit(jax.grad(loss))
    loss_fn = jax.jit(loss)

    def refit(coef):
        w, _ = nnls(np.asarray(design(coef)), np.asarray(y), ridge=ridge, smooth=smooth)
        return jnp.asarray(w)

    coef = jnp.zeros(Fz.shape[1], grid.dtype)
    w = refit(coef)
    cur = float(loss_fn(coef, w))
    hist = [cur]
    s = step
    for _ in range(n_iter):
        g = grad_fn(coef, w)
        gn = float(jnp.linalg.norm(g))
        if gn == 0.0:
            break
        accepted = False
        for _ in range(30):
            cand = coef - s * g / gn
            w_c = refit(cand)
            val = float(loss_fn(cand, w_c))
            if val < cur:
                coef, w, accepted = cand, w_c, True
                improvement = cur - val
                cur = val
                s *= 1.5
                break
            s *= 0.5
        hist.append(cur)
        if not accepted or improvement < tol * max(1.0, cur):
            break
    clk = BusinessClock.from_rate(grid, jnp.exp(Fz @ coef))
    return ClockFit(ExpKernel(jnp.asarray(rates), w), np.asarray(coef), clk, hist)
