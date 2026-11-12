"""Single-asset execution model: cost, optimal schedules and their sensitivities."""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from .kernels import ExpKernel
from .ops import cost_matrix
from .qp import QPOptions, QPSolution, solve_qp

Array = jnp.ndarray


class ImpactModel(NamedTuple):
    """Transient-impact model for one schedule.

    kernel : nonnegative multi-exponential kernel.
    clock  : optional business times of the trades, shape (N,). The kernel is evaluated
             on these instead of calendar times (Section 5.1). Must be increasing and
             exogenous, i.e. not a function of the trader's own orders.
    gains  : optional positive per-trade, per-state gains, shape (N, K), giving the
             symmetric amplitude modulation of Section 5.3.
    """
    kernel: ExpKernel
    clock: Optional[Array] = None
    gains: Optional[Array] = None


class Schedule(NamedTuple):
    """Scheduling problem: trade at ``times`` for a total signed ``size`` (buys > 0).

    eta   : temporary-cost coefficient (cost eta * q_i^2 per trade).
    sigma : volatility per square-root time unit of ``times``.
    gamma : mean-variance risk aversion.
    prior_state : optional (K,) impact states left by past trades just before the first
                  trade (see :func:`propax.state_from_history`), for receding-horizon use.
    """
    times: Array
    size: Array
    eta: Array = 0.0
    sigma: Array = 0.0
    gamma: Array = 0.0
    prior_state: Optional[Array] = None


class ScheduleResult(NamedTuple):
    q: Array                    # optimal signed trade sizes
    cost: Array                 # expected cost J(q*) including constant terms
    size_multiplier: Array      # KKT multiplier of the size constraint
    lower_multiplier: Array     # shadow prices of active lower bounds
    upper_multiplier: Array     # shadow prices of active upper bounds
    solution: QPSolution


def holding_intervals(times) -> Array:
    """Time each post-trade inventory is held; the last interval repeats the previous one."""
    times = jnp.asarray(times)
    if times.shape[0] == 1:
        return jnp.ones(1, times.dtype)
    dt = jnp.diff(times)
    return jnp.concatenate([dt, dt[-1:]])


def _business_times(model: ImpactModel, sched: Schedule):
    return jnp.asarray(sched.times) if model.clock is None else jnp.asarray(model.clock)


def quadratic_form(model: ImpactModel, sched: Schedule):
    """Return (H, c, const) with J(q) = 1/2 q'Hq + c'q + const (Section 3.1)."""
    times = jnp.asarray(sched.times)
    dtype = times.dtype
    N = times.shape[0]
    u = _business_times(model, sched)
    Gam = cost_matrix(model.kernel, u, model.gains)
    D = holding_intervals(times)
    L = jnp.tril(jnp.ones((N, N), dtype))
    risk = sched.gamma * sched.sigma ** 2
    H = Gam + 2.0 * sched.eta * jnp.eye(N, dtype=dtype) + risk * (L.T * D) @ L
    X = jnp.asarray(sched.size, dtype)
    c = -risk * X * (L.T @ D)
    const = 0.5 * risk * X ** 2 * jnp.sum(D)
    if sched.prior_state is not None:
        K = model.kernel.rates.shape[0]
        a = jnp.ones((N, K), dtype) if model.gains is None else jnp.asarray(model.gains)
        dec = jnp.exp(-(u - u[0])[:, None] * model.kernel.rates)
        c = c + jnp.sum(model.kernel.weights * a * dec * sched.prior_state, axis=-1)
    return H, c, const


def expected_cost(model: ImpactModel, sched: Schedule, q) -> Array:
    """Expected impact + temporary + risk cost of an arbitrary schedule ``q``."""
    H, c, const = quadratic_form(model, sched)
    q = jnp.asarray(q, H.dtype)
    return 0.5 * q @ H @ q + c @ q + const


def optimal_schedule(model: ImpactModel, sched: Schedule, lower=None, upper=None,
                     options: QPOptions | None = None) -> ScheduleResult:
    """Cost-minimizing schedule subject to sum(q) = size and optional bounds.

    Bounds express participation caps (``upper = cap * forecast_volume``) or a
    no-counter-trading constraint (``lower = 0`` for a purchase). The returned ``q`` and
    ``cost`` are differentiable in reverse mode with respect to every parameter of
    ``model`` and ``sched`` (and the bounds), via implicit differentiation.
    """
    H, c, const = quadratic_form(model, sched)
    N = H.shape[0]
    A = jnp.ones((1, N), H.dtype)
    b = jnp.atleast_1d(jnp.asarray(sched.size, H.dtype))
    sol = solve_qp(H, c, A, b, lower, upper, options)
    q = sol.x
    cost = 0.5 * q @ H @ q + c @ q + const
    return ScheduleResult(q, cost, sol.eq_multiplier[0], sol.lower_multiplier,
                          sol.upper_multiplier, sol)


def optimal_cost(model: ImpactModel, sched: Schedule, lower=None, upper=None,
                 options: QPOptions | None = None) -> Array:
    """Optimal cost V* with gradients by the envelope theorem (cheapest cost Greeks).

    The schedule and the multiplier are held fixed when differentiating the Lagrangian
    J(q) + mu (sum(q) - size), which is exact at the optimum. Bounds are treated as
    constants; differentiate :func:`optimal_schedule` instead if they depend on parameters.
    """
    res = optimal_schedule(model, sched, lower, upper, options)
    q = jax.lax.stop_gradient(res.q)
    mu = jax.lax.stop_gradient(res.size_multiplier)
    return expected_cost(model, sched, q) + mu * (jnp.sum(q) - sched.size)
