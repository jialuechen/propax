"""Economic diagnostics: positive definiteness, round trips and the manipulation gap."""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .model import ImpactModel, Schedule, optimal_schedule
from .ops import cost_matrix

Array = jnp.ndarray


def round_trip_basis(N: int, dtype=jnp.float64) -> Array:
    """Orthonormal basis (N, N-1) of round trips {q : sum(q) = 0}."""
    D = jnp.eye(N, dtype=dtype)[:, :-1] - jnp.eye(N, dtype=dtype)[:, 1:]
    Q, _ = jnp.linalg.qr(D)
    return Q


def round_trip_min_eigenvalue(Gamma) -> Array:
    """Smallest eigenvalue of the impact cost on round trips.

    Negative means some round trip has negative expected cost: standard price
    manipulation. Positive weights make this strictly positive (Proposition 1).
    """
    Gamma = jnp.asarray(Gamma)
    Z = round_trip_basis(Gamma.shape[0], Gamma.dtype)
    return jnp.linalg.eigvalsh(Z.T @ Gamma @ Z)[0]


def most_profitable_round_trip(Gamma):
    """Unit-norm round trip with the lowest expected cost, and that cost."""
    Gamma = jnp.asarray(Gamma)
    Z = round_trip_basis(Gamma.shape[0], Gamma.dtype)
    ev, V = jnp.linalg.eigh(Z.T @ Gamma @ Z)
    q = Z @ V[:, 0]
    return q, 0.5 * q @ Gamma @ q


class ManipulationGap(NamedTuple):
    gap: Array              # relative cost increase from forbidding counter-trades
    counter_trades: Array   # True if the unconstrained optimum trades against the order
    most_negative: Array    # most counter-directional trade, in the order's direction
    unconstrained_cost: Array
    one_sided_cost: Array


def manipulation_gap(model: ImpactModel, sched: Schedule, pure_impact: bool = True) -> ManipulationGap:
    """Transaction-triggered manipulation diagnostic delta of Section 5.2.

    Solves the problem with and without a one-sided constraint (buy-only for purchases,
    sell-only for sales). With ``pure_impact=True`` the temporary and risk terms are
    removed, which is the setting of Propositions 2 and 4: a gap of zero is guaranteed
    for a stationary kernel or a single clock, but not for per-timescale clocks or
    amplitude modulation.
    """
    s = sched._replace(eta=0.0, sigma=0.0, gamma=0.0) if pure_impact else sched
    side = jnp.sign(jnp.asarray(s.size))
    free = optimal_schedule(model, s)
    if float(side) >= 0:
        bounded = optimal_schedule(model, s, lower=0.0)
    else:
        bounded = optimal_schedule(model, s, upper=0.0)
    v0, v1 = free.cost, bounded.cost
    signed = side * free.q
    tol = 1e-10 * (1.0 + jnp.max(jnp.abs(free.q)))
    return ManipulationGap((v1 - v0) / jnp.abs(v0), jnp.min(signed) < -tol, jnp.min(signed), v0, v1)


def check_model(model: ImpactModel, u) -> dict:
    """Summary of the structural guarantees that apply to ``model`` on times ``u``."""
    k = model.kernel
    cm = k.is_completely_monotone()
    u = jnp.asarray(u if model.clock is None else model.clock)
    increasing = bool(jnp.all(jnp.diff(u) > 0))
    gains_ok = model.gains is None or bool(jnp.all(jnp.asarray(model.gains) > 0))
    return {
        "completely_monotone_kernel": cm,
        "increasing_clock": increasing,
        "positive_gains": gains_ok,
        "positive_definite_guaranteed": cm and increasing and gains_ok,
        "buy_only_guaranteed_pure_impact": cm and increasing and model.gains is None,
        "round_trip_min_eigenvalue": float(round_trip_min_eigenvalue(cost_matrix(k, u, model.gains))),
    }
