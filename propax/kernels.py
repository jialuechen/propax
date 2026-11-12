"""Completely monotone propagator kernels: G(tau) = sum_k w_k exp(-lambda_k tau)."""
from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


class ExpKernel(NamedTuple):
    """Multi-exponential decay kernel with nonnegative weights.

    ``G(tau) = sum_k weights[k] * exp(-rates[k] * |tau|)``.

    With ``weights >= 0`` and ``rates > 0`` the kernel is completely monotone, hence
    positive definite, convex and nonincreasing (Section 3 of the paper). The kernel is a
    JAX pytree, so it can be passed through ``jax.jit``, ``jax.grad`` and ``jax.vmap``.
    """

    rates: Array
    weights: Array

    # ------------------------------------------------------------------ evaluation
    @property
    def n_states(self) -> int:
        return self.rates.shape[-1]

    def __call__(self, tau):
        tau = jnp.abs(jnp.asarray(tau))
        return jnp.sum(self.weights * jnp.exp(-self.rates * tau[..., None]), axis=-1)

    def instantaneous(self):
        """G(0), the impact of a unit trade right after it executes."""
        return jnp.sum(self.weights)

    def integral(self):
        """Total area under the kernel, int_0^inf G(tau) dtau."""
        return jnp.sum(self.weights / self.rates)

    def half_life(self, tol: float = 1e-12) -> float:
        """Time at which G falls to G(0)/2 (bisection; not jittable)."""
        g0 = float(self.instantaneous())
        if g0 <= 0:
            return float("nan")
        lo, hi = 0.0, 1.0 / float(jnp.min(self.rates))
        while float(self(hi)) > g0 / 2:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(self(mid)) > g0 / 2:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(hi, 1.0):
                break
        return 0.5 * (lo + hi)

    def is_completely_monotone(self) -> bool:
        return bool(jnp.all(self.weights >= 0) & jnp.all(self.rates > 0))

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def rate_grid(tau_min: float, tau_max: float, n: int | None = None,
                  per_decade: int = 2, pad_decades: float = 0.5) -> Array:
        """Log-spaced decay rates covering timescales [tau_min, tau_max]."""
        lo = 1.0 / tau_max * 10 ** (-pad_decades)
        hi = 1.0 / tau_min * 10 ** pad_decades
        if n is None:
            n = max(2, int(math.ceil(per_decade * math.log10(hi / lo))) + 1)
        return jnp.asarray(np.geomspace(lo, hi, n))

    @classmethod
    def from_rates(cls, rates, weights=None):
        rates = jnp.asarray(rates, dtype=float)
        weights = jnp.full(rates.shape, 1.0 / rates.shape[0]) if weights is None else jnp.asarray(weights, dtype=float)
        return cls(rates, weights)

    @classmethod
    def from_power_law(cls, beta: float, tau_min: float, tau_max: float, *,
                       amplitude: float = 1.0, n: int | None = None,
                       per_decade: int = 3, fit: bool = True, n_fit: int = 400):
        """Approximate ``amplitude * (tau / tau_min) ** (-beta)`` on [tau_min, tau_max].

        Uses the Laplace representation tau^-beta = (1/Gamma(beta)) int lambda^(beta-1)
        exp(-lambda tau) d lambda. With ``fit=True`` the quadrature weights are refined
        by nonnegative least squares on the relative error over the target range, which
        is typically accurate to well under 1% with three rates per decade.
        """
        from ._nnls import nnls

        rates = np.asarray(cls.rate_grid(tau_min, tau_max, n=n, per_decade=per_decade))
        dlog = np.log(rates[1] / rates[0]) if rates.size > 1 else 1.0
        scale = amplitude * tau_min ** beta
        w = scale * rates ** beta * dlog / math.gamma(beta)          # midpoint rule in log(lambda)
        if fit:
            taus = np.geomspace(tau_min, tau_max, n_fit)
            target = amplitude * (taus / tau_min) ** (-beta)
            A = np.exp(-np.outer(taus, rates)) / target[:, None]        # relative-error weighting
            w, _ = nnls(A, np.ones_like(taus))
        return cls(jnp.asarray(rates), jnp.asarray(w))


def as_kernel(kernel_or_rates, weights=None) -> ExpKernel:
    if isinstance(kernel_or_rates, ExpKernel):
        return kernel_or_rates
    return ExpKernel.from_rates(kernel_or_rates, weights)
