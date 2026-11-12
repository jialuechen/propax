"""Linear-time operators for multi-exponential propagators (Section 3.4).

All functions take the kernel as ``(rates, weights)`` arrays or an :class:`ExpKernel`,
work on irregular (business or calendar) times ``u``, and are differentiable with
respect to every floating-point input.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax

from .kernels import ExpKernel

Array = jnp.ndarray


def _gains(u, K, gains):
    return jnp.ones((u.shape[0], K), dtype=u.dtype) if gains is None else jnp.asarray(gains)


def _linear_recurrence(dec, inp, method):
    """z_i = dec_i * (z_{i-1} + inp_i), z_{-1} = 0, along axis 0 (any trailing shape)."""
    if method == "scan":
        def step(z, xs):
            d, b = xs
            z = d * (z + b)
            return z, z
        _, Z = lax.scan(step, jnp.zeros(dec.shape[1:], dec.dtype), (dec, inp))
        return Z
    if method == "associative":
        def op(left, right):
            a1, b1 = left
            a2, b2 = right
            return a1 * a2, a2 * b1 + b2
        _, Z = lax.associative_scan(op, (dec, dec * inp))
        return Z
    raise ValueError(f"unknown method {method!r}; use 'scan' or 'associative'")


def impact_path(kernel: ExpKernel, u, q, gains=None, method: str = "scan") -> Array:
    """Pre-trade impact I_i = sum_{j<i} G(u_i - u_j) q_j, exact on irregular times.

    Runs the K-state recursion z_i = exp(-lambda Delta_i)(z_{i-1} + q_{i-1}) in O(NK).
    ``method='associative'`` uses a parallel prefix scan (O(log N) depth).
    ``gains`` (N, K) applies the symmetric amplitude modulation of Section 5.3.
    """
    u = jnp.asarray(u)
    q = jnp.asarray(q, dtype=u.dtype)
    K = kernel.rates.shape[0]
    a = _gains(u, K, gains)
    inp = a * q[:, None]
    inp_prev = jnp.concatenate([jnp.zeros((1, K), inp.dtype), inp[:-1]])
    dt = jnp.concatenate([jnp.zeros(1, u.dtype), jnp.diff(u)])
    dec = jnp.exp(-dt[:, None] * kernel.rates)
    Z = _linear_recurrence(dec, inp_prev, method)
    return jnp.sum(Z * a * kernel.weights, axis=-1)


def apply_cost_matrix(kernel: ExpKernel, u, q, gains=None, method: str = "scan") -> Array:
    """Gamma @ q without forming Gamma, via a forward and a backward sweep (O(NK))."""
    u = jnp.asarray(u)
    q = jnp.asarray(q, dtype=u.dtype)
    a = _gains(u, kernel.rates.shape[0], gains)
    fwd = impact_path(kernel, u, q, a, method)
    bwd = impact_path(kernel, -u[::-1], q[::-1], a[::-1], method)[::-1]
    diag = jnp.sum(kernel.weights * a ** 2, axis=-1) * q
    return fwd + bwd + diag


def cost_matrix(kernel: ExpKernel, u, gains=None) -> Array:
    """Dense Gamma_ij = sum_k w_k a_k(i) a_k(j) exp(-lambda_k |u_i - u_j|)  (O(N^2 K))."""
    u = jnp.asarray(u)
    E = jnp.exp(-jnp.abs(u[:, None] - u[None, :])[..., None] * kernel.rates)
    if gains is not None:
        a = jnp.asarray(gains)
        E = E * a[:, None, :] * a[None, :, :]
    return E @ kernel.weights


def fft_impact(kernel: ExpKernel, q, dt: float) -> Array:
    """Pre-trade impact on a uniform grid by FFT convolution (O(N log N)).

    Equals :func:`impact_path` with ``u = arange(N) * dt`` to rounding error.
    """
    q = jnp.asarray(q)
    N = q.shape[0]
    g = kernel(jnp.arange(N, dtype=q.dtype) * dt).at[0].set(0.0)
    n = 2 * N
    return jnp.fft.irfft(jnp.fft.rfft(g, n) * jnp.fft.rfft(q, n), n)[:N]


def state_from_history(kernel: ExpKernel, past_u, past_q, u_now, past_gains=None) -> Array:
    """Impact states z_k just before time ``u_now`` from past trades at ``past_u < u_now``.

    Use it as ``Schedule.prior_state`` to re-optimize in receding-horizon (MPC) fashion.
    """
    past_u = jnp.asarray(past_u)
    past_q = jnp.asarray(past_q, dtype=past_u.dtype)
    a = _gains(past_u, kernel.rates.shape[0], past_gains)
    lag = jnp.asarray(u_now, past_u.dtype) - past_u
    mask = (lag > 0).astype(past_u.dtype)
    return jnp.sum(mask[:, None] * a * past_q[:, None] * jnp.exp(-lag[:, None] * kernel.rates), axis=0)


# ---------------------------------------------------------------------- streaming
class StreamState(NamedTuple):
    """Online impact state: K decayed sums and the time they refer to."""
    z: Array
    t: Array


def stream_init(kernel: ExpKernel, t0=0.0) -> StreamState:
    return StreamState(jnp.zeros_like(kernel.rates), jnp.asarray(t0, kernel.rates.dtype))


def stream_update(kernel: ExpKernel, state: StreamState, t, q, gain=None) -> StreamState:
    """Record a trade of size ``q`` at time ``t >= state.t``: O(K) work."""
    a = jnp.ones_like(kernel.rates) if gain is None else jnp.asarray(gain)
    z = state.z * jnp.exp(-kernel.rates * (t - state.t)) + a * q
    return StreamState(z, jnp.asarray(t, state.t.dtype))


def stream_impact(kernel: ExpKernel, state: StreamState, t, gain=None) -> Array:
    """Current impact at time ``t >= state.t`` (trades at exactly ``t`` included)."""
    a = jnp.ones_like(kernel.rates) if gain is None else jnp.asarray(gain)
    return jnp.sum(kernel.weights * a * state.z * jnp.exp(-kernel.rates * (t - state.t)))


class ImpactTracker:
    """Convenience wrapper around the jittable streaming functions."""

    def __init__(self, kernel: ExpKernel, t0: float = 0.0):
        self.kernel = kernel
        self.state = stream_init(kernel, t0)
        self._update = jax.jit(stream_update)
        self._impact = jax.jit(stream_impact)

    def trade(self, t, q, gain=None):
        self.state = self._update(self.kernel, self.state, t, q, gain)
        return self

    def impact(self, t, gain=None):
        return self._impact(self.kernel, self.state, t, gain)
