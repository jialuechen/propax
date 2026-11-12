"""Portfolio execution with factor-structured cross-impact (Section 6).

Kernel: G(tau) = sum_k exp(-lambda_k tau) W_k,  W_k = B M_k B' + diag(d_k),
with M_k = C_k C_k' (PSD by construction) and d_k >= 0. The cost operator is applied
in O(K N n r) time without forming any Nn x Nn matrix, and the schedule is found by
projected preconditioned conjugate gradients. Gradients of the optimal schedule with
respect to every model and schedule parameter use an adjoint solve (implicit
differentiation), not backpropagation through the iterations.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax

from .model import holding_intervals

Array = jnp.ndarray


class FactorCrossImpact(NamedTuple):
    rates: Array        # (K,)
    loadings: Array     # (n, r)  factor loadings B, ideally from the risk model
    factor_chol: Array  # (K, r, r)  M_k = C_k C_k'
    idio: Array         # (K, n)  nonnegative own-impact weights d_k

    @classmethod
    def from_factor_matrices(cls, rates, loadings, factor_matrices, idio):
        """Build from PSD factor matrices M_k (K, r, r); negative eigenvalues are clipped."""
        M = jnp.asarray(factor_matrices)
        ev, V = jnp.linalg.eigh(0.5 * (M + jnp.swapaxes(M, -1, -2)))
        C = V * jnp.sqrt(jnp.maximum(ev, 0.0))[..., None, :]
        return cls(jnp.asarray(rates), jnp.asarray(loadings), C, jnp.maximum(jnp.asarray(idio), 0.0))

    def factor_matrices(self) -> Array:
        return jnp.einsum("kij,klj->kil", self.factor_chol, self.factor_chol)

    def weight_matrices(self) -> Array:
        """Dense W_k (K, n, n); for inspection and tests only."""
        M = self.factor_matrices()
        W = jnp.einsum("ar,krs,bs->kab", self.loadings, M, self.loadings)
        return W + jax.vmap(jnp.diag)(self.idio)


class PortfolioSchedule(NamedTuple):
    """Portfolio problem: column sums of the (N, n) schedule must equal ``sizes``.

    Risk uses the factor covariance Sigma = R F R' + diag(s) per unit of ``times``;
    ``risk_factor_cov`` F should be symmetric positive semidefinite.
    """
    times: Array
    sizes: Array
    eta: Array
    gamma: Array
    risk_loadings: Array
    risk_factor_cov: Array
    risk_specific: Array
    clock: Optional[Array] = None


# ------------------------------------------------------------------ operators
def _sym_exp_apply(rate, u, X):
    """Gamma X with Gamma_ij = exp(-rate |u_i - u_j|), X of shape (N, n), in O(Nn)."""
    dec = jnp.exp(-rate * jnp.diff(u))

    def step(s, xs):
        d, x_prev = xs
        s = d * (s + x_prev)
        return s, s

    z = jnp.zeros(X.shape[1:], X.dtype)
    _, F = lax.scan(step, z, (dec, X[:-1]))
    _, Bk = lax.scan(step, z, (dec[::-1], X[::-1][:-1]))
    zero = jnp.zeros((1,) + X.shape[1:], X.dtype)
    return jnp.concatenate([zero, F]) + jnp.concatenate([Bk[::-1], zero]) + X


def apply_impact(model: FactorCrossImpact, u, Q) -> Array:
    """sum_k (Gamma_k kron W_k) vec(Q), returned as (N, n), in O(K N n r)."""
    u = jnp.asarray(u)
    QB = Q @ model.loadings                                    # (N, r)

    def one(rate, C, d):
        QW = ((QB @ C) @ C.T) @ model.loadings.T + Q * d       # Q W_k
        return _sym_exp_apply(rate, u, QW)

    return jnp.sum(jax.vmap(one)(model.rates, model.factor_chol, model.idio), axis=0)


def _apply_sigma(sched: PortfolioSchedule, Y):
    R = sched.risk_loadings
    return ((Y @ R) @ sched.risk_factor_cov) @ R.T + Y * sched.risk_specific


def _u(sched):
    return jnp.asarray(sched.times) if sched.clock is None else jnp.asarray(sched.clock)


def hessian_apply(model: FactorCrossImpact, sched: PortfolioSchedule, Q) -> Array:
    """H Q for the portfolio objective J(Q) = 1/2 <Q, HQ> + <c, Q> + const."""
    D = holding_intervals(sched.times)
    LQ = jnp.cumsum(Q, axis=0)
    LtDLQ = jnp.cumsum((D[:, None] * LQ)[::-1], axis=0)[::-1]
    return apply_impact(model, _u(sched), Q) + 2.0 * sched.eta * Q + sched.gamma * _apply_sigma(sched, LtDLQ)


def linear_term(model: FactorCrossImpact, sched: PortfolioSchedule) -> Array:
    D = holding_intervals(sched.times)
    tail = jnp.cumsum(D[::-1])[::-1]                            # L' D 1
    return -sched.gamma * tail[:, None] * _apply_sigma(sched, sched.sizes[None, :])


def portfolio_cost(model: FactorCrossImpact, sched: PortfolioSchedule, Q) -> Array:
    D = holding_intervals(sched.times)
    X = sched.sizes
    const = 0.5 * sched.gamma * jnp.sum(D) * (X @ _apply_sigma(sched, X[None, :])[0])
    return 0.5 * jnp.sum(Q * hessian_apply(model, sched, Q)) + jnp.sum(linear_term(model, sched) * Q) + const


def hessian_diagonal(model: FactorCrossImpact, sched: PortfolioSchedule) -> Array:
    """Exact diagonal of H, shape (N, n); used as a Jacobi preconditioner."""
    N = sched.times.shape[0]
    M = model.factor_matrices()
    w_diag = jnp.einsum("ar,krs,as->a", model.loadings, M, model.loadings) + jnp.sum(model.idio, axis=0)
    R = sched.risk_loadings
    s_diag = jnp.einsum("ar,rs,as->a", R, sched.risk_factor_cov, R) + sched.risk_specific
    tail = jnp.cumsum(holding_intervals(sched.times)[::-1])[::-1]
    eta = jnp.broadcast_to(jnp.asarray(sched.eta), (sched.sizes.shape[0],))
    return jnp.ones((N, 1)) * (w_diag + 2.0 * eta)[None, :] + sched.gamma * tail[:, None] * s_diag[None, :]


def dense_hessian(model: FactorCrossImpact, sched: PortfolioSchedule) -> Array:
    """Dense (Nn x Nn) Hessian in row-major vec(Q) order; for tests on small problems."""
    u = _u(sched)
    N, n = u.shape[0], sched.sizes.shape[0]
    tau = jnp.abs(u[:, None] - u[None, :])
    W = model.weight_matrices()
    H = sum(jnp.kron(jnp.exp(-model.rates[k] * tau), W[k]) for k in range(model.rates.shape[0]))
    eta = jnp.broadcast_to(jnp.asarray(sched.eta), (n,))
    D = holding_intervals(sched.times)
    L = jnp.tril(jnp.ones((N, N)))
    R = sched.risk_loadings
    Sigma = R @ sched.risk_factor_cov @ R.T + jnp.diag(sched.risk_specific)
    # hessian_apply multiplies rows by Sigma on the right, i.e. kron(., Sigma'); identical for symmetric F
    return H + 2.0 * jnp.kron(jnp.eye(N), jnp.diag(eta)) + sched.gamma * jnp.kron((L.T * D) @ L, Sigma.T)


# ------------------------------------------------------------------ projected PCG
def _pcg(hvp, c, Q0, minv, tol, maxiter):
    """min 1/2<Q,HQ> + <c,Q> over {column sums of Q = column sums of Q0}.

    Projected preconditioned CG (Gould, Hribar and Nocedal) with the oblique projection
    that matches a diagonal preconditioner. The residual's range-space component (the
    column-constant part, i.e. the multiplier of the size constraints) is removed at
    every step; without this "residual update" the iteration stalls near sqrt(eps) and
    then diverges.
    """
    msum = jnp.sum(minv, axis=0)

    def clean(R):
        return R - (jnp.sum(minv * R, axis=0) / msum)[None, :]

    def project(R):
        Z = R * minv
        return Z - minv * (jnp.sum(Z, axis=0) / msum)[None, :]

    r = clean(-(hvp(Q0) + c))
    z = project(r)
    rz0 = jnp.sum(r * z)

    def cond(state):
        _, _, _, _, rz, k = state
        return (k < maxiter) & (rz > (tol ** 2) * rz0)

    def body(state):
        Q, r, z, p, rz, k = state
        Hp = hvp(p)
        pHp = jnp.sum(p * Hp)
        alpha = jnp.where(pHp > 0, rz / pHp, 0.0)
        Q = Q + alpha * p
        r = clean(r - alpha * Hp)
        z = project(r)
        rz_new = jnp.sum(r * z)
        p = z + (rz_new / rz) * p
        return Q, r, z, p, jnp.where(pHp > 0, rz_new, 0.0), k + 1

    Q, r, z, p, rz, k = lax.while_loop(cond, body, (Q0, r, z, z, rz0, jnp.asarray(0)))
    return Q, k, jnp.sqrt(jnp.maximum(rz, 0.0) / jnp.maximum(rz0, 1e-300))


def _forward(model, sched, tol, maxiter):
    N = sched.times.shape[0]
    Q0 = jnp.ones((N, 1)) * sched.sizes[None, :] / N
    minv = 1.0 / hessian_diagonal(model, sched)
    return _pcg(lambda Q: hessian_apply(model, sched, Q), linear_term(model, sched), Q0, minv, tol, maxiter)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _solve(model, sched, tol, maxiter):
    return _forward(model, sched, tol, maxiter)


def _solve_fwd(model, sched, tol, maxiter):
    out = _forward(model, sched, tol, maxiter)
    return out, (model, sched, out[0])


def _solve_bwd(tol, maxiter, res, ct):
    gQ = ct[0]                       # iteration count and residual are not differentiable
    model, sched, Q = res
    hvp = lambda P: hessian_apply(model, sched, P)
    minv = 1.0 / hessian_diagonal(model, sched)
    U, _, _ = _pcg(hvp, -gQ, jnp.zeros_like(Q), minv, tol, maxiter)    # adjoint on the null space
    v = jnp.mean(gQ - hvp(U), axis=0)                                  # size-constraint multiplier
    _, vjp = jax.vjp(lambda m, s: hessian_apply(m, s, Q) + linear_term(m, s), model, sched)
    m_bar, s_bar = vjp(-U)
    s_bar = s_bar._replace(sizes=s_bar.sizes + v)
    return m_bar, s_bar


_solve.defvjp(_solve_fwd, _solve_bwd)


class PortfolioResult(NamedTuple):
    Q: Array
    cost: Array
    iterations: Array
    relative_residual: Array


def optimal_portfolio_schedule(model: FactorCrossImpact, sched: PortfolioSchedule,
                               tol: float = 1e-10, maxiter: int = 20_000) -> PortfolioResult:
    """Optimal (N, n) schedule; ``Q`` and ``cost`` are reverse-mode differentiable."""
    Q, k, rel = _solve(model, sched, tol, maxiter)
    return PortfolioResult(Q, portfolio_cost(model, sched, Q), k, jax.lax.stop_gradient(rel))

