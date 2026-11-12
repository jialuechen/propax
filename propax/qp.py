"""Dense convex QP with implicit differentiation (Section 4).

Solves   min 1/2 x'Hx + c'x   s.t.  A x = b,  lower <= x <= upper
with H positive definite. Equality-only problems are solved exactly through the KKT
system. Bounded problems use ADMM (OSQP-style) followed by an exact active-set
"polish" solve. Gradients flow through ``jax.custom_vjp`` by implicit differentiation
of the KKT conditions with the active set held fixed, so no gradient ever
backpropagates through solver iterations.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve


class QPOptions(NamedTuple):
    """Static solver settings (hashable)."""
    bounded: bool = False
    max_iter: int = 50_000
    eps_abs: float = 1e-11
    eps_rel: float = 1e-11
    rho: float = 0.0        # ADMM penalty; 0 selects 0.1 * mean(diag(H))
    alpha: float = 1.6      # over-relaxation
    sigma: float = 1e-10    # proximal regularization


class QPSolution(NamedTuple):
    x: jnp.ndarray
    eq_multiplier: jnp.ndarray
    lower_multiplier: jnp.ndarray
    upper_multiplier: jnp.ndarray
    active_lower: jnp.ndarray
    active_upper: jnp.ndarray
    polished: jnp.ndarray
    iterations: jnp.ndarray


# ------------------------------------------------------------------ masked KKT
def _masked_matrix(H, A, free):
    nf = 1.0 - free
    Hm = H * free[:, None] * free[None, :] + jnp.diag(nf)
    Am = A * free[None, :]
    dead = (jnp.sum(jnp.abs(Am), axis=1) == 0).astype(H.dtype)
    K = jnp.block([[Hm, Am.T], [Am, jnp.diag(dead)]])
    return K, dead


def _masked_solve(H, c, A, b, free, x_fix):
    """KKT solve with coordinates where ``free == 0`` pinned at ``x_fix``."""
    K, dead = _masked_matrix(H, A, free)
    xf = (1.0 - free) * x_fix
    rhs = jnp.concatenate([-(c + H @ xf) * free + xf, (b - A @ xf) * (1.0 - dead)])
    s = jnp.linalg.solve(K, rhs)
    n = H.shape[0]
    return s[:n], s[n:]


# ------------------------------------------------------------------ ADMM
def _admm(H, c, A, b, lo, hi, opts: QPOptions):
    n, m = H.shape[0], A.shape[0]
    dtype = H.dtype
    rho = jnp.where(opts.rho > 0, opts.rho, 0.1 * jnp.trace(H) / n)
    rho_eq = 1e3 * rho
    M = H + (opts.sigma + rho) * jnp.eye(n, dtype=dtype) + rho_eq * (A.T @ A)
    cf = cho_factor(M)
    a = opts.alpha

    def residuals(x, zb, ye, yb):
        Ax = A @ x
        prim = jnp.maximum(jnp.max(jnp.abs(Ax - b), initial=0.0), jnp.max(jnp.abs(x - zb), initial=0.0))
        Hx, Aty = H @ x, A.T @ ye
        dual = jnp.max(jnp.abs(Hx + c + Aty + yb), initial=0.0)
        sp = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(Ax), initial=0.0), jnp.max(jnp.abs(b), initial=0.0)),
                         jnp.maximum(jnp.max(jnp.abs(x), initial=0.0), jnp.max(jnp.abs(zb), initial=0.0)))
        sd = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(Hx), initial=0.0), jnp.max(jnp.abs(c), initial=0.0)),
                         jnp.maximum(jnp.max(jnp.abs(Aty), initial=0.0), jnp.max(jnp.abs(yb), initial=0.0)))
        ok = (prim <= opts.eps_abs + opts.eps_rel * sp) & (dual <= opts.eps_abs + opts.eps_rel * sd)
        return ok

    def body(state):
        x, zb, ye, yb, k, _ = state
        rhs = opts.sigma * x - c + A.T @ (rho_eq * b - ye) + (rho * zb - yb)
        xt = cho_solve(cf, rhs)
        x_new = a * xt + (1 - a) * x
        ze_relax = a * (A @ xt) + (1 - a) * b
        zb_relax = a * xt + (1 - a) * zb
        zb_new = jnp.clip(zb_relax + yb / rho, lo, hi)
        ye_new = ye + rho_eq * (ze_relax - b)
        yb_new = yb + rho * (zb_relax - zb_new)
        done = residuals(x_new, zb_new, ye_new, yb_new)
        return x_new, zb_new, ye_new, yb_new, k + 1, done

    def cond(state):
        return (state[4] < opts.max_iter) & (~state[5])

    x0 = jnp.zeros(n, dtype)
    init = (x0, jnp.clip(x0, lo, hi), jnp.zeros(m, dtype), jnp.zeros(n, dtype),
            jnp.asarray(0), jnp.asarray(False))
    x, zb, ye, yb, k, _ = lax.while_loop(cond, body, init)
    return x, zb, ye, yb, k


def _solve_impl(H, c, A, b, lo, hi, opts: QPOptions):
    n = H.shape[0]
    dtype = H.dtype
    if not opts.bounded:
        free = jnp.ones(n, dtype)
        x, mu = _masked_solve(H, c, A, b, free, jnp.zeros(n, dtype))
        z = jnp.zeros(n, dtype)
        f = jnp.zeros(n, bool)
        return QPSolution(x, mu, z, z, f, f, jnp.asarray(True), jnp.asarray(0)), free

    x_a, zb, ye, yb, k = _admm(H, c, A, b, lo, hi, opts)
    # Active-set guess (OSQP polishing rule), then an exact reduced KKT solve.
    la = (zb - lo) < -yb
    ua = (hi - zb) < yb
    free = 1.0 - (la | ua).astype(dtype)
    x_fix = jnp.where(la, lo, jnp.where(ua, hi, 0.0))
    xp, mu = _masked_solve(H, c, A, b, free, x_fix)
    g = H @ xp + c + A.T @ mu
    scale = 1.0 + jnp.max(jnp.abs(xp), initial=0.0)
    tol_x = 1e-8 * scale
    tol_g = 1e-8 * (1.0 + jnp.max(jnp.abs(c), initial=0.0) + jnp.max(jnp.abs(H), initial=0.0) * scale)
    feasible = jnp.all((free == 0) | ((xp >= lo - tol_x) & (xp <= hi + tol_x)))
    dual_ok = jnp.all(jnp.where(la, g >= -tol_g, True) & jnp.where(ua, g <= tol_g, True))
    ok = feasible & dual_ok
    # Fallback: ADMM iterate with its own (tolerance-based) active set.
    la_f = jnp.where(ok, la, x_a <= lo + tol_x)
    ua_f = jnp.where(ok, ua, x_a >= hi - tol_x)
    free_f = 1.0 - (la_f | ua_f).astype(dtype)
    x = jnp.where(ok, xp, x_a)
    mu = jnp.where(ok, mu, ye)
    g = H @ x + c + A.T @ mu
    lam_lo = jnp.where(la_f, jnp.maximum(g, 0.0), 0.0)
    lam_hi = jnp.where(ua_f, jnp.maximum(-g, 0.0), 0.0)
    return QPSolution(x, mu, lam_lo, lam_hi, la_f, ua_f, ok, k), free_f


# ------------------------------------------------------------------ implicit VJP
@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _qp(H, c, A, b, lo, hi, opts):
    sol, _ = _solve_impl(H, c, A, b, lo, hi, opts)
    return sol


def _qp_fwd(H, c, A, b, lo, hi, opts):
    sol, free = _solve_impl(H, c, A, b, lo, hi, opts)
    return sol, (H, A, sol.x, free, sol.active_lower, sol.active_upper)


def _qp_bwd(opts, res, ct):
    H, A, x, free, la, ua = res
    gx = ct.x                               # multipliers are treated as non-differentiable
    n, m = H.shape[0], A.shape[0]
    K, dead = _masked_matrix(H, A, free)
    s = jnp.linalg.solve(K, jnp.concatenate([gx, jnp.zeros(m, H.dtype)]))   # K is symmetric
    u, v = s[:n], s[n:]
    uf = u * free
    H_bar = -jnp.outer(uf, x)
    c_bar = -uf
    b_bar = v * (1.0 - dead)
    fix_bar = (1.0 - free) * (u - H.T @ uf - A.T @ v)
    lo_bar = jnp.where(la, fix_bar, 0.0)
    hi_bar = jnp.where(ua, fix_bar, 0.0)
    return H_bar, c_bar, jnp.zeros_like(A), b_bar, lo_bar, hi_bar


_qp.defvjp(_qp_fwd, _qp_bwd)


def solve_qp(H, c, A=None, b=None, lower=None, upper=None, options: QPOptions | None = None) -> QPSolution:
    """Solve the QP; ``x`` is differentiable w.r.t. H, c, b, lower and upper (reverse mode)."""
    H = jnp.asarray(H)
    n = H.shape[0]
    dtype = H.dtype
    c = jnp.asarray(c, dtype)
    if A is None:
        A = jnp.zeros((0, n), dtype)
        b = jnp.zeros(0, dtype)
    A = jnp.atleast_2d(jnp.asarray(A, dtype))
    b = jnp.atleast_1d(jnp.asarray(b, dtype))
    bounded = lower is not None or upper is not None
    lo = jnp.full(n, -jnp.inf, dtype) if lower is None else jnp.broadcast_to(jnp.asarray(lower, dtype), (n,))
    hi = jnp.full(n, jnp.inf, dtype) if upper is None else jnp.broadcast_to(jnp.asarray(upper, dtype), (n,))
    opts = (options or QPOptions())._replace(bounded=bounded)
    return _qp(H, c, A, b, lo, hi, opts)
