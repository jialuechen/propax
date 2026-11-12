"""Exact nonnegative least squares with optional free columns and penalties (NumPy/SciPy).

Calibration runs offline, so we use SciPy's active-set NNLS (Lawson-Hanson), which is
exact and robust on the ill-conditioned designs produced by nearby exponential rates.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls as _scipy_nnls


def nnls(A, y, *, free=None, ridge=0.0, smooth=0.0, maxiter=None):
    """Solve min ||A w + F v - y||^2 + ridge ||w||^2 + smooth ||D w||^2  s.t. w >= 0.

    Parameters
    ----------
    A : (m, k) design for the nonnegative coefficients ``w``.
    y : (m,) target.
    free : optional (m, f) design for unconstrained coefficients ``v`` (e.g. controls).
    ridge, smooth : Tikhonov penalties; ``D`` is the first-difference operator, which
        penalizes roughness of ``w`` along the (log-spaced) rate grid.

    Returns
    -------
    w : (k,) nonnegative coefficients.
    v : (f,) free coefficients (empty if ``free`` is None).
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    k = A.shape[1]
    blocks = [A]
    rhs = [y]
    if ridge > 0:
        blocks.append(np.sqrt(ridge) * np.eye(k))
        rhs.append(np.zeros(k))
    if smooth > 0 and k > 1:
        D = np.diff(np.eye(k), axis=0)
        blocks.append(np.sqrt(smooth) * D)
        rhs.append(np.zeros(k - 1))
    M = np.vstack(blocks)
    r = np.concatenate(rhs)
    if free is None:
        w, _ = _scipy_nnls(M, r, maxiter=maxiter)
        return w, np.zeros(0)
    F = np.asarray(free, dtype=float)
    f = F.shape[1]
    pad = np.zeros((M.shape[0] - F.shape[0], f))
    Ff = np.vstack([F, pad])
    # Split free coefficients into positive and negative parts; a tiny ridge on them
    # makes the split unique without measurably biasing the fit.
    eps = 1e-12 * max(1.0, float(np.sum(F * F)))
    Mfull = np.vstack([np.hstack([M, Ff, -Ff]),
                       np.hstack([np.zeros((2 * f, k)), np.sqrt(eps) * np.eye(2 * f)])])
    rfull = np.concatenate([r, np.zeros(2 * f)])
    sol, _ = _scipy_nnls(Mfull, rfull, maxiter=maxiter)
    return sol[:k], sol[k:k + f] - sol[k + f:]
