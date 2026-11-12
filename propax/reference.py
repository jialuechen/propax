"""Closed-form reference solutions used for validation."""
from __future__ import annotations

import numpy as np


def almgren_chriss_inventory(N: int, size: float, eta: float, sigma: float, gamma: float, dt: float):
    """Inventory after each of N trades in the discrete Almgren-Chriss problem.

    Matches ``Schedule(times=arange(N)*dt, eta=eta, sigma=sigma, gamma=gamma)`` with a
    zero impact kernel: r_k = X sinh(kappa (N-k)) / sinh(kappa N), where
    2 (cosh kappa - 1) = gamma sigma^2 dt / (2 eta).
    """
    kappa = np.arccosh(1.0 + gamma * sigma ** 2 * dt / (4.0 * eta))
    k = np.arange(1, N + 1)
    if kappa == 0:
        return size * (N - k) / N
    return size * np.sinh(kappa * (N - k)) / np.sinh(kappa * N)


def almgren_chriss_trades(N, size, eta, sigma, gamma, dt):
    r = almgren_chriss_inventory(N, size, eta, sigma, gamma, dt)
    return -np.diff(np.concatenate([[size], r]))
