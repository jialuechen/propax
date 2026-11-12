# propax

**JAX Optimal Execution Library for Large Orders & Portfolios**

`propax` computes cost-minimizing trading schedules under propagator models of market impact. It represents transient impact as a causal convolution of trading flow with a multi-exponential kernel that approximates power-law decay.

The nonnegative exponential weights preserve positive-definite impact costs under the model's assumptions, ensuring convex optimization and ruling out negative-cost round trips. The same structure enables O(K)-per-trade streaming updates, linear-time impact operators, and implicit differentiation of optimal execution schedules.

The library implements *Structure-Preserving Optimal Execution with Multi-Exponential Propagators* (J. Chen, 2026). Section references below correspond to the paper.

## Technical Highlights

### 1. Structure-Preserving Optimization

Multi-exponential propagator kernels with nonnegative weights preserve the positive-definite structure of market impact costs. This ensures strictly convex execution problems under the model's assumptions, allowing globally optimal schedules to be computed using KKT-based quadratic programming and bound-constrained optimization.

The exponential representation also supports recursive state updates and matrix-free impact computation without constructing dense convolution matrices.

### 2. Implicit Differentiation

`propax` differentiates optimal execution schedules directly through the KKT conditions, rather than unrolling numerical optimization iterations.

This enables reverse-mode sensitivities of optimal schedules and execution costs with respect to impact parameters, risk parameters, and trading constraints, subject to standard active-set regularity conditions.

### 3. Factor-Structured Portfolio Cross-Impact

For multi-asset execution, `propax` represents cross-impact through low-rank factor structures:

$$
G(\tau) = \sum_{k=1}^{K} e^{-\lambda_k \tau} W_k,
\qquad
W_k = B M_k B^\top + D_k.
$$

This representation reduces the computational burden of portfolio optimization while preserving the required impact-cost structure under suitable parameter constraints. Matrix-free operators and preconditioned conjugate gradients avoid constructing dense asset-by-time impact matrices.

## Installation

```bash
pip install -e .                 
pip install -e ".[test]" && pytest
```

Enable 64-bit precision for numerical accuracy. The solvers and tests assume this setting:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import propax as px

jax.config.update("jax_enable_x64", True)

# Approximate G(tau) ~ tau^-0.5 from 1 minute to 1 trading day.
kernel = px.ExpKernel.from_power_law(
    0.5, 1 / 390, 1.0, amplitude=40.0
)

# 78 five-minute execution intervals.
times = jnp.arange(78) / 78
sched = px.Schedule(
    times, size=1.0, eta=20.0, sigma=200.0, gamma=0.002
)
model = px.ImpactModel(kernel)

# Equality-constrained optimal schedule.
res = px.optimal_schedule(model, sched)

# Buy-only schedule with a 3% per-interval cap.
capped = px.optimal_schedule(
    model, sched, lower=0.0, upper=0.03
)
capped.upper_multiplier  # Shadow prices of the caps

# Sensitivity of the optimal schedule to kernel weights.
dq_dw = jax.jacrev(
    lambda w: px.optimal_schedule(
        px.ImpactModel(kernel._replace(weights=w)), sched
    ).q
)(kernel.weights)

# Optimal-cost sensitivities via the envelope theorem.
greeks = jax.grad(
    lambda s: px.optimal_cost(model, s)
)(sched)
```

See `examples/` for liquidity clocks, portfolio execution with cross-impact and kernel calibration.

## Features and Modules

| Module | Description | Paper |
|---|---|---|
| `kernels` | `ExpKernel`: kernel evaluation, half-life, log-spaced decay rates, and power-law approximation | §3.2 |
| `ops` | O(NK) impact recursion (`scan` and parallel `associative` scan), matrix-free Γq, FFT convolution, streaming `ImpactTracker`, and `state_from_history` | §3.4 |
| `model` | `ImpactModel`, `Schedule`, quadratic execution costs, `optimal_schedule`, and `optimal_cost` | §3.1, §4 |
| `qp` | Dense QP solvers: exact KKT, ADMM with active-set refinement for bounds, and reverse-mode implicit differentiation | §4 |
| `clock` | `BusinessClock` with normalized business time, log-linear rates, and gains | §5 |
| `diagnostics` | Round-trip eigenvalues, most profitable round trip, manipulation gap δ, and `check_model` | §3.3, §5.2 |
| `portfolio` | `FactorCrossImpact`, O(KNnr) cost operator, projected preconditioned CG, and adjoint gradients | §6 |
| `calibrate` | Decayed order-flow features, NNLS kernel fitting with controls and smoothing, pooled and block-bootstrap estimation, delta method, and joint kernel-clock fitting | §7 |
| `simulate` | Synthetic order flow with persistent signs and prices with transient impact | §8 |
| `reference` | Discrete Almgren–Chriss closed-form solution | §8.1 |

## Model Guarantees and Limitations

- **Positive weights, increasing clocks, and positive gains:** The impact cost matrix is positive definite, excluding negative-cost round trips and ensuring strict convexity of the scheduling problem.
- **Stationary kernels or a single exogenous clock:** Pure-impact purchases have buy-only optimal schedules, ruling out transaction-triggered manipulation in these settings.
- **Per-timescale clocks or amplitude modulation:** Positive definiteness is preserved, but intermediate counter-trades may be optimal. `manipulation_gap` measures the cost of prohibiting them, while `lower=0` enforces buy-only execution without losing convexity.
- **Exogenous clocks and gains:** The theoretical guarantees require these quantities to be independent of the trader's own orders.
- **Scope:** `propax` optimizes execution schedules. Tick sizes, queue position, transaction fees, and passive fill dynamics are outside the scheduling model.

## Differentiation

`optimal_schedule(...).q`, `.cost`, and `optimal_portfolio_schedule(...).Q` support reverse-mode automatic differentiation (`jax.grad`, `jax.jacrev`, and `jax.vjp`) with respect to floating-point model and schedule parameters, including bounds.

Gradients are computed by implicit differentiation of the KKT conditions, holding the active set fixed rather than differentiating through solver iterations.

Forward-mode differentiation (`jax.jvp` and `jax.jacfwd`) is not supported because the solvers use `jax.custom_vjp`. Sensitivities may be discontinuous when the active set changes.

## Numerical Considerations

- **Single-asset optimization:** The dense solver is intended for schedules with up to a few thousand trades. Streaming and matrix-free impact operations can handle substantially longer histories.
- **Portfolio optimization:** The portfolio solver uses conjugate gradients with a Jacobi preconditioner. Convergence depends on conditioning, and highly ill-conditioned problems may require improved preconditioning.
- **Calibration:** Kernel estimation uses SciPy's exact NNLS implementation and runs outside JAX `jit`.
- **Precision:** 64-bit floating-point arithmetic is recommended for reliable optimization and gradient calculations.

## Status

The library has been tested on synthetic data. Calibration using real metaorder data, including the associated identification challenges discussed in Section 7.3 of the paper, remains future work.

## License
This library is distributed under Apahce-2.0.