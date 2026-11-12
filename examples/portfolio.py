"""Portfolio execution with factor cross-impact at scale."""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import propax as px

rng = np.random.default_rng(0)
n, r, N, K = 500, 10, 78, 4
B = jnp.array(rng.normal(size=(n, r)) / np.sqrt(r))
model = px.FactorCrossImpact(
    rates=jnp.array([1.0, 5.0, 25.0, 125.0]),
    loadings=B,
    factor_chol=jnp.array([np.eye(r) * s for s in (2.0, 2.0, 1.5, 1.0)]),
    idio=jnp.array(rng.uniform(5, 15, (K, n))),
)
sched = px.PortfolioSchedule(
    times=jnp.arange(N) / N, sizes=jnp.array(rng.normal(size=n)), eta=10.0, gamma=0.002,
    risk_loadings=B, risk_factor_cov=jnp.eye(r) * 400.0, risk_specific=jnp.full(n, 100.0),
)
solve = jax.jit(px.optimal_portfolio_schedule)
res = solve(model, sched)
res.Q.block_until_ready()
t0 = time.perf_counter()
res = solve(model, sched)
res.Q.block_until_ready()
print(f"{n} assets x {N} buckets = {n * N:,} unknowns: {int(res.iterations)} CG iterations, "
      f"{time.perf_counter() - t0:.2f}s, residual {float(res.relative_residual):.1e}")
print("size constraint error:", float(jnp.abs(res.Q.sum(0) - sched.sizes).max()))

# Gradient of the first-hour participation of asset 0 w.r.t. every factor matrix entry.
g = jax.grad(lambda m: px.optimal_portfolio_schedule(m, sched).Q[:12, 0].sum())(model)
print("d(first-hour asset 0)/d(factor_chol) norm:", float(jnp.linalg.norm(g.factor_chol)))
