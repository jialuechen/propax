"""Calibrate a kernel (and a clock) from synthetic data, with bootstrap uncertainty."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import propax as px

true = px.ExpKernel(jnp.array([2.0, 20.0, 200.0]), jnp.array([1.0, 0.5, 2.0]))
rates = px.ExpKernel.rate_grid(1 / 500, 1.0, per_decade=2)        # fixed grid, wider than the truth

blocks = []
for day in range(20):
    t, q = px.random_order_flow(500, 1.0, persistence=0.6, seed=day)
    pt = jnp.linspace(0, 1, 1001)
    blocks.append((t, q, pt, px.simulate_prices(true, t, q, pt, noise=0.3, seed=1000 + day)))

fit = px.fit_kernel_pooled(blocks, rates, smooth=1e-3)
taus = jnp.array([1 / 390, 10 / 390, 60 / 390])
print("G at 1, 10, 60 min  true:", np.asarray(true(taus)).round(3), " fitted:", np.asarray(fit.kernel(taus)).round(3))

bs = px.bootstrap_kernel(blocks, rates, n_boot=100, smooth=1e-3)
sd = np.sqrt(np.diag(np.asarray(px.kernel_value_covariance(rates, bs.cov, taus))))
print("bootstrap s.d. of G at those lags:", sd.round(3))
