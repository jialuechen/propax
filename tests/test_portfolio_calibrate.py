import jax
import jax.numpy as jnp
import numpy as np
import pytest

import propax as px


def _problem(seed=0, K=3, n=12, r=2, N=15):
    rng = np.random.default_rng(seed)
    m = px.FactorCrossImpact(jnp.array([0.5, 5.0, 50.0][:K]), jnp.array(rng.normal(size=(n, r))),
                             jnp.array(rng.normal(size=(K, r, r))) * 0.3, jnp.array(rng.uniform(0.5, 1.5, (K, n))))
    t = jnp.sort(jnp.array(rng.uniform(0, 1, N)))
    s = px.PortfolioSchedule(t, jnp.array(rng.normal(size=n)), 0.1, 0.5, m.loadings, jnp.eye(r) * 0.2, jnp.full(n, 0.1))
    return m, s


def _dense_Q(m, s):
    N, n = s.times.shape[0], s.sizes.shape[0]
    H = px.dense_hessian(m, s)
    A = jnp.kron(jnp.ones((1, N)), jnp.eye(n))
    c = px.linear_term(m, s).ravel()
    Kk = jnp.block([[H, A.T], [A, jnp.zeros((n, n))]])
    return jnp.linalg.solve(Kk, jnp.concatenate([-c, s.sizes]))[: N * n].reshape(N, n)


def test_operator_matches_dense_and_is_psd():
    m, s = _problem()
    H = px.dense_hessian(m, s)
    Q = jnp.array(np.random.default_rng(1).normal(size=(15, 12)))
    np.testing.assert_allclose(px.hessian_apply(m, s, Q).ravel(), H @ Q.ravel(), atol=1e-12)
    assert np.linalg.eigvalsh(np.asarray(H)).min() > 0


def test_portfolio_solution_and_gradients():
    m, s = _problem()
    res = px.optimal_portfolio_schedule(m, s, tol=1e-12)
    np.testing.assert_allclose(res.Q, _dense_Q(m, s), atol=1e-10)
    np.testing.assert_allclose(res.Q.sum(0), s.sizes, atol=1e-10)
    f = lambda m, s: jnp.sum(px.optimal_portfolio_schedule(m, s, tol=1e-12).Q[:5] ** 2)
    fd_ = lambda m, s: jnp.sum(_dense_Q(m, s)[:5] ** 2)
    g = jax.grad(f, argnums=(0, 1))(m, s)
    d = jax.grad(fd_, argnums=(0, 1))(m, s)
    for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(d)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_noisy_pairwise_cross_impact_breaks_psd():
    m, s = _problem(n=40, r=3)
    W = np.asarray(m.weight_matrices())
    rng = np.random.default_rng(5)
    off = np.abs(W[0] - np.diag(np.diag(W[0]))).mean()
    E = rng.normal(scale=2 * off, size=W[0].shape)
    E = (E + E.T) / np.sqrt(2)
    np.fill_diagonal(E, 0)
    assert np.linalg.eigvalsh(W[0]).min() >= -1e-12
    assert np.linalg.eigvalsh(W[0] + E).min() < 0


def test_calibration_recovers_weights():
    k = px.ExpKernel(jnp.array([2.0, 20.0, 200.0]), jnp.array([1.0, 0.5, 2.0]))
    t, q = px.random_order_flow(3000, 10.0, persistence=0.6, seed=1)
    pt = jnp.linspace(0, 10, 4001)
    p = px.simulate_prices(k, t, q, pt, noise=0.05, seed=2)
    fit = px.fit_kernel(t, q, pt, p, k.rates)
    np.testing.assert_allclose(fit.kernel.weights, k.weights, rtol=1e-3)


def test_bootstrap_and_delta_method():
    k = px.ExpKernel(jnp.array([2.0, 20.0, 200.0]), jnp.array([1.0, 0.5, 2.0]))
    blocks = []
    for d in range(8):
        t, q = px.random_order_flow(400, 1.0, persistence=0.5, seed=10 + d)
        pt = jnp.linspace(0, 1, 801)
        blocks.append((t, q, pt, px.simulate_prices(k, t, q, pt, noise=0.5, seed=100 + d)))
    bs = px.bootstrap_kernel(blocks, k.rates, n_boot=50)
    assert bs.samples.shape == (50, 3) and np.all(bs.samples >= 0)
    cov_G = px.kernel_value_covariance(k.rates, bs.cov, jnp.array([0.001, 0.01, 0.1]))
    assert np.all(np.linalg.eigvalsh(np.asarray(cov_G)) >= -1e-10)


def test_clock_calibration_recovers_coefficient():
    k = px.ExpKernel(jnp.array([2.0, 20.0, 200.0]), jnp.array([1.0, 0.5, 2.0]))
    t, q = px.random_order_flow(3000, 10.0, persistence=0.6, seed=1)
    pt = jnp.linspace(0, 10, 4001)
    grid = jnp.linspace(0, 10, 2001)
    z = jnp.sin(2 * np.pi * grid / 2.5)[:, None]
    clk = px.BusinessClock.from_rate(grid, jnp.exp(z @ jnp.array([0.8])))
    p = px.simulate_prices(k, t, q, pt, noise=0.02, seed=3, clock=clk)
    cf = px.fit_kernel_and_clock(t, q, pt, p, k.rates, grid, z)
    assert abs(cf.coef[0] - 0.8) < 0.01
    assert cf.loss_history[-1] < cf.loss_history[0]
