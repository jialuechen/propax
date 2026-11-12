import jax
import jax.numpy as jnp
import numpy as np
import pytest

import propax as px


@pytest.fixture
def kernel():
    return px.ExpKernel(jnp.array([0.3, 3.0, 30.0, 300.0]), jnp.array([0.5, 1.0, 0.7, 2.0]))


def test_power_law_fit_accuracy():
    k = px.ExpKernel.from_power_law(0.5, 1e-3, 1.0, amplitude=2.0)
    taus = np.geomspace(1e-3, 1.0, 200)
    target = 2.0 * (taus / 1e-3) ** -0.5
    rel = np.abs(np.asarray(k(taus)) / target - 1)
    assert rel.max() < 0.01
    assert k.is_completely_monotone()


def test_kernel_summaries(kernel):
    assert np.isclose(kernel.instantaneous(), 4.2)
    assert np.isclose(kernel.integral(), np.sum(np.array([0.5, 1.0, 0.7, 2.0]) / np.array([0.3, 3, 30, 300])))
    h = kernel.half_life()
    assert np.isclose(kernel(h), kernel.instantaneous() / 2, rtol=1e-8)


@pytest.mark.parametrize("method", ["scan", "associative"])
def test_impact_path_matches_dense(kernel, method):
    rng = np.random.default_rng(0)
    u = jnp.sort(jnp.array(rng.uniform(0, 2, 60)))
    q = jnp.array(rng.normal(size=60))
    G = px.cost_matrix(kernel, u)
    dense = jnp.tril(G, -1) @ q
    np.testing.assert_allclose(px.impact_path(kernel, u, q, method=method), dense, rtol=1e-12, atol=1e-12)


def test_apply_cost_matrix_with_gains(kernel):
    rng = np.random.default_rng(1)
    u = jnp.sort(jnp.array(rng.uniform(0, 2, 40)))
    q = jnp.array(rng.normal(size=40))
    a = jnp.exp(jnp.array(rng.normal(size=(40, 4))) * 0.3)
    G = px.cost_matrix(kernel, u, a)
    np.testing.assert_allclose(px.apply_cost_matrix(kernel, u, q, a), G @ q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(G, G.T, atol=1e-14)
    assert np.linalg.eigvalsh(np.asarray(G)).min() > 0


def test_fft_matches_recursion(kernel):
    q = jnp.array(np.random.default_rng(2).normal(size=256))
    dt = 0.01
    u = jnp.arange(256) * dt
    np.testing.assert_allclose(px.fft_impact(kernel, q, dt), px.impact_path(kernel, u, q), atol=1e-10)


def test_streaming_matches_path(kernel):
    rng = np.random.default_rng(3)
    u = np.sort(rng.uniform(0, 1, 30))
    q = rng.normal(size=30)
    path = np.asarray(px.impact_path(kernel, jnp.array(u), jnp.array(q)))
    tr = px.ImpactTracker(kernel, t0=0.0)
    for i in range(30):
        assert np.isclose(float(tr.impact(u[i])), path[i], atol=1e-12)
        tr.trade(u[i], q[i])


def test_state_from_history(kernel):
    rng = np.random.default_rng(4)
    u = np.sort(rng.uniform(0, 1, 20))
    q = rng.normal(size=20)
    z = px.state_from_history(kernel, jnp.array(u[:12]), jnp.array(q[:12]), u[12])
    lhs = float(jnp.sum(kernel.weights * z))
    rhs = float(px.impact_path(kernel, jnp.array(u), jnp.array(q))[12])
    assert np.isclose(lhs, rhs, atol=1e-12)
