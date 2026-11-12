import jax.numpy as jnp
import numpy as np

import propax as px


def test_positive_weights_no_round_trip_manipulation():
    k = px.ExpKernel(jnp.array([1.0, 5.0, 25.0, 125.0]), jnp.full(4, 25.0))
    G = px.cost_matrix(k, jnp.arange(50) / 50)
    assert px.round_trip_min_eigenvalue(G) > 0


def test_positive_but_not_positive_definite_kernel():
    k = px.ExpKernel(jnp.array([2.0, 40.0]), jnp.array([1.0, -0.6]))
    t = jnp.arange(50) / 50
    assert bool(jnp.all(k(t) > 0))
    assert px.round_trip_min_eigenvalue(px.cost_matrix(k, t)) < 0
    q, cost = px.most_profitable_round_trip(px.cost_matrix(k, t))
    assert abs(float(q.sum())) < 1e-10 and float(cost) < 0


def test_paper_multi_clock_counterexample():
    th1 = np.array([0, 0.10, 0.11, 1.11])
    th2 = np.array([0, 1.00, 1.10, 1.11])
    G = 0.5 * np.exp(-np.abs(th1[:, None] - th1)) + 0.5 * np.exp(-np.abs(th2[:, None] - th2))
    assert np.isclose(np.linalg.eigvalsh(G).min(), 0.0500, atol=1e-4)
    res = px.solve_qp(jnp.array(G), jnp.zeros(4), jnp.ones((1, 4)), jnp.array([1.0]))
    np.testing.assert_allclose(res.x, [0.46275, 0.15547, -0.09122, 0.47300], atol=1e-5)
    bo = px.solve_qp(jnp.array(G), jnp.zeros(4), jnp.ones((1, 4)), jnp.array([1.0]), lower=0.0)
    v0, v1 = 0.5 * res.x @ G @ res.x, 0.5 * bo.x @ G @ bo.x
    assert np.isclose((v1 - v0) / v0, 0.00114, atol=2e-5)


def test_single_clock_is_buy_only_random():
    rng = np.random.default_rng(0)
    k = px.ExpKernel(jnp.array([0.5, 5.0, 50.0]), jnp.ones(3))
    for _ in range(100):
        t = np.sort(rng.uniform(0, 1, 20))
        rho = np.exp(rng.normal(0, 1.5, 20))
        u = np.concatenate([[0], np.cumsum(np.diff(t) * rho[1:])])
        model = px.ImpactModel(k, clock=jnp.array(u))
        gap = px.manipulation_gap(model, px.Schedule(jnp.array(t), 1.0))
        assert not bool(gap.counter_trades)


def test_business_clock_normalization_and_pd():
    grid = jnp.linspace(0, 1, 101)
    clk = px.BusinessClock.from_rate(grid, jnp.exp(jnp.sin(6 * grid)))
    assert np.isclose(clk(1.0) - clk(0.0), 1.0)
    assert bool(jnp.all(jnp.diff(clk.cumulative) > 0))
    k = px.ExpKernel(jnp.array([1.0, 10.0]), jnp.array([1.0, 1.0]))
    t = jnp.linspace(0, 1, 30)
    info = px.check_model(px.ImpactModel(k, clock=clk(t)), t)
    assert info["positive_definite_guaranteed"] and info["round_trip_min_eigenvalue"] > 0
