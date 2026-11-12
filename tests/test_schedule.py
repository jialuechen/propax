import jax
import jax.numpy as jnp
import numpy as np
import pytest

import propax as px

N = 50
T = jnp.arange(N) / N
K0 = px.ExpKernel(jnp.array([1.0, 5.0, 25.0, 125.0]), jnp.full(4, 25.0))
S0 = px.Schedule(T, 1.0, 50.0, 200.0, 0.005)


def test_almgren_chriss_recovery():
    res = px.optimal_schedule(px.ImpactModel(K0._replace(weights=jnp.zeros(4))), S0)
    ref = px.almgren_chriss_inventory(N, 1.0, 50.0, 200.0, 0.005, 1 / N)
    np.testing.assert_allclose(1 - np.cumsum(res.q), ref, atol=1e-12)


def test_paper_optimal_cost():
    res = px.optimal_schedule(px.ImpactModel(K0), S0)
    assert np.isclose(res.cost, 30.104148805, rtol=1e-8)
    assert np.isclose(res.q.sum(), 1.0)


def test_cost_matches_direct_formula():
    q = px.optimal_schedule(px.ImpactModel(K0), S0).q
    G = px.cost_matrix(K0, T)
    r = 1 - jnp.cumsum(q)
    direct = 0.5 * q @ G @ q + 50.0 * q @ q + 0.5 * 0.005 * 200.0 ** 2 * jnp.sum(px.holding_intervals(T) * r ** 2)
    assert np.isclose(px.expected_cost(px.ImpactModel(K0), S0, q), direct)


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_policy_sensitivity_matches_finite_differences():
    def sched_of(w):
        return px.optimal_schedule(px.ImpactModel(K0._replace(weights=w)), S0).q
    J = jax.jacrev(sched_of)(K0.weights)
    for k in range(4):
        e = jnp.zeros(4).at[k].set(1e-5)
        fd = (sched_of(K0.weights + e) - sched_of(K0.weights - e)) / 2e-5
        np.testing.assert_allclose(J[:, k], fd, rtol=1e-5, atol=1e-9)


def test_envelope_cost_sensitivity():
    g_env = jax.grad(lambda s: px.optimal_cost(px.ImpactModel(K0), s))(S0)
    g_full = jax.grad(lambda s: px.optimal_schedule(px.ImpactModel(K0), s).cost)(S0)
    for name in ["eta", "sigma", "gamma", "size"]:
        v = getattr(S0, name)
        fd = _fd(lambda x: px.optimal_schedule(px.ImpactModel(K0), S0._replace(**{name: x})).cost, v, 1e-6 * max(1, abs(v)))
        assert np.isclose(getattr(g_env, name), fd, rtol=1e-5)
        assert np.isclose(getattr(g_full, name), fd, rtol=1e-5)


def test_bounded_schedule_kkt_and_gradients():
    cap = 0.05
    res = px.optimal_schedule(px.ImpactModel(K0), S0, lower=0.0, upper=cap)
    q = np.asarray(res.q)
    assert bool(res.solution.polished)
    assert np.isclose(q.sum(), 1.0) and q.min() >= -1e-10 and q.max() <= cap + 1e-10
    # KKT: gradient + multiplier equals bound multipliers
    H, c, _ = px.quadratic_form(px.ImpactModel(K0), S0)
    g = H @ res.q + c + res.size_multiplier
    np.testing.assert_allclose(g, res.lower_multiplier - res.upper_multiplier, atol=1e-7)

    def f(w, cap):
        q = px.optimal_schedule(px.ImpactModel(K0._replace(weights=w)), S0, lower=0.0, upper=cap).q
        return jnp.sum(q * jnp.arange(N) ** 2) / N ** 2
    gw, gc = jax.grad(f, argnums=(0, 1))(K0.weights, cap)
    for k in range(4):
        fd = _fd(lambda x: f(K0.weights.at[k].set(x), cap), K0.weights[k], 1e-6)
        assert np.isclose(gw[k], fd, rtol=1e-5, atol=1e-10)
    assert np.isclose(gc, _fd(lambda x: f(K0.weights, x), cap, 1e-7), rtol=1e-5)


def test_bounded_matches_scipy():
    from scipy.optimize import minimize
    model = px.ImpactModel(K0)
    res = px.optimal_schedule(model, S0, lower=0.0, upper=0.04)
    H, c, const = px.quadratic_form(model, S0)
    H, c = np.asarray(H), np.asarray(c)
    ref = minimize(lambda x: 0.5 * x @ H @ x + c @ x, np.full(N, 1 / N), jac=lambda x: H @ x + c,
                   bounds=[(0, 0.04)] * N, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000})
    assert float(res.cost - const) <= ref.fun + 1e-8


def test_receding_horizon_prior_state():
    # Solving the second half with the first half's residual impact equals the
    # second-half block of the full problem's cost (pure impact, fixed first half).
    q_full = np.asarray(px.optimal_schedule(px.ImpactModel(K0), S0._replace(eta=1.0, gamma=0.0)).q)
    z = px.state_from_history(K0, T[:25], jnp.array(q_full[:25]), T[25])
    sub = px.Schedule(T[25:], float(q_full[25:].sum()), 1.0, 0.0, 0.0, prior_state=z)
    q_sub = np.asarray(px.optimal_schedule(px.ImpactModel(K0), sub).q)
    np.testing.assert_allclose(q_sub, q_full[25:], atol=1e-9)
