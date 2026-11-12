"""Reproduce the synthetic results of Section 8 of the paper with propax.

Run:  python examples/reproduce_paper.py [--trials 2000]
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import propax as px

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=300)
args = ap.parse_args()

N = 50
times = jnp.arange(N) / N
kernel = px.ExpKernel(jnp.array([1.0, 5.0, 25.0, 125.0]), jnp.full(4, 25.0))
sched = px.Schedule(times, 1.0, 50.0, 200.0, 0.005)
model = px.ImpactModel(kernel)

print("8.1 Correctness")
ac = px.optimal_schedule(px.ImpactModel(kernel._replace(weights=jnp.zeros(4))), sched)
ref = px.almgren_chriss_inventory(N, 1.0, 50.0, 200.0, 0.005, 1 / N)
print(f"  Almgren-Chriss max inventory error: {np.abs(1 - np.cumsum(ac.q) - ref).max():.1e}")
print(f"  round-trip min eigenvalue, positive weights: {float(px.round_trip_min_eigenvalue(px.cost_matrix(kernel, times))):+.1f}")
bad = px.ExpKernel(jnp.array([2.0, 40.0]), jnp.array([1.0, -0.6]))
print(f"  round-trip min eigenvalue, e^-2t - 0.6e^-40t: {float(px.round_trip_min_eigenvalue(px.cost_matrix(bad, times))):+.2f}")

def sched_of(theta):
    k = px.ExpKernel(theta[:4], theta[4:8])
    s = px.Schedule(times, 1.0, theta[8], theta[9], theta[10])
    return px.optimal_schedule(px.ImpactModel(k), s).q

theta = jnp.concatenate([kernel.rates, kernel.weights, jnp.array([50.0, 200.0, 0.005])])
J = jax.jacrev(sched_of)(theta)
fd = jnp.stack([(sched_of(theta.at[p].add(h)) - sched_of(theta.at[p].add(-h))) / (2 * h)
                for p, h in enumerate(1e-6 * jnp.maximum(1.0, jnp.abs(theta)))], axis=1)
print(f"  policy sensitivities, implicit vs finite differences: rel. error {float(jnp.linalg.norm(J - fd) / jnp.linalg.norm(fd)):.1e}")
res = px.optimal_schedule(model, sched)
print(f"  optimal cost V* = {float(res.cost):.2f} bp")
d = jnp.zeros(11).at[2].set(-5.0)
q_new = sched_of(theta + d)
q_lin = res.q + J @ d
print(f"  lambda_3 25 -> 20: first 20% share {float(res.q[:10].sum()):.2%} -> {float(q_new[:10].sum()):.2%} "
      f"(first order {float(q_lin[:10].sum()):.2%})")

print("8.2 Single vs per-timescale clocks")
rng = np.random.default_rng(1)
k3 = px.ExpKernel(jnp.array([0.5, 5.0, 50.0]), jnp.ones(3))

def gap_for(betas):
    sells, gaps = 0, []
    for _ in range(args.trials):
        t = np.sort(rng.uniform(0, 1, 20)); t[0] = 0.0
        lr = rng.normal(0, 1.5, 20)
        us = [np.concatenate([[0], np.cumsum(np.diff(t) * np.exp(b * lr[1:]))]) for b in betas]
        G = sum(np.exp(-k3.rates[i] * np.abs(us[i][:, None] - us[i][None, :])) for i in range(3))
        free = px.solve_qp(jnp.array(G), jnp.zeros(20), jnp.ones((1, 20)), jnp.array([1.0]))
        if float(free.x.min()) < -1e-10:
            sells += 1
            bo = px.solve_qp(jnp.array(G), jnp.zeros(20), jnp.ones((1, 20)), jnp.array([1.0]), lower=0.0)
            v0, v1 = 0.5 * free.x @ G @ free.x, 0.5 * bo.x @ G @ bo.x
            gaps.append(float((v1 - v0) / v0))
    return sells / args.trials, (np.median(gaps) if gaps else 0.0), (max(gaps) if gaps else 0.0)

for betas in [(1, 1, 1), (0.75, 1, 1.25), (0.5, 1, 1.5), (0, 1, 2)]:
    frac, med, mx = gap_for(betas)
    print(f"  sensitivities {betas}: sells in {frac:.1%} of trials, median gap {med:.5%}, max {mx:.4%}")

print("8.3 Factor cross-impact")
n, r, Nt = 60, 3, 20
B = jnp.array(rng.normal(size=(n, r)))
pm = px.FactorCrossImpact(jnp.array([0.3, 3.0, 30.0, 300.0]), B, jnp.array(rng.normal(size=(4, r, r))) * 0.5,
                          jnp.array(rng.uniform(0.5, 1.5, (4, n))))
W = np.asarray(pm.weight_matrices())
off = np.abs(W - np.eye(n) * np.diagonal(W, axis1=1, axis2=2)[:, None, :]).mean()
for mult in [0.5, 1, 2]:
    bad_draws = 0
    for _ in range(50):
        E = rng.normal(scale=mult * off, size=(n, n)); E = (E + E.T) / np.sqrt(2); np.fill_diagonal(E, 0)
        bad_draws += np.linalg.eigvalsh(W[0] + E).min() < 0
    print(f"  pairwise noise {mult}x: {bad_draws / 50:.0%} of draws lose positive semidefiniteness")
