"""Optimal schedule, constraints, sensitivities, streaming and receding horizon."""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import propax as px

# Kernel ~ power law tau^-0.5 from 1 minute to 1 day (time unit: trading days, 390 min).
kernel = px.ExpKernel.from_power_law(0.5, 1 / 390, 1.0, amplitude=40.0)   # bp per unit size
print(f"{kernel.n_states} states, G(0) = {float(kernel.instantaneous()):.1f} bp, "
      f"half-life = {kernel.half_life() * 390:.1f} min")

N = 78                                       # 5-minute buckets
times = jnp.arange(N) / N
sched = px.Schedule(times, size=1.0, eta=20.0, sigma=200.0, gamma=0.002)
model = px.ImpactModel(kernel)

res = px.optimal_schedule(model, sched)
print(f"unconstrained: cost {float(res.cost):.2f} bp, first hour {float(res.q[:12].sum()):.1%}")

capped = px.optimal_schedule(model, sched, lower=0.0, upper=0.03)   # buy-only, 3% per bucket
print(f"capped:        cost {float(capped.cost):.2f} bp, "
      f"value of a looser cap = {float(capped.upper_multiplier.sum()):.2f} bp per unit")

# Policy sensitivity: how the first-hour share responds to each kernel weight.
first_hour = lambda w: px.optimal_schedule(px.ImpactModel(kernel._replace(weights=w)), sched).q[:12].sum()
print("d(first-hour share)/d(weights):", jax.grad(first_hour)(kernel.weights).round(5))

# Cost sensitivities by the envelope theorem.
g = jax.grad(lambda s: px.optimal_cost(model, s))(sched)
print(f"dV/d(sigma) = {float(g.sigma):.4f}, dV/d(size) = {float(g.size):.2f}")

# Streaming: O(K) per trade on irregular event times.
tracker = px.ImpactTracker(kernel, t0=0.0)
for t, q in [(0.001, 0.01), (0.0013, 0.02), (0.004, 0.01)]:
    tracker.trade(t, q)
print(f"live impact at t=0.005: {float(tracker.impact(0.005)):.3f} bp")

# Receding horizon: re-solve the rest of the day given what has been traded.
done = res.q[:20]
z = px.state_from_history(kernel, times[:20], done, times[20])
rest = px.Schedule(times[20:], 1.0 - float(done.sum()), 20.0, 200.0, 0.002, prior_state=z)
print(f"re-optimized remaining {float(rest.size):.1%} over {N - 20} buckets")
print(px.optimal_schedule(model, rest).q[:5].round(4))
