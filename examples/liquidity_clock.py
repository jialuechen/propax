"""Liquidity-dependent decay with a business-time clock and the manipulation-gap diagnostic."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import propax as px

kernel = px.ExpKernel(jnp.array([1.0, 5.0, 25.0, 125.0]), jnp.full(4, 25.0))
N = 78
times = jnp.arange(N) / N
grid = jnp.linspace(0, 1, 391)
# U-shaped intraday activity: impact recovers faster near the open and close.
activity = 1.0 + 3.0 * ((grid - 0.5) / 0.5) ** 2
clock = px.BusinessClock.from_rate(grid, activity)

sched = px.Schedule(times, 1.0, 50.0, 200.0, 0.005)
plain = px.optimal_schedule(px.ImpactModel(kernel), sched)
timed = px.optimal_schedule(px.ImpactModel(kernel, clock=clock(times)), sched)
print("share in first/last 30 min, calendar time:", float(plain.q[:6].sum()), float(plain.q[-6:].sum()))
print("share in first/last 30 min, business time:", float(timed.q[:6].sum()), float(timed.q[-6:].sum()))
print(px.check_model(px.ImpactModel(kernel, clock=clock(times)), times))

gap = px.manipulation_gap(px.ImpactModel(kernel, clock=clock(times)), sched)
print(f"single clock: counter-trades={bool(gap.counter_trades)}, gap={float(gap.gap):.2e}")

# Symmetric amplitude modulation: thinner books amplify short-lived impact more.
gains = px.log_linear_gains(jnp.asarray(1 / activity[::5][:N])[:, None] - 0.5, jnp.array([[0.8, 0.4, 0.1, 0.0]]))
gap = px.manipulation_gap(px.ImpactModel(kernel, clock=clock(times), gains=gains), sched)
print(f"with gains:   counter-trades={bool(gap.counter_trades)}, gap={float(gap.gap):.2e}")
