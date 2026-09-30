"""40 Adam steps on a synthetic multivariate sum of sines."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from spikelite import SpikeLite


def sum_of_sines(n, length, horizon, n_channels, key):
    k_phase, k_noise = jax.random.split(key)
    total = length + horizon
    t = jnp.arange(total, dtype=jnp.float32)
    freqs = jnp.array([0.07, 0.17, 0.31], dtype=jnp.float32)
    channel_phase = jnp.linspace(0.0, 1.2, n_channels, dtype=jnp.float32)
    phase = jax.random.uniform(k_phase, (n, 1, 1), dtype=jnp.float32) * 0.4
    tt = t[None, :, None]
    series = jnp.zeros((n, total, n_channels), dtype=jnp.float32)
    for freq_i, freq in enumerate(freqs):
        amp = 1.0 / (freq_i + 1)
        series = series + amp * jnp.sin(
            2.0 * jnp.pi * freq * tt + phase + channel_phase[None, None, :]
        )
    series = series + 0.08 * tt
    noise = 0.01 * jax.random.normal(k_noise, series.shape, dtype=jnp.float32)
    return (series + noise)[:, :length], (series + noise)[:, length:]


def main():
    key = jax.random.PRNGKey(0)
    k_model, k_data = jax.random.split(key)
    lookback, horizon, n_channels = 16, 4, 3
    model = SpikeLite(
        lookback,
        horizon,
        n_channels,
        key=k_model,
        d_model=16,
        n_branches=2,
        n_steps=2,
        n_heads=2,
        mask_rank=4,
    )
    x, y = sum_of_sines(32, lookback, horizon, n_channels, k_data)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, state, xb, yb):
        def loss_fn(mm):
            pred = mm(xb)
            return jnp.mean((pred - yb) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, state = optimizer.update(grads, state, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, updates)
        return m, state, loss

    model, opt_state, initial = step(model, opt_state, x, y)
    loss = initial
    for _ in range(39):
        model, opt_state, loss = step(model, opt_state, x, y)
    initial_value = float(initial)
    final_value = float(loss)
    print(f"initial loss {initial_value:.6f}")
    print(f"final loss {final_value:.6f}")
    return 0 if final_value < initial_value else 1


if __name__ == "__main__":
    raise SystemExit(main())
