"""Checks for the SpikeLite kernel. CPU only, fixed keys, small shapes."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from spikelite import SpikeLite, masked_attention


def _small_model(key, **kwargs):
    params = dict(
        lookback=8,
        horizon=4,
        n_channels=3,
        d_model=8,
        n_branches=3,
        n_steps=2,
        n_heads=2,
        mask_rank=4,
        use_ssca=True,
    )
    params.update(kwargs)
    return SpikeLite(**params, key=key)


def test_telescoping_components():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 3), dtype=jnp.float32)
    model = SpikeLite(
        16,
        4,
        3,
        key=jax.random.PRNGKey(1),
        d_model=8,
        n_branches=3,
        n_steps=2,
        n_heads=2,
        mask_rank=4,
    )
    components = model.fsse_components(x)
    assert components.shape == (4, 2, 16, 3)
    assert jnp.allclose(components.sum(axis=0), x, atol=1e-4, rtol=1e-4)


def test_decay_init_strictly_decreasing():
    model = _small_model(jax.random.PRNGKey(2), n_branches=5, n_channels=3)
    tau = jax.nn.sigmoid(model.rho)
    assert jnp.all(tau[:-1] > tau[1:])
    expected = jnp.linspace(0.9, 0.3, 5, dtype=jnp.float32)
    assert jnp.allclose(tau[:, 0], expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(tau, tau[:, :1])
    assert jnp.allclose(model.theta, 1.0)


def test_surrogate_grad_reaches_threshold():
    model = SpikeLite(
        4,
        2,
        1,
        key=jax.random.PRNGKey(3),
        d_model=4,
        n_branches=1,
        n_steps=1,
        n_heads=1,
        mask_rank=2,
        use_ssca=False,
    )
    # U_1 = X_1 when S_0 = 0. X = theta puts the membrane on the threshold.
    x = jnp.ones((1, 4, 1), dtype=jnp.float32)

    def loss(m):
        return jnp.sum(m.fsse_spikes(x))

    grads = eqx.filter_grad(loss)(model)
    assert jnp.all(jnp.isfinite(grads.theta))
    assert float(jnp.sum(jnp.abs(grads.theta))) > 0.0


def test_mask_diagonal_and_binary():
    model = _small_model(jax.random.PRNGKey(4), n_channels=4, d_model=8)
    z = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 8), dtype=jnp.float32)
    mask = model.channel_mask(z)
    assert mask.shape == (2, 4, 4)
    diag = mask[:, jnp.arange(4), jnp.arange(4)]
    assert jnp.allclose(diag, 1.0)
    assert jnp.all((mask == 0.0) | (mask == 1.0))


def test_mask_ste_grad_wrt_wr():
    model = _small_model(jax.random.PRNGKey(6), n_channels=4, d_model=8, mask_rank=4)
    t = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
    rows = jnp.stack(
        [
            jnp.sin(2.0 * jnp.pi * t),
            jnp.sin(4.0 * jnp.pi * t),
            jnp.sin(8.0 * jnp.pi * t),
            jnp.cos(3.0 * jnp.pi * t),
        ]
    )
    z = rows[None, :, :]

    def loss(m):
        return jnp.sum(m.channel_mask(z))

    grads = eqx.filter_grad(loss)(model)
    assert jnp.all(jnp.isfinite(grads.W_r))
    assert float(jnp.sum(jnp.abs(grads.W_r))) > 0.0


def test_identical_channels_are_connected():
    model = SpikeLite(
        8,
        2,
        3,
        key=jax.random.PRNGKey(7),
        d_model=8,
        n_branches=2,
        n_steps=2,
        n_heads=2,
        mask_rank=4,
        gamma=1.0,
        eta=0.3,
    )
    row = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32)
    other = jnp.sin(jnp.linspace(0.0, 12.0, 8, dtype=jnp.float32))
    z = jnp.stack([row, row, other], axis=0)[None, :, :]
    mask = model.channel_mask(z)
    assert float(mask[0, 0, 1]) == 1.0
    assert float(mask[0, 1, 0]) == 1.0


def test_masked_attention_linearity_and_identity():
    key = jax.random.PRNGKey(8)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (2, 2, 3, 4), dtype=jnp.float32)
    k = jax.random.normal(kk, (2, 2, 3, 4), dtype=jnp.float32)
    v = jax.random.normal(kv, (2, 2, 3, 4), dtype=jnp.float32)
    mask = jnp.ones((3, 3), dtype=jnp.float32)
    out = masked_attention(q, k, v, mask, 0.125)
    out_double = masked_attention(q, 2.0 * k, v, mask, 0.125)
    assert jnp.allclose(out_double, 2.0 * out, atol=1e-5, rtol=1e-5)

    eye = jnp.eye(3, dtype=jnp.float32)
    v_perturbed = v.at[:, :, 1, :].add(1.5)
    out_a = masked_attention(q, k, v, eye, 0.125)
    out_b = masked_attention(q, k, v_perturbed, eye, 0.125)
    assert jnp.allclose(out_a[:, :, 0], out_b[:, :, 0], atol=1e-6)
    assert jnp.allclose(out_a[:, :, 2], out_b[:, :, 2], atol=1e-6)
    assert not jnp.allclose(out_a[:, :, 1], out_b[:, :, 1], atol=1e-5)


def test_forward_shape():
    x = jax.random.normal(jax.random.PRNGKey(9), (2, 12, 4), dtype=jnp.float32)
    model = SpikeLite(12, 5, 4, key=jax.random.PRNGKey(10))
    y = model(x)
    assert y.shape == (2, 5, 4)
    assert jnp.all(jnp.isfinite(y))


def test_ssca_changes_output():
    key = jax.random.PRNGKey(11)
    x = jax.random.normal(jax.random.PRNGKey(12), (2, 12, 4), dtype=jnp.float32)
    common = dict(
        d_model=16,
        n_branches=2,
        n_steps=2,
        n_heads=2,
        mask_rank=4,
    )
    with_ssca = SpikeLite(12, 5, 4, key=key, use_ssca=True, **common)
    without = SpikeLite(12, 5, 4, key=key, use_ssca=False, **common)
    y_on = with_ssca(x)
    y_off = without(x)
    assert jnp.all(jnp.isfinite(y_on))
    assert jnp.all(jnp.isfinite(y_off))
    assert not jnp.allclose(y_on, y_off)


def test_single_channel_forward():
    x = jax.random.normal(jax.random.PRNGKey(13), (2, 8, 1), dtype=jnp.float32)
    model = SpikeLite(
        8,
        3,
        1,
        key=jax.random.PRNGKey(14),
        d_model=8,
        n_branches=2,
        n_steps=2,
        n_heads=2,
        mask_rank=2,
        use_ssca=True,
    )
    y = model(x)
    assert y.shape == (2, 3, 1)
    assert jnp.all(jnp.isfinite(y))
    mask = model.channel_mask(jnp.zeros((2, 1, 8), dtype=jnp.float32))
    assert mask.shape == (2, 1, 1)
    assert jnp.allclose(mask, 1.0)


def _sum_of_sines(n, length, horizon, n_channels, key):
    """Sliding windows of a multivariate sum of sines, plus a slow shared rise."""
    k_phase, k_noise = jax.random.split(key)
    total = length + horizon
    t = jnp.arange(total, dtype=jnp.float32)
    freqs = jnp.array([0.07, 0.17, 0.31], dtype=jnp.float32)
    channel_phase = jnp.linspace(0.0, 1.2, n_channels, dtype=jnp.float32)
    phase = jax.random.uniform(k_phase, (n, 1, 1), dtype=jnp.float32) * 0.4
    tt = t[None, :, None]
    series = jnp.zeros((n, total, n_channels), dtype=jnp.float32)
    for i, freq in enumerate(freqs):
        amp = 1.0 / (i + 1)
        series = series + amp * jnp.sin(
            2.0 * jnp.pi * freq * tt + phase + channel_phase[None, None, :]
        )
    series = series + 0.08 * tt
    noise = 0.01 * jax.random.normal(k_noise, series.shape, dtype=jnp.float32)
    series = series + noise
    return series[:, :length], series[:, length:]


def test_adam_reduces_mse():
    key = jax.random.PRNGKey(15)
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
    x, y = _sum_of_sines(32, lookback, horizon, n_channels, k_data)
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
    initial_value = float(initial)
    for _ in range(24):
        model, opt_state, loss = step(model, opt_state, x, y)
    final_value = float(loss)
    assert jnp.isfinite(initial_value)
    assert jnp.isfinite(final_value)
    assert final_value < initial_value
