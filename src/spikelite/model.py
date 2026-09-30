"""SpikeLite kernel.

Readings used where the paper names a step but gives no formula:

- RevIN is standard instance norm over the time axis, per channel, with affine
  gamma and beta. The forecast is mapped back with the same mean, variance, and
  inverse affine.
- Component norm is per-component, per-channel standardization over time, with
  no affine, applied before the linear map. The telescoping sum is on the
  pre-norm components.
- Spikes use a sigmoid surrogate with scale 4. Forward values are hard
  thresholds. Backward values follow the sigmoid.
- Spike replication is one scalar-tau LIF. Its input current is Z at every
  step, with the same delayed subtractive reset as FSSE.
- Q, K, and V are a linear map followed by LayerNorm over the feature axis.
  The figure shows batch norm. LayerNorm avoids batch-stat state.
- Horizon scale is a learnable vector of length P, applied after the shared
  linear map and before RevIN denormalization.
- Branch decays start on linspace(0.9, 0.3, K), mapped by a logit, copied to
  every channel. The paper only says the start is descending. Thresholds start
  at 1.0. The paper does not give that start. Differences follow branch index
  order. Branches are not re-sorted.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp


KAPPA = 0.125
_EPS = 1e-5
_DIST_EPS = 1e-6


def _logit(p):
    p = jnp.asarray(p, dtype=jnp.float32)
    return jnp.log(p) - jnp.log1p(-p)


def heaviside(x, scale):
    """Sigmoid surrogate for the Heaviside step.

    Forward value is 1 when x > 0 and 0 otherwise. The backward pass follows
    sigmoid(scale * x).
    """
    hard = (x > 0).astype(x.dtype)
    soft = jax.nn.sigmoid(scale * x)
    return jax.lax.stop_gradient(hard - soft) + soft


def masked_attention(q, k, v, mask, kappa):
    """Softmax-free masked attention over the channel axis.

    q, k, v have shape (B, H, C, d). mask is (B, C, C) or (C, C).
    Returns (B, H, C, d). A = kappa * (Q K^T) * M, then O = A V.
    """
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    if mask.ndim == 2:
        mask_b = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask_b = mask[:, None, :, :]
    else:
        raise ValueError("mask must have shape (C, C) or (B, C, C)")
    attended = jnp.matmul(kappa * scores * mask_b, v)
    return attended


def _time_standardize(components):
    """Per component, per channel, standardize over time. No affine."""
    mean = jnp.mean(components, axis=2, keepdims=True)
    var = jnp.var(components, axis=2, keepdims=True)
    return (components - mean) / jnp.sqrt(var + _EPS)


class SpikeLite(eqx.Module):
    """Multivariate forecaster: RevIN, FSSE, optional SSCA, linear head."""

    lookback: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    n_branches: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    mask_rank: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    eta: float = eqx.field(static=True)
    surrogate_scale: float = eqx.field(static=True)
    use_ssca: bool = eqx.field(static=True)
    kappa: float = eqx.field(static=True)

    rho: jax.Array
    theta: jax.Array
    proj_weight: jax.Array
    proj_bias: jax.Array
    alpha: jax.Array
    revin_gamma: jax.Array
    revin_beta: jax.Array
    W_r: jax.Array
    lin_q: eqx.nn.Linear
    lin_k: eqx.nn.Linear
    lin_v: eqx.nn.Linear
    ln_q: eqx.nn.LayerNorm
    ln_k: eqx.nn.LayerNorm
    ln_v: eqx.nn.LayerNorm
    lin_out: eqx.nn.Linear
    rep_rho: jax.Array
    rep_theta: jax.Array
    head: eqx.nn.Linear
    horizon_scale: jax.Array

    def __init__(
        self,
        lookback,
        horizon,
        n_channels,
        *,
        key,
        d_model=128,
        n_branches=7,
        n_steps=4,
        n_heads=4,
        mask_rank=8,
        gamma=1.0,
        eta=0.3,
        surrogate_scale=4.0,
        use_ssca=True,
    ):
        if use_ssca and d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if n_branches < 1:
            raise ValueError("n_branches must be at least 1")

        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)
        self.n_branches = int(n_branches)
        self.n_steps = int(n_steps)
        self.n_heads = int(n_heads)
        self.mask_rank = int(mask_rank)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.surrogate_scale = float(surrogate_scale)
        self.use_ssca = bool(use_ssca)
        self.kappa = KAPPA

        k_proj, k_bias, k_q, k_k, k_v, k_out, k_wr, k_head = jax.random.split(key, 8)

        # Branch 0 is the slowest init. Nominal index order is kept.
        tau0 = jnp.linspace(0.9, 0.3, self.n_branches, dtype=jnp.float32)
        rho0 = _logit(tau0)
        self.rho = jnp.repeat(rho0[:, None], self.n_channels, axis=1)
        self.theta = jnp.ones((self.n_branches, self.n_channels), dtype=jnp.float32)

        n_comp = self.n_branches + 1
        lim = 1.0 / math.sqrt(self.lookback)
        # W^(j) in R^{D x L}, shared across channels. b^(j) in R^{D}.
        self.proj_weight = jax.random.uniform(
            k_proj,
            (n_comp, self.d_model, self.lookback),
            minval=-lim,
            maxval=lim,
            dtype=jnp.float32,
        )
        self.proj_bias = jax.random.uniform(
            k_bias,
            (n_comp, self.d_model),
            minval=-lim,
            maxval=lim,
            dtype=jnp.float32,
        )
        self.alpha = jnp.ones((n_comp,), dtype=jnp.float32)

        self.revin_gamma = jnp.ones((self.n_channels,), dtype=jnp.float32)
        self.revin_beta = jnp.zeros((self.n_channels,), dtype=jnp.float32)

        n_freq = self.d_model // 2 + 1
        wr_lim = 1.0 / math.sqrt(n_freq)
        self.W_r = jax.random.uniform(
            k_wr,
            (self.mask_rank, n_freq),
            minval=-wr_lim,
            maxval=wr_lim,
            dtype=jnp.float32,
        )

        self.lin_q = eqx.nn.Linear(self.d_model, self.d_model, key=k_q)
        self.lin_k = eqx.nn.Linear(self.d_model, self.d_model, key=k_k)
        self.lin_v = eqx.nn.Linear(self.d_model, self.d_model, key=k_v)
        self.ln_q = eqx.nn.LayerNorm(self.d_model)
        self.ln_k = eqx.nn.LayerNorm(self.d_model)
        self.ln_v = eqx.nn.LayerNorm(self.d_model)
        self.lin_out = eqx.nn.Linear(self.d_model, self.d_model, key=k_out)

        # Scalar replication neuron. tau init 0.5 via logit, theta init 0.5.
        self.rep_rho = jnp.asarray(_logit(0.5), dtype=jnp.float32)
        self.rep_theta = jnp.asarray(0.5, dtype=jnp.float32)

        self.head = eqx.nn.Linear(self.d_model, self.horizon, key=k_head)
        self.horizon_scale = jnp.ones((self.horizon,), dtype=jnp.float32)

    def _lif(self, x):
        """Delayed-reset LIF on x of shape (B, L, C).

        Returns gates G and spikes S, each (K, B, L, C).
        """
        tau = jax.nn.sigmoid(self.rho)  # (K, C)
        theta = self.theta
        x_lbc = jnp.swapaxes(x, 0, 1)  # (L, B, C)
        scale = self.surrogate_scale

        def one_branch(tau_c, theta_c):
            def step(carry, x_l):
                u_prev, s_prev = carry
                u = tau_c * u_prev + x_l - theta_c * s_prev
                s = heaviside(u - theta_c, scale)
                g = u * s
                return (u, s), (g, s)

            zeros = jnp.zeros_like(x_lbc[0])
            _, (g, s) = jax.lax.scan(step, (zeros, zeros), x_lbc)
            return g, s

        g, s = jax.vmap(one_branch)(tau, theta)  # (K, L, B, C)
        g = jnp.transpose(g, (0, 2, 1, 3))
        s = jnp.transpose(s, (0, 2, 1, 3))
        return g, s

    def _components_from_gates(self, gates, x):
        """0-based successive differences. Sums to x.

        B0 = G0, B_j = G_j - G_{j-1} for j = 1..K-1, B_K = x - G_{K-1}.
        This matches the paper's 1-based Eq. 7 after shifting the branch index.
        """
        diffs = gates[1:] - gates[:-1]
        residual = x - gates[-1]
        return jnp.concatenate([gates[:1], diffs, residual[None]], axis=0)

    def fsse_components(self, x):
        """Pre-norm FSSE components of x, shape (K+1, B, L, C). Sums to x."""
        gates, _ = self._lif(x)
        return self._components_from_gates(gates, x)

    def fsse_spikes(self, x):
        """Branch spikes, shape (K, B, L, C)."""
        _, spikes = self._lif(x)
        return spikes

    def _encode(self, x):
        components = self.fsse_components(x)
        normed = _time_standardize(components)
        # z_c^(j) = W^(j) @ B_norm[:, c] + b^(j)
        z = jnp.einsum("jdl,jblc->jbcd", self.proj_weight, normed)
        z = z + self.proj_bias[:, None, None, :]
        return jnp.einsum("j,jbcd->bcd", self.alpha, z)

    def _revin_norm(self, x):
        mean = jnp.mean(x, axis=1, keepdims=True)
        var = jnp.var(x, axis=1, keepdims=True)
        std = jnp.sqrt(var + _EPS)
        x_hat = self.revin_gamma * (x - mean) / std + self.revin_beta
        return x_hat, mean, std

    def _revin_denorm(self, y, mean, std):
        y_hat = (y - self.revin_beta) / self.revin_gamma
        return y_hat * std + mean

    def channel_mask(self, z):
        """Sample-dependent binary channel mask, shape (B, C, C).

        Forward values are in {0, 1}. Off-diagonal backward values follow p.
        """
        batch, n_channels, _ = z.shape
        if n_channels == 1:
            return jnp.ones((batch, 1, 1), dtype=z.dtype)

        spectrum = jnp.abs(jnp.fft.rfft(z, axis=-1))
        relation = jnp.einsum("rf,bcf->bcr", self.W_r, spectrum)
        diff = relation[:, :, None, :] - relation[:, None, :, :]
        dist = jnp.sum(diff * diff, axis=-1) + _DIST_EPS
        affinity = 1.0 / dist

        eye = jnp.eye(n_channels, dtype=bool)
        masked = jnp.where(eye, jnp.asarray(-jnp.inf, dtype=z.dtype), affinity)
        max_affinity = jnp.max(masked, axis=-1, keepdims=True)
        p = jnp.clip(self.gamma * affinity / max_affinity, 0.0, 1.0)
        binary = jnp.where(p > self.eta, 1.0, 0.0)
        binary = jnp.where(eye, 1.0, binary)
        ste = jax.lax.stop_gradient(binary - p) + p
        return jnp.where(eye, 1.0, ste)

    def _replicate_spikes(self, z):
        """Repeat Z as the input current of one scalar LIF for Ts steps."""
        tau = jax.nn.sigmoid(self.rep_rho)
        theta = self.rep_theta
        scale = self.surrogate_scale

        def step(carry, _):
            u_prev, s_prev = carry
            u = tau * u_prev + z - theta * s_prev
            s = heaviside(u - theta, scale)
            return (u, s), s

        zeros = jnp.zeros_like(z)
        _, spikes = jax.lax.scan(step, (zeros, zeros), None, length=self.n_steps)
        return spikes

    def _split_heads(self, tensor):
        batch, n_channels, dim = tensor.shape
        n_heads = self.n_heads
        width = dim // n_heads
        tensor = tensor.reshape(batch, n_channels, n_heads, width)
        return jnp.transpose(tensor, (0, 2, 1, 3))

    def _merge_heads(self, tensor):
        batch, n_heads, n_channels, width = tensor.shape
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        return tensor.reshape(batch, n_channels, n_heads * width)

    def _project_qkv(self, spikes_t):
        def one(vec):
            q = self.ln_q(self.lin_q(vec))
            k = self.ln_k(self.lin_k(vec))
            v = self.ln_v(self.lin_v(vec))
            return q, k, v

        return jax.vmap(jax.vmap(one))(spikes_t)

    def _ssca(self, z):
        mask = self.channel_mask(z)
        spikes = self._replicate_spikes(z)

        def one_step(spikes_t):
            q, k, v = self._project_qkv(spikes_t)
            attended = masked_attention(
                self._split_heads(q),
                self._split_heads(k),
                self._split_heads(v),
                mask,
                self.kappa,
            )
            merged = self._merge_heads(attended)
            return jax.vmap(jax.vmap(self.lin_out))(merged)

        outputs = jax.vmap(one_step)(spikes)
        return jnp.sum(outputs, axis=0)

    def _predict(self, z):
        y = jax.vmap(jax.vmap(self.head))(z)  # (B, C, P)
        y = jnp.swapaxes(y, 1, 2)  # (B, P, C)
        return y * self.horizon_scale[None, :, None]

    def __call__(self, x):
        """Forecast. x is (B, L, C), returns (B, P, C) in the original scale."""
        x_hat, mean, std = self._revin_norm(x)
        z = self._encode(x_hat)
        if self.use_ssca:
            z = self._ssca(z)
        y = self._predict(z)
        return self._revin_denorm(y, mean, std)
