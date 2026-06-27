"""Propriétés du noyau de Peskin (F2). Ce sont des invariants mathématiques du
delta 4-points, pas des tolérances arbitraires : ils doivent passer net."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cascade.coupling import peskin


@pytest.mark.parametrize("x", np.linspace(0.0, 1.0, 11))
def test_phi4_partition_of_unity(x):
    # Σ_i φ(i - x) = 1 sur tout entier i couvrant le support
    i = jnp.arange(-3, 5, dtype=jnp.float64)
    s = jnp.sum(peskin.phi4(i - x))
    assert abs(float(s) - 1.0) < 1e-12


@pytest.mark.parametrize("x", np.linspace(0.0, 1.0, 11))
def test_phi4_first_moment(x):
    # Σ_i (i - x) φ(i - x) = 0 (reproduit les champs linéaires)
    i = jnp.arange(-3, 5, dtype=jnp.float64)
    m1 = jnp.sum((i - x) * peskin.phi4(i - x))
    assert abs(float(m1)) < 1e-12


def test_interpolate_constant():
    nx, ny = 32, 24
    field = jnp.full((1, nx, ny), 2.5, dtype=jnp.float64)
    markers = jnp.array([[8.3, 10.7], [15.0, 5.5], [20.9, 18.2]], dtype=jnp.float64)
    out = peskin.interpolate(field, markers)
    assert np.allclose(np.asarray(out), 2.5, atol=1e-12)


def test_interpolate_linear_exact():
    # champ linéaire f(x,y) = 3x - 2y + 1 ; Peskin doit le reproduire exactement
    nx, ny = 40, 40
    xs = jnp.arange(nx, dtype=jnp.float64)
    ys = jnp.arange(ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    field = (3.0 * X - 2.0 * Y + 1.0)[None]  # (1, nx, ny)
    markers = jnp.array([[12.34, 21.78], [25.5, 9.1]], dtype=jnp.float64)
    out = peskin.interpolate(field, markers)[:, 0]
    exact = 3.0 * markers[:, 0] - 2.0 * markers[:, 1] + 1.0
    assert np.allclose(np.asarray(out), np.asarray(exact), atol=1e-10)


def test_spread_conserves_total():
    # Σ_grid spread(F) = Σ_m F · ds  (adjoint de l'interp à Σδ=1)
    nx, ny = 48, 48
    markers = jnp.array([[10.2, 12.9], [30.7, 20.1], [25.0, 25.0]], dtype=jnp.float64)
    values = jnp.array([[1.0, -0.5], [2.0, 0.3], [-1.0, 0.7]], dtype=jnp.float64)  # (M,2)
    ds = 0.75
    g = peskin.spread(values, markers, (nx, ny), ds=ds)
    total_grid = jnp.sum(g, axis=(1, 2))
    total_lag = jnp.sum(values, axis=0) * ds
    assert np.allclose(np.asarray(total_grid), np.asarray(total_lag), atol=1e-10)


def test_spread_interpolate_adjoint():
    # <spread(F), u> = <F, interp(u)> · ds  (adjointness)
    key = jax.random.PRNGKey(0)
    nx, ny = 40, 36
    u = jax.random.normal(key, (2, nx, ny), dtype=jnp.float64)
    markers = jnp.array([[8.1, 9.4], [22.6, 18.3], [30.2, 25.7], [12.0, 30.0]], dtype=jnp.float64)
    F = jax.random.normal(jax.random.PRNGKey(1), (4, 2), dtype=jnp.float64)
    ds = 1.0
    lhs = jnp.sum(peskin.spread(F, markers, (nx, ny), ds=ds) * u)
    rhs = jnp.sum(F * peskin.interpolate(u, markers)) * ds
    assert abs(float(lhs) - float(rhs)) < 1e-9


def test_differentiable_wrt_markers():
    nx, ny = 40, 40
    xs = jnp.arange(nx, dtype=jnp.float64)
    ys = jnp.arange(ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    field = jnp.sin(0.2 * X) * jnp.cos(0.15 * Y)
    field = field[None]

    def loss(markers):
        return jnp.sum(peskin.interpolate(field, markers) ** 2)

    markers = jnp.array([[15.3, 20.7], [25.1, 10.4]], dtype=jnp.float64)
    g = jax.grad(loss)(markers)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.linalg.norm(g)) > 0.0
