"""Invariants de la MRA de Harten (F3). Propriétés mathématiques, pas tolérances arbitraires."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cascade.harten import windowing as H


def _field(nx=32, ny=24, C=3, key=0):
    return jax.random.normal(jax.random.PRNGKey(key), (C, nx, ny), dtype=jnp.float64)


def test_reconstruction_exact():
    f = _field()
    c, d = H.detail(f)
    assert np.allclose(np.asarray(H.reconstruct(c, d)), np.asarray(f), atol=1e-12)


def test_decompose_recompose_exact():
    f = _field(nx=32, ny=32)
    coarsest, dets = H.decompose(f, levels=3)
    assert np.allclose(np.asarray(H.recompose(coarsest, dets)), np.asarray(f), atol=1e-12)


def test_coarsen_preserves_mean():
    f = _field()
    assert np.allclose(float(H.coarsen(f).mean()), float(f.mean()), atol=1e-12)


def test_detail_zero_for_constant():
    f = jnp.full((2, 16, 16), 3.7, dtype=jnp.float64)
    _, d = H.detail(f)
    assert float(jnp.abs(d).max()) < 1e-12
    assert float(H.detail_magnitude(d).max()) < 1e-12


def test_detail_flags_subgrid_structure():
    # structure sous-grille (blob étroit ~ cœur de vortex) vs champ lisse large : le détail
    # doit être bien plus grand sur la structure fine. (Le détail capte la variation INTRA-bloc ;
    # un saut aligné sur 2^k est invisible à ce niveau — propriété de la MRA dyadique.)
    xs = jnp.arange(32, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    sharp = jnp.exp(-((X - 13.0) ** 2 + (Y - 13.0) ** 2) / 2.0)[None]      # σ~1 cellule
    smooth = jnp.exp(-((X - 16.0) ** 2 + (Y - 16.0) ** 2) / 200.0)[None]   # large, lisse
    dm_sharp = float(H.detail_magnitude(H.detail(sharp)[1]).max())
    dm_smooth = float(H.detail_magnitude(H.detail(smooth)[1]).max())
    assert dm_sharp > 5 * dm_smooth   # structure sous-grille nettement plus détaillée


def test_detail_max_for_checkerboard():
    # oscillation par cellule (extrême sous-grille) -> détail maximal ; constant -> 0
    i = jnp.arange(16, dtype=jnp.float64)
    cb = ((-1.0) ** (i[:, None] + i[None, :]))[None]   # (1,16,16) damier
    dm_cb = float(H.detail_magnitude(H.detail(cb)[1]).max())
    assert dm_cb > 1.0   # déviations ±1 par cellule -> norme bloc = 2


def test_detail_sums_to_zero_per_block():
    # le détail d'un bloc 2×2 est de somme nulle (la moyenne est leur centre)
    f = _field(nx=8, ny=8, C=1)
    _, d = H.detail(f)
    blocks = np.asarray(d).reshape(1, 4, 2, 4, 2)
    block_sums = blocks.sum(axis=(2, 4))
    assert np.allclose(block_sums, 0.0, atol=1e-12)


def test_window_features_shape():
    f = _field(nx=64, ny=32, C=4)
    wf = H.window_features(f, levels=2)        # grille fenêtre = 64/4 × 32/4 = 16×8
    assert wf.shape == (4 + 1, 16, 8)          # C halo grossier + 1 détail


def test_differentiable():
    f = _field()
    def loss(x):
        return jnp.sum(H.window_features(x, levels=2) ** 2)
    g = jax.grad(loss)(f)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.linalg.norm(g)) > 0
