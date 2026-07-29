"""Invariants du pas couplé IB (`cascade/coupling/ib_lbm.py`) — sans XLB ni GPU.

`_ib_force` porte le Newton 3 discret du harnais : la réaction sur le corps est
`−Σ F_L·ds`, et la source eulérienne `g` doit injecter EXACTEMENT cette quantité
de mouvement dans le fluide. C'est un invariant mathématique du couple
interpolate/spread (partition de l'unité de φ4), pas une tolérance arbitraire —
il était vérifié analytiquement juste mais jamais testé (review 28/07)."""

import types

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from cascade.coupling.ib_lbm import _ib_force, make_cylinder


def _coupled_minimal(nx=64, ny=48, n_iters=3):
    """`_ib_force` ne touche que `body`, `fluid.cfg.nx/ny` et
    `n_forcing_iters` : un SimpleNamespace suffit — pas de XLB, pas de GPU."""
    body = make_cylinder(cx=nx // 2, cy=ny / 2, radius=6.0, fill=True)
    return types.SimpleNamespace(
        body=body,
        fluid=types.SimpleNamespace(cfg=types.SimpleNamespace(nx=nx, ny=ny)),
        n_forcing_iters=n_iters,
    ), body


def test_ib_force_conserve_la_quantite_de_mouvement():
    """Σ_grille g == Σ_L F_L·ds == −F_hydro, composante par composante.

    Si `spread` perdait de la masse de forçage (support tronqué au bord,
    normalisation cassée), le fluide et le corps ne verraient pas des forces
    opposées : dérive de quantité de mouvement silencieuse dans chaque pas."""
    coupled, body = _coupled_minimal()
    nx, ny = 64, 48
    yy, xx = jnp.meshgrid(jnp.arange(ny, dtype=jnp.float64),
                          jnp.arange(nx, dtype=jnp.float64))
    rho = jnp.ones((1, nx, ny), dtype=jnp.float64)
    u = jnp.stack([0.05 * jnp.sin(2.0 * jnp.pi * yy / ny),
                   0.02 * jnp.cos(2.0 * jnp.pi * xx / nx)])
    g_total, F_hydro = _ib_force(coupled, rho, u, body.center0,
                                 jnp.array([0.0, 0.1], jnp.float64))
    injecte = jnp.sum(g_total, axis=(1, 2))
    np.testing.assert_allclose(np.asarray(injecte), -np.asarray(F_hydro),
                               rtol=0, atol=1e-10)


def test_ib_force_nulle_si_le_fluide_suit_le_corps():
    """Cas dégénéré exact : u == U_corps partout ⇒ F_L = 0 à chaque itération
    ⇒ g ≡ 0 et F_hydro ≡ 0 (aucune force fabriquée de nulle part)."""
    coupled, body = _coupled_minimal()
    vel = jnp.array([0.03, -0.01], jnp.float64)
    rho = jnp.ones((1, 64, 48), dtype=jnp.float64)
    u = jnp.broadcast_to(vel[:, None, None], (2, 64, 48))
    g_total, F_hydro = _ib_force(coupled, rho, u, body.center0, vel)
    assert float(jnp.max(jnp.abs(g_total))) == 0.0
    assert float(jnp.max(jnp.abs(F_hydro))) == 0.0
