"""Noyau de frontière immergée (Peskin) — spread / interpolate différentiables en JAX.

Convention de coordonnées (cf. layout XLB) :
- champ eulérien : ``(C, nx, ny)`` ; axe 1 = x, axe 2 = y ; centres de cellules aux
  entiers, pas de grille dx = 1 (unités réseau LBM).
- marqueurs lagrangiens : ``(M, 2)`` en coordonnées continues (x, y).

Le delta régularisé 4-points de Peskin (support |r| < 2) satisfait les conditions
de moments Σ δ = 1 et Σ (x_i − x) δ = 0 — il reproduit donc exactement les champs
constants et linéaires. ``interpolate`` et ``spread`` sont adjoints (à ds près),
ce qui garantit la conservation de quantité de mouvement à l'interface (le « bus »).

Référence : Peskin, *The immersed boundary method*, Acta Numerica (2002).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# rayon de support du delta 4-points (cellules de chaque côté)
HALF_SUPPORT = 2


def phi4(r: jax.Array) -> jax.Array:
    """Delta régularisé 4-points de Peskin, évalué en ``r`` (distance signée).

    Lisse par morceaux → différentiable presque partout (les raccords à |r|=1,2
    sont C¹). Renvoie 0 hors de [-2, 2].
    """
    a = jnp.abs(r)
    # Safe-sqrt (double-where) : l'argument passé à sqrt est toujours > 0, sinon le
    # backward produit 0*inf = NaN hors support. Dans le support les arguments valent
    # >= 1, donc le clamp n'altère jamais une valeur réellement utilisée.
    g1 = 1.0 + 4.0 * a - 4.0 * a * a
    s1 = jnp.where(g1 > 0.0, jnp.sqrt(jnp.where(g1 > 0.0, g1, 1.0)), 0.0)
    b1 = (3.0 - 2.0 * a + s1) / 8.0
    g2 = -7.0 + 12.0 * a - 4.0 * a * a
    s2 = jnp.where(g2 > 0.0, jnp.sqrt(jnp.where(g2 > 0.0, g2, 1.0)), 0.0)
    b2 = (5.0 - 2.0 * a - s2) / 8.0
    out = jnp.where(a <= 1.0, b1, jnp.where(a <= 2.0, b2, 0.0))
    return out


def _marker_stencil(pos: jax.Array, n: int):
    """Pour un marqueur 1D en ``pos`` sur un axe de taille ``n`` : (indices(4,), poids(4,)).

    Indices ramenés dans [0, n) par clip (les marqueurs restent intérieurs au domaine ;
    le clip est une garde de bord, pas un wrap périodique).
    """
    base = jnp.floor(pos).astype(jnp.int32) - 1  # début du stencil 4 points
    offs = jnp.arange(4, dtype=jnp.int32)
    idx = base[..., None] + offs                 # (..., 4)
    w = phi4(idx.astype(pos.dtype) - pos[..., None])  # poids AVANT clip (positions vraies)
    idx = jnp.clip(idx, 0, n - 1)
    return idx, w


def _stencils_2d(markers: jax.Array, nx: int, ny: int):
    """(M,2) -> (ix (M,4), iy (M,4), W (M,4,4)) : stencils 4x4 par marqueur."""
    ix, wx = _marker_stencil(markers[:, 0], nx)   # (M,4),(M,4)
    iy, wy = _marker_stencil(markers[:, 1], ny)
    W = wx[:, :, None] * wy[:, None, :]           # (M,4,4)
    return ix, iy, W


def interpolate(field: jax.Array, markers: jax.Array) -> jax.Array:
    """Interpole un champ eulérien ``(C, nx, ny)`` aux marqueurs ``(M, 2)`` -> ``(M, C)``.

    U_L(m) = Σ_{i,j} field(i,j) · δ(i − x_m) · δ(j − y_m).
    """
    C, nx, ny = field.shape
    ix, iy, W = _stencils_2d(markers, nx, ny)     # (M,4),(M,4),(M,4,4)
    # gather des blocs 4x4 par marqueur, pour chaque composante
    # field[c, ix[:,a], iy[:,b]] -> (C, M, 4, 4)
    stomp = field[:, ix[:, :, None], iy[:, None, :]]   # (C, M, 4, 4)
    out = jnp.einsum("cmab,mab->mc", stomp, W)
    return out


def spread(values: jax.Array, markers: jax.Array, shape, ds: jax.Array | float = 1.0) -> jax.Array:
    """Étale des valeurs lagrangiennes ``(M, C)`` vers un champ eulérien ``(C, nx, ny)``.

    g(i,j) = Σ_m values(m) · δ(i − x_m) · δ(j − y_m) · ds_m.

    Adjoint de :func:`interpolate` (à ds près) → Σ_grid g = Σ_m values·ds (conservation).
    """
    nx, ny = shape
    M, C = values.shape
    ix, iy, W = _stencils_2d(markers, nx, ny)     # (M,4),(M,4),(M,4,4)
    ds_arr = jnp.broadcast_to(jnp.asarray(ds, values.dtype), (M,))
    # contribution (M,C,4,4)
    contrib = values[:, :, None, None] * W[:, None, :, :] * ds_arr[:, None, None, None]
    # indices plats dans (nx*ny) pour scatter-add
    flat_idx = (ix[:, :, None] * ny + iy[:, None, :]).reshape(M, 16)  # (M,16)
    contrib_flat = contrib.reshape(M, C, 16)
    g = jnp.zeros((C, nx * ny), values.dtype)
    g = g.at[:, flat_idx.reshape(-1)].add(
        jnp.transpose(contrib_flat, (1, 0, 2)).reshape(C, M * 16)
    )
    return g.reshape(C, nx, ny)
