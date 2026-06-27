"""Analyse multirésolution de Harten en JAX (différentiable).

Convention : champs ``(C, nx, ny)`` (pile de features : densité, qdm vectorielle,
déformation scalaire, marqueur d'interface…). Coarsening 2×2 par moyenne de cellule.
Prédiction = injection (broadcast) de la moyenne grossière ; détail = fine − prédiction.

Le détail à 4 valeurs par bloc 2×2 est de somme nulle (la moyenne est leur centre) →
exactement (coarse, détail) reconstruit le fin : MRA inversible (cf. Harten 1995).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def coarsen(f: jax.Array) -> jax.Array:
    """Moyenne de cellule 2×2 : (C, nx, ny) -> (C, nx//2, ny//2). nx, ny pairs."""
    C, nx, ny = f.shape
    return f.reshape(C, nx // 2, 2, ny // 2, 2).mean(axis=(2, 4))


def upsample(c: jax.Array) -> jax.Array:
    """Injection (broadcast) d'un champ grossier vers le fin : chaque cellule -> bloc 2×2."""
    C, hx, hy = c.shape
    return jnp.broadcast_to(c[:, :, None, :, None], (C, hx, 2, hy, 2)).reshape(C, hx * 2, hy * 2)


def detail(f: jax.Array):
    """(coarse, det) : coarse = moyenne 2×2 ; det = f − upsample(coarse) (somme nulle/bloc)."""
    c = coarsen(f)
    return c, f - upsample(c)


def reconstruct(coarse: jax.Array, det: jax.Array) -> jax.Array:
    """Inverse exact de :func:`detail` : f = upsample(coarse) + det."""
    return upsample(coarse) + det


def detail_magnitude(det: jax.Array) -> jax.Array:
    """Coefficient de détail scalaire par cellule grossière : norme L2 des déviations du
    bloc 2×2, agrégée sur les canaux. (C, nx, ny) -> (nx//2, ny//2).

    C'est l'indicateur de raffinement de Harten : ~0 en zone lisse, grand aux fronts.
    """
    C, nx, ny = det.shape
    blocks = det.reshape(C, nx // 2, 2, ny // 2, 2)
    return jnp.sqrt(jnp.sum(blocks ** 2, axis=(0, 2, 4)))


def decompose(f: jax.Array, levels: int = 2):
    """MRA sur ``levels`` niveaux. Renvoie (coarsest, [det_1, …, det_levels]) où det_1 est
    le détail le plus fin. ``reconstruct`` itéré inverse exactement."""
    dets = []
    cur = f
    for _ in range(levels):
        c, d = detail(cur)
        dets.append(d)
        cur = c
    return cur, dets


def recompose(coarsest: jax.Array, dets) -> jax.Array:
    """Inverse de :func:`decompose`."""
    cur = coarsest
    for d in reversed(dets):
        cur = reconstruct(cur, d)
    return cur


def window_features(f: jax.Array, levels: int = 2) -> jax.Array:
    """Features de routage par fenêtre (= cellule au niveau le plus grossier), sur le HALO
    GROSSIER seul (§6). Empile, par fenêtre : la valeur grossière de chaque canal + le
    coefficient de détail multirésolution agrégé sur la fenêtre.

    Renvoie ``(F, wx, wy)`` : F = C (halo grossier) + 1 (détail), wx,wy = grille grossière.
    """
    coarsest, dets = decompose(f, levels)            # coarsest: (C, wx, wy)
    C, wx, wy = coarsest.shape
    # détail par fenêtre : on agrège le coefficient de détail de chaque niveau jusqu'à la
    # résolution de la fenêtre (max sur les niveaux = « y a-t-il de la structure sous-grille »).
    dmag_w = jnp.zeros((wx, wy))
    for lvl, d in enumerate(dets):
        dm = detail_magnitude(d)                     # (nx/2^(lvl+1), ny/2^(lvl+1))
        # ramener à la grille fenêtre par coarsening de la magnitude
        k = dm.shape[0] // wx
        if k > 1:
            dm = dm.reshape(wx, k, wy, k).max(axis=(1, 3))
        dmag_w = jnp.maximum(dmag_w, dm)
    return jnp.concatenate([coarsest, dmag_w[None]], axis=0)
