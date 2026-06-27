"""Définition PHYSIQUE des régimes — indépendante du détail de Harten (ANTI-CIRCULARITÉ).

Feedback critique : si les régimes étaient définis par des bandes de détail de Harten, le
routeur nul (seuil de détail) serait par construction un classificateur de régime parfait
→ C2 (« routeur appris bat le nul ») deviendrait invérifiable (cible = baseline). Les régimes
sont donc définis par la VORTICITÉ / le critère-Q + le marqueur de structure — des grandeurs
que le routeur NE VOIT PAS (il ne voit que le halo grossier + le détail). Ainsi « le détail
prédit-il le régime ? » redevient empirique, et C2 redevient un vrai test ; le routeur appris
peut battre le nul précisément là où détail et vorticité divergent.

Régimes :
- CONVECTIVE : |ω| sous seuil, hors interface (écoulement lisse/convectif).
- SHEAR      : |ω| au-dessus du seuil OU Q>0, hors interface (cisaillement/lâcher, vortical).
- INTERFACE  : proximité de la structure (marqueur de structure).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

CONVECTIVE = 0
SHEAR = 1
INTERFACE = 2


def _ddx(a):  # dérivée centrée selon x (axe 0), bords par roll (intérieur de domaine visé)
    return 0.5 * (jnp.roll(a, -1, axis=0) - jnp.roll(a, 1, axis=0))


def _ddy(a):
    return 0.5 * (jnp.roll(a, -1, axis=1) - jnp.roll(a, 1, axis=1))


def vorticity(u: jax.Array) -> jax.Array:
    """ω = ∂v/∂x − ∂u/∂y. u (2, nx, ny) -> (nx, ny)."""
    return _ddx(u[1]) - _ddy(u[0])


def q_criterion(u: jax.Array) -> jax.Array:
    """Critère-Q 2D : Q = ½(‖Ω‖² − ‖S‖²). Q>0 = rotation domine (cœur de vortex)."""
    dudx, dudy = _ddx(u[0]), _ddy(u[0])
    dvdx, dvdy = _ddx(u[1]), _ddy(u[1])
    omega = dvdx - dudy
    Sxx, Syy, Sxy = dudx, dvdy, 0.5 * (dudy + dvdx)
    normO2 = 0.5 * omega ** 2
    normS2 = Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2
    return 0.5 * (normO2 - normS2)


def structure_mask(shape, center, radius: float, margin: float = 2.0) -> jax.Array:
    """Cellules dans le corps + marge (la région d'interface). center=(cx,cy)."""
    nx, ny = shape
    xs = jnp.arange(nx)[:, None]
    ys = jnp.arange(ny)[None, :]
    r2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    return r2 <= (radius + margin) ** 2


def calibrate_omega_threshold(u: jax.Array, center, radius: float, q=0.85, margin=2.0) -> float:
    """Seuil de vorticité physique = quantile q de |ω| HORS interface. Sépare le sillage
    vortical (queue haute) de la zone convective (cœur bas) sans présupposer la valeur."""
    omega = jnp.abs(vorticity(u))
    interface = structure_mask(u.shape[1:], center, radius, margin)
    vals = jnp.where(interface, jnp.nan, omega)
    return float(jnp.nanquantile(vals, q))


def label_regimes(u: jax.Array, center, radius: float, omega_threshold: float,
                  margin: float = 2.0, use_q: bool = True) -> jax.Array:
    """Étiquette de régime par cellule, depuis le champ FIN (vérité-terrain). (nx,ny) int.

    Indépendant du détail de Harten : le routeur ne voit jamais ces grandeurs.
    """
    omega = vorticity(u)
    interface = structure_mask(u.shape[1:], center, radius, margin)
    vortical = jnp.abs(omega) >= omega_threshold
    if use_q:
        vortical = vortical | (q_criterion(u) > 0.0)
    shear = vortical & (~interface)
    return jnp.where(interface, INTERFACE, jnp.where(shear, SHEAR, CONVECTIVE)).astype(jnp.int32)
