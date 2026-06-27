"""Pas couplé IB-LBM monolithique (§5) + EDO de structure (config ressort).

Schéma de forçage direct (direct-forcing IB) en exact-difference spatialement variable :

  1. (ρ, u) = macroscopic(f)                     champ prédit
  2. u_b    = interpolate(u, marqueurs)          vitesse fluide vue par l'interface
  3. F_L    = U_corps − u_b                       forçage direct (impose le non-glissement)
  4. g      = spread(F_L · ds)                    source de qdm eulérienne (le « bus »)
  5. f     += feq(ρ, u + g/ρ) − feq(ρ, u)        injection exact-difference (Kupershtokh)
  6. f      = pas fluide XLB (stream+collide+BC)
  7. F_hydro = −Σ F_L · ds  (Newton 3) → EDO ressort → nouvelle position de l'interface

Tout est en JAX pur → un seul graphe différentiable (levier §7 pour la Phase 2).
Le cylindre rigide se déplace sans remaillage (les marqueurs translatent).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from cascade.coupling.peskin import interpolate, spread
from cascade.fluid.lbm import Fluid


# ---------------------------------------------------------------------------
# Corps immergé : cylindre rigide discrétisé en marqueurs lagrangiens
# ---------------------------------------------------------------------------
@dataclass
class CylinderBody:
    base: jax.Array     # (M,2) marqueurs sur le cercle, centrés en 0
    center0: jax.Array  # (2,) centre au repos
    ds: float           # longueur d'arc par marqueur
    radius: float


def make_cylinder(cx: float, cy: float, radius: float, n_markers: int = 80,
                  fill: bool = True, spacing: float = 1.0) -> CylinderBody:
    """Cylindre rigide en marqueurs lagrangiens.

    fill=True (défaut) : marqueurs VOLUMIQUES tapissant le disque (espacement ~1 cellule),
    ``ds`` = aire par marqueur. Impose u=U_corps dans tout le solide → corps réellement
    imperméable (≠ coque-anneau creuse qui laisse circuler l'intérieur).
    fill=False : anneau frontière (ds = longueur d'arc) — conservé pour diagnostic.
    """
    if fill:
        rng = jnp.arange(-radius, radius + spacing, spacing, dtype=jnp.float32)
        gx, gy = jnp.meshgrid(rng, rng, indexing="ij")
        inside = (gx ** 2 + gy ** 2) <= radius ** 2
        base = jnp.stack([gx[inside], gy[inside]], axis=1)  # (M,2)
        ds = spacing ** 2
    else:
        theta = jnp.linspace(0.0, 2.0 * math.pi, n_markers, endpoint=False)
        base = jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=1)
        ds = 2.0 * math.pi * radius / n_markers
    return CylinderBody(base=base, center0=jnp.array([cx, cy], jnp.float32), ds=ds, radius=radius)


# ---------------------------------------------------------------------------
# Structure : oscillateur masse-ressort-amortisseur (1–2 DDL)
# ---------------------------------------------------------------------------
@dataclass
class Spring:
    m: float                 # masse (par unité d'envergure)
    k: float                 # raideur
    c: float                 # amortissement visqueux
    dof_mask: jax.Array      # (2,) : [inline, crossflow] actifs (1) ou bloqués (0)


def spring_from_viv(diameter, u_in, mass_ratio, reduced_velocity, zeta,
                    crossflow_only=True, rho_f=1.0) -> Spring:
    """Construit le ressort à partir des nombres VIV sans dimension.

    m*  = m / (ρ_f D²)        (rapport de masse, 2D)
    Ur  = U / (f_n D)         (vitesse réduite)  → f_n = U / (Ur D)
    ω_n = 2π f_n,  k = m ω_n², c = 2 ζ √(k m)
    Lock-in attendu quand f_n ≈ f_shed = St·U/D, i.e. Ur ≈ 1/St (~5 pour St≈0.2).
    """
    m = mass_ratio * rho_f * diameter ** 2
    f_n = u_in / (reduced_velocity * diameter)
    omega_n = 2.0 * math.pi * f_n
    k = m * omega_n ** 2
    c = 2.0 * zeta * math.sqrt(k * m)
    dof = jnp.array([0.0, 1.0] if crossflow_only else [1.0, 1.0], jnp.float32)
    return Spring(m=m, k=k, c=c, dof_mask=dof)


# ---------------------------------------------------------------------------
# Configuration couplée + pas + rollout
# ---------------------------------------------------------------------------
@dataclass
class Coupled:
    fluid: Fluid
    body: CylinderBody
    spring: Spring
    n_forcing_iters: int      # passes de forçage direct (multi-direct-forcing)
    probe_xy: tuple           # (px, py) sonde de sillage pour le Strouhal
    prescribed: object = None  # None => dynamique (EDO ressort) ; sinon fn(t)->(disp(2,),vel(2,))
                               # pour le test de masse ajoutée à amplitude IMPOSÉE.


def init_state(coupled: Coupled):
    return {
        "f": coupled.fluid.f_init,
        "f_buf": coupled.fluid.f_buf,
        "disp": jnp.zeros(2, jnp.float32),
        "vel": jnp.zeros(2, jnp.float32),
    }


def _ib_force(coupled: Coupled, rho, u, center, vel):
    """Calcule la source eulérienne g et la réaction sur le corps (forçage direct itéré)."""
    body = coupled.body
    markers = body.base + center            # (M,2)
    U_body = jnp.broadcast_to(vel, markers.shape)
    nx, ny = coupled.fluid.cfg.nx, coupled.fluid.cfg.ny
    g_total = jnp.zeros_like(u)
    F_sum = jnp.zeros(2, u.dtype)
    u_cur = u
    for _ in range(coupled.n_forcing_iters):
        u_b = interpolate(u_cur, markers)          # (M,2)
        F_L = U_body - u_b                          # (M,2)
        g = spread(F_L, markers, (nx, ny), ds=body.ds)  # (2,nx,ny)
        g_total = g_total + g
        F_sum = F_sum + jnp.sum(F_L, axis=0) * body.ds
        u_cur = u_cur + g / jnp.clip(rho, 1e-3)     # correction de vitesse pour l'itération
    return g_total, -F_sum                           # réaction = −Σ F_L ds (Newton 3)


def coupled_step(coupled: Coupled, state, t):
    fl = coupled.fluid
    f, f_buf = state["f"], state["f_buf"]
    # cinématique du corps : imposée (test masse ajoutée) ou dynamique (EDO ressort)
    if coupled.prescribed is not None:
        disp, vel = coupled.prescribed(t)
    else:
        disp, vel = state["disp"], state["vel"]
    rho, u = fl.macroscopic(f)
    center = coupled.body.center0 + disp
    g, F_hydro = _ib_force(coupled, rho, u, center, vel)
    # injection exact-difference dans f (avant le pas fluide)
    du = g / jnp.clip(rho, 1e-3)
    f_forced = f + (fl.equilibrium(rho, u + du) - fl.equilibrium(rho, u))
    # un pas fluide XLB (nouvel état en 2e position → swap)
    a, b = fl.step(f_forced, f_buf, t)
    f_new, f_buf_new = b, a
    if coupled.prescribed is not None:
        # cinématique réimposée au prochain pas ; on porte les valeurs courantes
        new_disp, new_vel = disp, vel
    else:
        # EDO de structure (Euler semi-implicite, dt = 1). F_IB corrigée de la masse
        # interne ajoutée via m_eff (cf. test masse ajoutée) — porté par spring.m.
        s = coupled.spring
        F = F_hydro * s.dof_mask
        acc = (F - s.c * vel - s.k * disp) / s.m
        new_vel = (vel + acc) * s.dof_mask
        new_disp = disp + new_vel
    new = {"f": f_new, "f_buf": f_buf_new, "disp": new_disp, "vel": new_vel}
    px, py = coupled.probe_xy
    diag = {
        "drag": F_hydro[0],
        "lift": F_hydro[1],
        "disp_x": disp[0],
        "disp_y": disp[1],
        "vel_y": vel[1],
        "uy_probe": u[1, px, py],
        "rho_mean": rho.mean(),
    }
    return new, diag


def rollout(coupled: Coupled, state, n_steps: int, t0: int = 0):
    """Avance n_steps et renvoie (état_final, diagnostics empilés (n_steps,))."""
    def body(carry, i):
        st, diag = coupled_step(coupled, carry, t0 + i)
        return st, diag
    state, diags = jax.lax.scan(body, state, jnp.arange(n_steps))
    return state, diags
