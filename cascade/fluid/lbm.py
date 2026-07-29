"""Oracle fluide LBM (XLB, backend JAX) pour le substrat VIV (§2).

On réutilise le ``IncompressibleNavierStokesStepper`` JAX de XLB (stream+collide+BC,
validé et différentiable) et ses opérateurs. Le forçage IB spatialement variable est
injecté par le module de couplage, PAS ici (XLB ne supporte qu'un vecteur force constant).

Conventions : champs ``(C, nx, ny)`` ; axe 1 = x (sens du courant), axe 2 = y.
Unités réseau (dx = dt = 1). Relation BGK : ν = c_s²(τ − 1/2), c_s² = 1/3, ω = 1/τ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

import xlb
from xlb import ComputeBackend, PrecisionPolicy
from xlb.velocity_set import D2Q9
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition import (
    RegularizedBC,
    ExtrapolationOutflowBC,
    FullwayBounceBackBC,
)


@dataclass(frozen=True)
class FluidConfig:
    nx: int = 256
    ny: int = 128
    u_in: float = 0.05          # vitesse d'entrée (unités réseau, Mach bas)
    diameter: float = 20.0      # D de l'obstacle (sert à ν via Re)
    reynolds: float = 100.0     # Re = u_in · D / ν
    walls: Literal["freestream", "bounceback"] = "bounceback"
    seed_perturb: float = 1e-3  # bruit initial pour amorcer le lâcher (brise la symétrie)
    seed: int = 0               # graine du bruit initial. Le prereg exige « ≥ 5 seeds »
                                # pour C2 : câblée à 0 jusqu'ici, la clause n'a JAMAIS été
                                # exécutable (review 28/07, M13). Exposée AVANT toute
                                # réutilisation de l'instrument en T1.5.
    inflow: float | None = None  # vitesse aux BC/IC ; None => u_in. Mettre 0 pour fluide quiescent
                                 # (la viscosité reste fixée par u_in/Re ci-dessus).

    @property
    def u_bc(self) -> float:
        return self.u_in if self.inflow is None else self.inflow

    @property
    def nu(self) -> float:
        return self.u_in * self.diameter / self.reynolds

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def omega(self) -> float:
        return 1.0 / self.tau


@dataclass
class Fluid:
    cfg: FluidConfig
    stepper: object
    macroscopic: object
    equilibrium: object
    bc_mask: jax.Array
    missing_mask: jax.Array
    omega: jax.Array
    f_init: jax.Array
    f_buf: jax.Array
    grid: object

    def step(self, f, f_buf, t):
        """Un pas fluide pur (stream+collide+BC), sans forçage IB."""
        return self.stepper(f, f_buf, self.bc_mask, self.missing_mask, self.omega, t)


def _stack_indices(*blocks):
    """Concatène des blocs d'indices (2,N) en une liste [[x...],[y...]] pour XLB."""
    xs = np.concatenate([np.asarray(b)[0] for b in blocks])
    ys = np.concatenate([np.asarray(b)[1] for b in blocks])
    return [xs.tolist(), ys.tolist()]


def build_fluid(cfg: FluidConfig) -> Fluid:
    backend, pp = ComputeBackend.JAX, PrecisionPolicy.FP32FP32
    vs = D2Q9(precision_policy=pp, compute_backend=backend)
    xlb.init(velocity_set=vs, default_backend=backend, default_precision_policy=pp)

    grid = grid_factory((cfg.nx, cfg.ny), compute_backend=backend)
    bb = grid.bounding_box_indices()
    bb_inner = grid.bounding_box_indices(remove_edges=True)  # exclut les coins

    u_in = cfg.u_bc  # vitesse d'entrée effective (0 pour fluide quiescent)
    # Entrée (gauche) : vitesse prescrite (u_in, 0). Sortie (droite) : extrapolation.
    inlet = RegularizedBC("velocity", prescribed_value=(u_in, 0.0), indices=bb["left"])
    outlet = ExtrapolationOutflowBC(indices=bb["right"])
    bcs = [inlet, outlet]

    if cfg.walls == "bounceback":
        walls = FullwayBounceBackBC(indices=_stack_indices(bb_inner["bottom"], bb_inner["top"]))
        bcs.append(walls)
    else:  # freestream : haut/bas en vitesse libre (u_in,0) -> approx domaine non borné
        fs = RegularizedBC(
            "velocity",
            prescribed_value=(u_in, 0.0),
            indices=_stack_indices(bb_inner["bottom"], bb_inner["top"]),
        )
        bcs.append(fs)

    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=bcs, collision_type="BGK"
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    macroscopic = Macroscopic(velocity_set=vs, precision_policy=pp, compute_backend=backend)
    equilibrium = QuadraticEquilibrium(velocity_set=vs, precision_policy=pp, compute_backend=backend)

    # Condition initiale : flux uniforme (ρ=1, u=(u_in,0)) + petit bruit pour amorcer le lâcher.
    rho0 = jnp.ones((1, cfg.nx, cfg.ny), dtype=f_0.dtype)
    key = jax.random.PRNGKey(cfg.seed)
    noise = cfg.seed_perturb * jax.random.normal(key, (2, cfg.nx, cfg.ny), dtype=f_0.dtype)
    u0 = jnp.stack(
        [jnp.full((cfg.nx, cfg.ny), u_in, dtype=f_0.dtype), jnp.zeros((cfg.nx, cfg.ny), f_0.dtype)]
    ) + noise
    f_init = equilibrium(rho0, u0)

    return Fluid(
        cfg=cfg,
        stepper=stepper,
        macroscopic=macroscopic,
        equilibrium=equilibrium,
        bc_mask=bc_mask,
        missing_mask=missing_mask,
        omega=jnp.float32(cfg.omega),
        f_init=f_init,
        f_buf=f_1,
        grid=grid,
    )
