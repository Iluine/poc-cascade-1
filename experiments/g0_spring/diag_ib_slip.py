"""DIAGNOSTIC IB (amont du Cl_rms) — le non-glissement est-il imposé ?

Hypothèse : la portance ×4–6 trop haute + le battement viennent d'une coque de marqueurs
PERMÉABLE (cylindre poreux) → écoulement à travers l'obstacle → portance fausse + modulée.

Mesures (snapshot après formation du sillage) :
  1. glissement résiduel |u_interp − U_body|/U aux marqueurs (max, moyen) ;
  2. vitesse à l'INTÉRIEUR du cylindre |u|/U (doit être ~0 pour un solide) ;
  3. effet du nombre d'itérations de forçage sur le glissement résiduel (au même snapshot).
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.coupling.peskin import interpolate, spread


def main(nx=512, ny=400, D=20.0, U=0.08, Re=100.0, cx=120.0, n_warm=25000, iters=3, markers=80):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=Re, walls="bounceback")
    fl = build_fluid(cfg)
    cy = ny / 2 + 2.0
    body = make_cylinder(cx=cx, cy=cy, radius=D / 2, n_markers=markers)
    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=iters, probe_xy=(int(cx+4*D), int(cy)))
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    state = init_state(cpl)
    state, _ = roll(state, n_steps=n_warm)
    rho, u = fl.macroscopic(state["f"])
    rho = np.asarray(rho); u = np.asarray(u)

    center = np.array([cx, cy])
    markers_xy = np.asarray(body.base) + center
    R = D / 2

    # 1. glissement résiduel aux marqueurs (U_body = 0)
    slip = np.asarray(interpolate(jnp.asarray(u), jnp.asarray(markers_xy)))  # (M,2)
    slip_mag = np.linalg.norm(slip, axis=1) / U
    # 2. vitesse intérieure
    xs = np.arange(nx)[:, None]; ys = np.arange(ny)[None, :]
    inside = ((xs - cx) ** 2 + (ys - cy) ** 2) < (0.7 * R) ** 2
    umag = np.sqrt(u[0] ** 2 + u[1] ** 2)
    u_inside = umag[inside] / U
    # vitesse de référence libre (loin, en amont)
    u_free = umag[20, ny // 2] / U

    print("========== DIAGNOSTIC IB (snapshot t=%d) ==========" % n_warm)
    print("glissement résiduel /U  : max=%.3f  moyen=%.3f   (0 = non-glissement parfait)" % (
        slip_mag.max(), slip_mag.mean()))
    print("vitesse intérieure /U   : max=%.3f  moyenne=%.3f  (0 = solide imperméable)" % (
        u_inside.max(), u_inside.mean()))
    print("vitesse libre amont /U  : %.3f  (référence ~1)" % u_free)

    # 3. effet des itérations de forçage sur le glissement (au même snapshot)
    print("\nréduction du glissement par itérations de forçage (snapshot figé) :")
    u_cur = jnp.asarray(u); rho_j = jnp.asarray(rho)
    U_body = jnp.zeros_like(jnp.asarray(markers_xy))
    for it in range(1, 7):
        u_b = interpolate(u_cur, jnp.asarray(markers_xy))
        resid = float(jnp.linalg.norm(u_b, axis=1).max()) / U
        F_L = U_body - u_b
        g = spread(F_L, jnp.asarray(markers_xy), (nx, ny), ds=body.ds)
        u_cur = u_cur + g / jnp.clip(rho_j, 1e-3)
        print("  avant iter %d : glissement max /U = %.3f" % (it, resid))


if __name__ == "__main__":
    main()
