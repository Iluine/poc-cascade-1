"""Test du forçage VOLUMIQUE (correctif du cylindre poreux).

Attendu si l'intérieur creux était la cause : vitesse intérieure → ~0, Cl_rms sature
vers ~0.3 (réf), battement disparu. St reste ~0.18–0.19 (variable rapide inchangée).
"""

import functools, time
import jax, jax.numpy as jnp, numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.coupling.peskin import interpolate
from cascade.metrics import strouhal


def main(nx=512, ny=400, D=20.0, U=0.08, Re=100.0, cx=120.0, n_total=80000, iters=3):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=Re, walls="bounceback")
    fl = build_fluid(cfg)
    cy = ny / 2 + 2.0
    body = make_cylinder(cx=cx, cy=cy, radius=D / 2, fill=True, spacing=1.0)
    print("marqueurs volumiques M=%d  ds=%.3f" % (body.base.shape[0], body.ds))
    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=iters, probe_xy=(int(cx+4*D), int(cy)))
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    state = init_state(cpl)
    norm = 0.5 * U ** 2 * D
    T_shed = D / (0.185 * U)

    block = 20000; done = 0; t0 = time.time(); lift_all = []
    while done < n_total:
        nb = min(block, n_total - done)
        state, d = roll(state, n_steps=nb); jax.block_until_ready(d)
        lift_all.append(np.asarray(d["lift"])); done += nb
        print("  t=%6d Cl_rms[bloc]=%.3f (%.0f/s)" % (done, np.std(d["lift"])/norm, done/(time.time()-t0)))
    lift = np.concatenate(lift_all)

    # vitesse intérieure finale
    rho, u = fl.macroscopic(state["f"]); u = np.asarray(u)
    R = D/2; xs = np.arange(nx)[:,None]; ys = np.arange(ny)[None,:]
    inside = ((xs-cx)**2 + (ys-cy)**2) < (0.7*R)**2
    u_in = np.sqrt(u[0]**2+u[1]**2)[inside]/U

    # Cl_rms glissant + St sur la 2e moitié (≈ stationnaire)
    W = int(6*T_shed)
    cl_series = [np.std(lift[t-W:t])/norm for t in range(W, n_total+1, 2000)]
    half = lift[n_total//2:]
    St,_ = strouhal(half, D=D, U=U, band=(0.3*0.185*U/D, 3*0.185*U/D))
    Cd = float(np.mean(np.asarray(d["drag"])))/norm if False else None

    print("\n========== FORÇAGE VOLUMIQUE ==========")
    print("vitesse intérieure /U : max=%.3f moyenne=%.3f  (était 0.237/0.105)" % (u_in.max(), u_in.mean()))
    print("Cl_rms glissant : min=%.3f max=%.3f final=%.3f (réf ~0.3 ; battement = max-min)" % (
        min(cl_series), max(cl_series), cl_series[-1]))
    print("St (2e moitié) : %.4f" % St)


if __name__ == "__main__":
    main()
