"""Mesure de force PROPRE — Cl_rms stationnaire par blocs de périodes entières.

Tranche : la force est-elle fausse partout (formule) ou existe-t-il un régime sain
à obstruction modérée ? Pas de fenêtre glissante (scalloping) — std par blocs de ~10
périodes entières ; stationnaire si les blocs sont cohérents.
"""

import functools, time, sys
import jax, jax.numpy as jnp, numpy as np
from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.metrics import strouhal


def run(ny=192, fill=False, iters=3, nx=512, D=20.0, U=0.08, Re=100.0, cx=120.0, n_total=150000):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=Re, walls="bounceback")
    fl = build_fluid(cfg)
    cy = ny / 2 + 2.0
    body = make_cylinder(cx=cx, cy=cy, radius=D / 2, fill=fill, spacing=1.0)
    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=iters, probe_xy=(int(cx+4*D), int(cy)))
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    state = init_state(cpl); norm = 0.5 * U ** 2 * D
    M = body.base.shape[0]
    print(f"β={D/ny:.3f} fill={fill} M={M} iters={iters} grille {nx}x{ny} horizon={n_total}")

    block = 25000; done = 0; t0 = time.time(); lift_all=[]; drag_all=[]
    while done < n_total:
        nb = min(block, n_total - done)
        state, d = roll(state, n_steps=nb); jax.block_until_ready(d)
        lift_all.append(np.asarray(d["lift"])); drag_all.append(np.asarray(d["drag"])); done += nb
        print(f"  t={done:>7d}  ({done/(time.time()-t0):.0f}/s)")
    lift = np.concatenate(lift_all); drag = np.concatenate(drag_all)

    # St sur la 2e moitié -> période en pas
    St, f = strouhal(lift[n_total//2:], D=D, U=U, band=(0.3*0.18*U/D, 3*0.18*U/D))
    Tsh = int(round(1.0/f)) if f and f>0 else int(D/(0.18*U))
    # std par blocs de 10 périodes ENTIÈRES sur la 2e moitié (stationnarité + niveau)
    W = 10*Tsh
    sta = lift[n_total//2:]
    blocks = [np.std(sta[i:i+W])/norm for i in range(0, len(sta)-W, W)]
    Cd = float(np.mean(drag[n_total//2:]))/norm
    print(f"\n========== FORCE (β={D/ny:.3f}, fill={fill}) ==========")
    print(f"St={St:.4f}  T_shed={Tsh} pas  Cd={Cd:.3f}")
    print(f"Cl_rms par blocs de 10 périodes : {[f'{b:.2f}' for b in blocks]}")
    print(f"Cl_rms stationnaire = {np.mean(blocks):.3f} ± {np.std(blocks):.3f}   (réf ~0.3)")
    return dict(St=St, Cd=Cd, clrms=float(np.mean(blocks)), blocks=blocks)


if __name__ == "__main__":
    ny = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    fill = (sys.argv[2] == "fill") if len(sys.argv) > 2 else False
    run(ny=ny, fill=fill)
