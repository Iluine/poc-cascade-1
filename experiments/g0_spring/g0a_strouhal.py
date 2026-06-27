"""G0a (§4a) — Strouhal du sillage d'un cylindre FIXE, confronté à la littérature.

Premier chiffre qui compte (spec). Valide l'INSTRUMENT, pas l'architecture.
Référence Re=100, cylindre 2D : St ≈ 0.164–0.166 (Williamson), Cd ≈ 1.32–1.40.
Tolérance pré-enregistrée G0a : ±15 % sur St.

Usage: .venv/bin/python experiments/g0_spring/g0a_strouhal.py
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.metrics import strouhal, amplitude_envelope


def run(nx=512, ny=192, D=20.0, U=0.08, Re=100.0, cx=120.0,
        n_total=44000, n_transient=18000, markers=80, forcing_iters=3):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=Re, walls="bounceback")
    fl = build_fluid(cfg)
    cy = ny / 2 + 2.0  # léger décalage vertical : brise la symétrie, amorce le lâcher
    body = make_cylinder(cx=cx, cy=cy, radius=D / 2, n_markers=markers)
    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    probe = (int(cx + 4 * D), int(cy))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=forcing_iters, probe_xy=probe)

    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    state = init_state(cpl)
    blockage = D / ny
    T_shed_guess = D / (0.165 * U)
    print(f"grille {nx}x{ny}  D={D}  U={U}  Re={Re}  blockage={blockage:.3f}  "
          f"omega={cfg.omega:.4f}  T_shed~{T_shed_guess:.0f} pas")

    t0 = time.time()
    state, d = roll(state, n_steps=n_total)
    jax.block_until_ready(d)
    print(f"{n_total} pas en {time.time()-t0:.1f}s ({n_total/(time.time()-t0):.0f}/s)")

    lift = np.asarray(d["lift"]); drag = np.asarray(d["drag"])
    dispy = np.asarray(d["disp_y"]); rho = np.asarray(d["rho_mean"])
    win = slice(n_transient, None)  # fenêtre stationnaire
    St, f = strouhal(lift[win], D=D, U=U, band=(0.3 * 0.165 * U / D, 3 * 0.165 * U / D))
    Cl_amp, _ = amplitude_envelope(lift[win])
    Cd = float(np.mean(drag[win])) / (0.5 * U ** 2 * D)
    Cl_rms = float(np.std(lift[win])) / (0.5 * U ** 2 * D)

    St_ref = 0.165
    err = abs(St - St_ref) / St_ref
    verdict = "PASS" if err <= 0.15 else "FAIL"
    print("\n========== G0a ==========")
    print(f"finite          : {bool(np.isfinite(lift).all())}")
    print(f"rho_mean (fin)  : {rho[-1]:.4f}  (conservation masse)")
    print(f"lift std (window): {np.std(lift[win]):.3e}  (lâcher établi si >> 0)")
    print(f"St mesuré       : {St:.4f}   (réf {St_ref}, ±15% -> [{St_ref*0.85:.3f},{St_ref*1.15:.3f}])")
    print(f"  -> écart {100*err:.1f}%   G0a : {verdict}")
    print(f"Cd              : {Cd:.3f}   (réf ~1.32–1.40)   [check secondaire]")
    print(f"Cl_rms          : {Cl_rms:.3f}   (réf ~0.22–0.33)")
    return dict(St=St, f=f, Cd=Cd, Cl_rms=Cl_rms, err=err, verdict=verdict,
                lift=lift, drag=drag, dispy=dispy, rho=rho)


if __name__ == "__main__":
    run()
