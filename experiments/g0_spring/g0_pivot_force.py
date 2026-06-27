"""TEST PIVOT (§G0, canal couplage) — Cl_rms(t) vs St(t) à β=0.05, horizon long.

Tranche la question qui décide entre instrument propre et vraie impasse :

  - Hypothèse transitoire (probable) : près de la bifurcation de Hopf, St (variable
    RAPIDE) se verrouille tôt mais l'enveloppe d'amplitude (variable LENTE) croît sur un
    temps lent. Le Cl_rms=1.18 mesuré à t~44k serait alors la force PAS ENCORE saturée,
    pas une force fausse. Signature : Cl_rms(t) culmine puis redescend vers ~0.3.
  - Hypothèse bug forçage IB : un biais direct-forcing est ~constant (≠ domaine-dépendant).
    Signature : Cl_rms(t) plateau à 1.18. Alors seulement → itérations de forçage.

On suit les DEUX échelles de temps simultanément (St rapide, amplitude lente) : si
l'horizon est trop court pour l'amplitude tout en étant ample pour la fréquence, M3/M4
(enveloppe, lock-in) en pâtissent → impact C4, pas seulement G0b.

Sauve la série complète (npz) pour ré-analyse des fenêtres sans re-run.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.metrics import strouhal


def run_pivot(nx=512, ny=400, D=20.0, U=0.08, Re=100.0, cx=120.0,
              n_total=220000, markers=80, forcing_iters=3,
              out="experiments/g0_spring/pivot_b05.npz"):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=Re, walls="bounceback")
    fl = build_fluid(cfg)
    cy = ny / 2 + 2.0
    body = make_cylinder(cx=cx, cy=cy, radius=D / 2, n_markers=markers)
    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=forcing_iters,
                  probe_xy=(int(cx + 4 * D), int(cy)))
    T_shed = D / (0.185 * U)  # St mesuré à β=0.05
    print(f"PIVOT β={D/ny:.3f}  grille {nx}x{ny}  T_shed~{T_shed:.0f}  horizon={n_total} "
          f"(~{n_total/T_shed:.0f} T_shed)")

    # Rollout en blocs pour borner la mémoire du scan et suivre la progression.
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    state = init_state(cpl)
    block = 20000
    lift_blocks, drag_blocks = [], []
    t0 = time.time()
    done = 0
    while done < n_total:
        nb = min(block, n_total - done)
        state, d = roll(state, n_steps=nb)
        jax.block_until_ready(d)
        lift_blocks.append(np.asarray(d["lift"]))
        drag_blocks.append(np.asarray(d["drag"]))
        done += nb
        # Cl_rms instantané sur le dernier bloc (diagnostic de progression)
        seg = np.asarray(d["lift"])
        clr = np.std(seg) / (0.5 * U ** 2 * D)
        print(f"  t={done:>7d}  Cl_rms[bloc]={clr:6.3f}  ({done/(time.time()-t0):.0f}/s)")
    lift = np.concatenate(lift_blocks)
    drag = np.concatenate(drag_blocks)
    np.savez(out, lift=lift, drag=drag, U=U, D=D, T_shed=T_shed, n_total=n_total)

    # --- Analyse à deux échelles : sliding window ---
    W = int(6 * T_shed)             # fenêtre = 6 périodes de lâcher
    step = max(2000, W // 6)
    norm = 0.5 * U ** 2 * D
    band = (0.3 * 0.185 * U / D, 3 * 0.185 * U / D)
    print(f"\n{'t':>8} {'Cl_rms':>8} {'St_local':>9}")
    rows = []
    for t in range(W, n_total + 1, step):
        seg = lift[t - W:t]
        clrms = float(np.std(seg)) / norm
        st, _ = strouhal(seg, D=D, U=U, band=band)
        rows.append((t, clrms, st))
        print(f"{t:>8} {clrms:>8.3f} {st:>9.4f}")

    arr = np.array(rows)
    clrms_series = arr[:, 1]
    peak = float(clrms_series.max())
    last = float(np.mean(clrms_series[-3:]))   # moyenne des 3 dernières fenêtres
    descended = peak > 1.1 * last              # culmine puis redescend
    plateaued_high = last > 0.6 and not descended
    print("\n========== VERDICT PIVOT ==========")
    print(f"Cl_rms : pic={peak:.3f}  final(≈stationnaire)={last:.3f}")
    print(f"St final : {arr[-1,2]:.4f}  (vs St à t~44k = 0.185 ; doit être ~stable = variable rapide)")
    if descended and last < 0.5:
        print("=> TRANSITOIRE : l'amplitude a culminé puis redescendu. "
              "Le 1.18 était la force NON saturée. Instrument propre sur la fenêtre finale.")
    elif plateaued_high:
        print("=> PLATEAU HAUT : Cl_rms reste >> réf. Vrai bug de forçage IB -> "
              "isoler l'accumulation des itérations de forçage (UNE variable).")
    else:
        print("=> INDÉTERMINÉ : prolonger l'horizon (amplitude encore en évolution).")
    return arr, lift, drag


if __name__ == "__main__":
    run_pivot()
