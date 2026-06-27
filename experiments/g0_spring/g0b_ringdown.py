"""G0b prérequis — RING-DOWN en fluide quiescent : pine ω_n du sim + teste la LINÉARITÉ
de la masse entraînée (feedback : la cancellation de la masse ajoutée n'est exacte que si
l'inertie entraînée est linéaire en a).

Cylindre déplacé puis relâché (v=0), ζ_struct=0 → ne décroît que par amortissement FLUIDE.
On mesure la période cycle par cycle pendant que l'amplitude décroît : période constante
=> inertie linéaire, ω_n bien défini, cancellation exacte. Période qui dérive avec
l'amplitude => C_a non-linéaire (à porter avant de fermer la boucle).

m_eff = m_corps − ρ_f·V (= m_corps − 314), correction de masse interne validée.
"""
import functools, math
import jax, jax.numpy as jnp, numpy as np
from scipy.signal import find_peaks

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout


def main(nx=512, ny=192, D=20.0, mstar=5.0, A0=10.0, k=0.0578, N=16000, iters=3):
    R = D / 2
    rho_V = math.pi * R ** 2
    m_body = mstar * 1.0 * D ** 2          # m* = m/(ρ_f D²)
    m_eff = m_body - rho_V                 # correction masse interne
    cfg = FluidConfig(nx=nx, ny=ny, u_in=0.08, inflow=0.0, diameter=D, reynolds=100,
                      walls="bounceback", seed_perturb=0.0)   # fluide quiescent
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=R, fill=True)
    spring = Spring(m=m_eff, k=k, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))  # crossflow
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=iters, probe_xy=(nx//2, ny//2))
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))

    st = init_state(cpl)
    st = {**st, "disp": jnp.array([0.0, A0], jnp.float32)}    # déplacé, relâché
    st, d = roll(st, n_steps=N); jax.block_until_ready(d)
    y = np.asarray(d["disp_y"])

    # période cycle par cycle (maxima successifs) vs amplitude
    pks, _ = find_peaks(y)
    print(f"m*={mstar} m_body={m_body:.0f} m_eff={m_eff:.0f} k={k}  A0={A0}")
    print(f"{len(pks)} maxima détectés")
    if len(pks) >= 3:
        periods = np.diff(pks)
        amps = y[pks[:-1]]
        fns = 1.0 / periods
        print(f"\n{'cycle':>5} {'amp':>7} {'période':>8} {'f_n':>9}")
        for i in range(len(periods)):
            print(f"{i:>5} {amps[i]:>7.2f} {periods[i]:>8d} {fns[i]:>9.3e}")
        # linéarité : variation de période entre grande et petite amplitude
        f_big = fns[0]; f_small = fns[-1]
        drift = abs(f_big - f_small) / f_small
        print(f"\nf_n grande amp = {f_big:.3e} ; petite amp = {f_small:.3e} ; dérive = {100*drift:.1f}%")
        print(f"f_n moyen = {np.mean(fns):.3e}  (cible f_shed=7.5e-4 ; St_n={np.mean(fns)*D/0.08:.3f})")
        # log-décrément (amortissement fluide)
        if len(amps) >= 3 and amps[0] > 0 and amps[-1] > 0:
            ld = np.log(amps[0]/amps[-1]) / (len(amps)-1)
            zeta_f = ld / (2*math.pi)
            print(f"log-décrément={ld:.3f} -> ζ_fluide≈{zeta_f:.3f}")
        verdict = "LINÉAIRE (cancellation exacte)" if drift < 0.03 else "NON-LINÉAIRE (C_a dépend de l'amplitude -> à porter)"
        print(f"=> masse entraînée : {verdict}")
    return d


if __name__ == "__main__":
    main()
