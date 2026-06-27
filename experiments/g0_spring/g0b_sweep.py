"""G0b — balayage en vitesse réduite : existence et BORNES de la bande de lock-in.

Ur = U/(f_n·D) centré sur f_shed MESURÉE (pas le St de manuel : sinon biais confinement
14% rentre dans le ±20% -> PASS contaminé). k(Ur)=k_center·(Ur_center/Ur)².
m*=5, ζ_struct=0, crossflow, m_eff=m_corps−ρ_f·V.

Par point : A_y/D (enveloppe stationnaire), f_struct, f_lift, M4 (cohérence lift↔disp),
ET stationnarité de l'enveloppe (la variable LENTE — feedback) en 3 blocs.

Lock-in = synchronisation (f_lift suit f_n=f_struct, quitte f_shed0) + A_y grande + M4>0.8.
Verdict = bornes onset/offset vs théorie VIV (±20%), centre retiré du confinement.
"""
import functools, math, sys
import jax, jax.numpy as jnp, numpy as np
from scipy.signal import hilbert

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.metrics import dominant_frequency, phase_coherence

U, D = 0.08, 20.0
ST_MEAS = 0.188
F_SHED0 = ST_MEAS * U / D            # 7.52e-4
UR_CENTER = 1.0 / ST_MEAS            # 5.32
K_CENTER = 0.0578                    # ring-down : f_n=f_shed0 à ce k
RHO_V = math.pi * (D / 2) ** 2


def one_point(Ur, nx=512, ny=192, mstar=5.0, N=150000, iters=3):
    k = K_CENTER * (UR_CENTER / Ur) ** 2
    m_eff = mstar * D ** 2 - RHO_V
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=100, walls="bounceback",
                      seed_perturb=1e-3)   # inflow = u_in (sillage entretenu)
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=D/2, fill=True)
    spring = Spring(m=m_eff, k=k, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=iters, probe_xy=(nx//2+80, ny//2))
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    st = init_state(cpl)
    st, d = roll(st, n_steps=N); jax.block_until_ready(d)
    y = np.asarray(d["disp_y"]); lift = np.asarray(d["lift"]); uyp = np.asarray(d["uy_probe"])
    np.savez(f"experiments/g0_spring/g0b_pt_Ur{Ur:.2f}.npz", disp_y=y, lift=lift, uy_probe=uyp)
    band = (0.3 * F_SHED0, 3 * F_SHED0)
    # stationnarité de l'enveloppe (3 blocs sur la 2e moitié) — la variable LENTE
    sta = y[N//2:]
    env = np.abs(hilbert(sta - sta.mean()))
    th = len(env) // 3
    blk = [env[:th].mean(), env[th:2*th].mean(), env[2*th:].mean()]
    A_y = blk[-1] / D
    sat = (max(blk) - min(blk)) / (np.mean(blk) + 1e-12)   # <0.1 => saturé
    f_struct, _ = dominant_frequency(sta, band=band)
    f_lift, _ = dominant_frequency(lift[N//2:], band=band)
    # M4 CORRIGÉ : cohérence sillage↔structure via sonde de sillage INDÉPENDANTE (pas la portance)
    M4 = phase_coherence(uyp[N//2:], sta, band=(0.5*F_SHED0, 1.5*F_SHED0))
    fr_n = 1.0 / (Ur * ST_MEAS)   # f_n/f_shed0 attendu (entraînement = f_lift tiré vers fr_n)
    return dict(Ur=Ur, k=k, A_y=A_y, f_struct=f_struct, f_lift=f_lift,
                fr_struct=f_struct/F_SHED0, fr_lift=f_lift/F_SHED0, fr_n=fr_n, M4=M4, sat=sat)


def main():
    Urs = [4.0, 4.25, 4.5, 5.32, 6.0, 7.0, 7.5, 8.5] if len(sys.argv) < 2 else [float(x) for x in sys.argv[1:]]
    print(f"f_shed0={F_SHED0:.3e} Ur_center={UR_CENTER:.2f}  (lock-in attendu ~Ur[4,8])")
    print(f"\n{'Ur':>5} {'k':>8} {'A_y/D':>7} {'f_lift/f0':>10} {'f_n/f0':>7} {'M4(wake)':>9} {'sat':>6}")
    rows = []
    for Ur in Urs:
        r = one_point(Ur)
        rows.append(r)
        flag = "" if r["sat"] < 0.12 else " <PAS SATURÉ>"
        print(f"{r['Ur']:>5.2f} {r['k']:>8.4f} {r['A_y']:>7.3f} "
              f"{r['fr_lift']:>10.3f} {r['fr_n']:>7.3f} {r['M4']:>9.2f} {r['sat']:>6.2f}{flag}")
    np.save("experiments/g0_spring/g0b_sweep.npy", rows, allow_pickle=True)
    # bande : points synchronisés (f_lift suit f_struct ~ f_n) + M4>0.8 + A_y notable
    locked = [r for r in rows if r["M4"] > 0.8 and abs(r["fr_lift"] - r["fr_struct"]) < 0.1 and r["A_y"] > 0.05]
    if locked:
        print(f"\nbande lock-in (M4>0.8, synchronisé) : Ur ∈ [{min(r['Ur'] for r in locked):.2f}, "
              f"{max(r['Ur'] for r in locked):.2f}]  (théorie VIV ~[4,8], ±20%)")
    else:
        print("\naucun point clairement verrouillé — à investiguer")


if __name__ == "__main__":
    main()
