"""G0c — convergence d'AMPLITUDE au BORD de bande (le seul chiffre G0c encore ouvert).

L'onset (Ur≈4.5) est une borne d'AMPLITUDE (saut ×6.7), pas seulement fréquentielle. La
force statique bouge de −20%/×2 ; au bord, point marginal, ça peut décaler l'onset. On
teste donc l'amplitude au bord au raffinement ×1.5 (D=30, β constant), jugée en TAUX :

  taux force statique : Cl_rms 0.372→0.342 = −8.1% sur D=20→30.
  - A_y(D=30)/A_y(D=20) ≈ 0.92  -> amplitude suit la force ; extrapolable ; onset tient.
  - ratio > 0.92 (→1)          -> couplage amortit l'erreur de force (structure intègre). OK.
  - ratio < 0.92 / désynchro   -> onset résolution-dépendant, borne G0b fragile.

Stage 1 : ring-down D=30 (cale k pour Ur=4.5 EXACT, f_n∝√k). Stage 2 : run corps-mobile.
Réf D=20, Ur=4.5 : A_y/D = 0.536.
"""
import functools, math, time
import jax, jax.numpy as jnp, numpy as np
from scipy.signal import find_peaks, hilbert

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.metrics import dominant_frequency, phase_coherence

U, BETA, D = 0.08, 0.104, 30.0
R = D / 2
ny = int(round(D / BETA)); nx = int(round(ny * 512 / 192))
M_BODY = 5.0 * D ** 2
M_EFF = M_BODY - math.pi * R ** 2
ST20_AT_45 = 0.536          # A_y/D à D=20, Ur=4.5 (du sweep G0b)
UR = 4.5


def build(k, inflow):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, inflow=inflow, diameter=D, reynolds=100,
                      walls="bounceback", seed_perturb=(0.0 if inflow == 0.0 else 1e-3))
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=R, fill=True)
    spring = Spring(m=M_EFF, k=k, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=3, probe_xy=(int(nx//3+4*D), ny//2))
    return cpl


def ringdown_fn(k0, A0=8.0, N=18000):
    cpl = build(k0, inflow=0.0)
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    s = init_state(cpl); s = {**s, "disp": jnp.array([0.0, A0], jnp.float32)}
    s, d = roll(s, n_steps=N); jax.block_until_ready(d)
    y = np.asarray(d["disp_y"]); pks, _ = find_peaks(y)
    f_n = float(np.mean(1.0 / np.diff(pks))) if len(pks) >= 3 else float("nan")
    return f_n


def main():
    print(f"D={D} {nx}x{ny}  m_body={M_BODY:.0f} m_eff={M_EFF:.0f}")
    # Stage 1 : ring-down pour caler k à Ur=4.5
    k0 = 0.0812
    t0 = time.time(); f_n0 = ringdown_fn(k0)
    f_n_target = U / (UR * D)
    k45 = k0 * (f_n_target / f_n0) ** 2
    print(f"ring-down: f_n(k0={k0})={f_n0:.3e} -> k pour Ur={UR}: {k45:.4f} "
          f"(f_n_target={f_n_target:.3e}) [{time.time()-t0:.0f}s]")

    # Stage 2 : run corps-mobile au bord de bande
    cpl = build(k45, inflow=U)
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    N = 150000
    t0 = time.time(); s, d = roll(init_state(cpl), n_steps=N); jax.block_until_ready(d)
    y = np.asarray(d["disp_y"]); lift = np.asarray(d["lift"]); uyp = np.asarray(d["uy_probe"])
    f_shed0 = 0.18 * U / D
    band = (0.3 * f_shed0, 3 * f_shed0)
    sta = y[N//2:]
    env = np.abs(hilbert(sta - sta.mean()))
    th = len(env)//3; blk = [env[:th].mean(), env[th:2*th].mean(), env[2*th:].mean()]
    A_y = blk[-1] / D
    sat = (max(blk)-min(blk))/(np.mean(blk)+1e-12)
    f_lift, _ = dominant_frequency(lift[N//2:], band=band)
    M4 = phase_coherence(uyp[N//2:], sta, band=(0.5*f_shed0, 1.5*f_shed0))
    print(f"run corps-mobile Ur={UR}: {N/(time.time()-t0):.0f} st/s")

    ratio = A_y / ST20_AT_45
    rate_force = 0.342 / 0.372   # = 0.919, taux force statique D20->D30
    print("\n========== G0c amplitude au bord (Ur=4.5) ==========")
    print(f"A_y/D : D=20 → {ST20_AT_45:.3f} ; D=30 → {A_y:.3f}  (sat={sat:.2f}, M4={M4:.2f}, locked={'oui' if A_y>0.1 else 'NON'})")
    print(f"ratio amplitude D30/D20 = {ratio:.3f}")
    print(f"taux force statique     = {rate_force:.3f}  (−8.1%)")
    if A_y < 0.1:
        verdict = "ONSET DÉPLACÉ : plus verrouillé à Ur=4.5 → borne G0b résolution-dépendante"
    elif ratio >= rate_force - 0.05:
        verdict = ("amplitude suit (ou bat) le taux force → onset TIENT ; couplage amortit l'erreur"
                   if ratio > rate_force + 0.03 else "amplitude suit le taux force → cohérent, onset tient")
    else:
        verdict = "amplitude chute PLUS vite que la force → onset fragile, à creuser"
    print(f"=> {verdict}")


if __name__ == "__main__":
    main()
