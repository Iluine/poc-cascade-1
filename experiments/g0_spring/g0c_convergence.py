"""G0c — convergence maillage = TEST DE LA THÉORIE D'ERREUR (feedback).

Prédiction falsifiable : biais ∝ 1/D (racine = étalement non-glissement Peskin sur ~2
cellules). 3 points D=20,30,40 à β CONSTANT (0.104) → seule la résolution change.
- 3 points testent l'ORDRE (pas seulement assument 1/D) : Richardson sur (St_30,St_40)
  vs (St_20,St_40) doit donner le même St_∞ si l'ordre est 1.
- Verdict double : (a) St converge-t-il au taux 1/D prédit ? (b) -> extrapolation St_∞.
  Si non-1/D : seconde source d'erreur = information précieuse.

Cylindre FIXE (St = canal fréquence). L'amplitude au bord de bande = run séparé ensuite.
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np
from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.metrics import strouhal

U = 0.08
BETA = 0.104  # tenu CONSTANT (= celui des runs G0a/G0b validés)


def st_at(D, n_periods=45, discard=18, iters=3):
    ny = int(round(D / BETA)); nx = int(round(ny * 512 / 192))  # même aspect que ×1
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=100, walls="bounceback",
                      seed_perturb=1e-3)
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=D/2, fill=True)
    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=iters, probe_xy=(int(nx//3+4*D), ny//2))
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    T = D / (0.18 * U); N = int(n_periods * T)
    s = init_state(cpl)
    t0 = time.time(); s, d = roll(s, n_steps=N); jax.block_until_ready(d)
    lift = np.asarray(d["lift"]); norm = 0.5 * U**2 * D
    w = slice(int(discard * T), None)
    St, f = strouhal(lift[w], D=D, U=U, band=(0.3*0.18*U/D, 3*0.18*U/D))
    Cl = float(np.std(lift[w])) / norm
    print(f"D={D:>2} {nx}x{ny}  N={N}  St={St:.4f}  Cl_rms={Cl:.3f}  ({N/(time.time()-t0):.0f} st/s)")
    return St


def main():
    print(f"β={BETA} constant ; biais St vs réf non-confinée 0.165 ; prédiction biais∝1/D")
    Ds = [20, 30, 40]
    sts = {D: st_at(D) for D in Ds}
    print("\n========== CONVERGENCE St (théorie d'erreur) ==========")
    for D in Ds:
        print(f"  D={D:>2}  St={sts[D]:.4f}  biais/0.165 = {100*(sts[D]-0.165)/0.165:+.1f}%")
    # Richardson (ordre 1) sur deux paires -> St_inf ; cohérence = ordre 1 confirmé
    def rich(D1, D2):  # St_inf = St2 + (St2-St1)/(D2/D1 - 1)
        return sts[D2] + (sts[D2]-sts[D1])/(D2/D1 - 1)
    inf_2040 = 2*sts[40]-sts[20]            # ordre 1 exact si D40=2·D20
    inf_3040 = rich(30, 40)
    print(f"\nRichardson ordre-1 St_∞ : via (20,40)={inf_2040:.4f} ; via (30,40)={inf_3040:.4f}")
    print(f"  cohérence des deux extrapolations : écart {100*abs(inf_2040-inf_3040)/inf_2040:.1f}%")
    print(f"  (écart faible => ordre 1 confirmé / théorie 1/D tient ; écart fort => 2e source)")
    # taux observé vs 1/D
    d2030 = sts[20]-sts[30]; d3040 = sts[30]-sts[40]
    print(f"\nΔSt(20→30)={d2030:+.4f}  ΔSt(30→40)={d3040:+.4f}")
    print(f"  ratio observé {d3040/d2030 if d2030 else float('nan'):.2f} ; prédit 1/D : "
          f"{(1/30-1/40)/(1/20-1/30):.2f}")


if __name__ == "__main__":
    main()
