"""LA mesure qui décide binaire-vs-ternaire AVANT le gros build (feedback, 3 corrections en 1).

Principe unique : la métrique doit voir ce que le verdict tranche. Donc :
- contre la vérité FINE (pas grossière — sinon la non-fermabilité est invisible) ;
- mémoïser vs expert vs DESCEND (pas binaire présupposé — le bord de bande réveille le descend) ;
- en distance PERCEPTUELLE (fréq ±10 %, amplitude ±20 %), pas erreur de cache L2 ;
- sur les fenêtres ÉNERGÉTIQUES (faible amplitude = perceptuellement hors-sujet même si non-fermable).

Par fenêtre active, signal local = u_y fin au centre de fenêtre, sur l'horizon stationnaire :
  descend  = vérité fine            (erreur perceptuelle 0, coût haut)
  expert   = contenu grossier (ce que POD+DMD reconstruit au mieux ; manque le détail/phase fin)
  mémoïser = réutilisation clé-halo de la vérité fine passée (saute la phase si non-fermable)
Action juste = la moins chère sous JND. %descend > 0 => C2 TERNAIRE => chirurgie obligatoire (le test).
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.harten import windowing as H
from cascade.metrics import dominant_frequency

U, D = 0.08, 20.0
ST = 0.188
UR_CENTER = 1.0 / ST
K_CENTER = 0.0578
M_EFF = 5.0 * D ** 2 - np.pi * (D / 2) ** 2
LEVELS = 3              # fenêtre = 8×8 fin ; grille 64×24
JND_F, JND_A = 0.10, 0.20   # seuils perceptuels (M2 fréq, M3 amplitude)


def collect(Ur, nx=512, ny=192, stride=20, n_snap=900, warm_periods=40):
    k = K_CENTER * (UR_CENTER / Ur) ** 2
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=100, walls="bounceback", seed_perturb=1e-3)
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=D/2, fill=True)
    spring = Spring(m=float(M_EFF), k=k, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=3, probe_xy=(nx//2, ny//2))
    step = 2 ** LEVELS
    off = step // 2

    def sample(state):
        rho, u = fl.macroscopic(state["f"])
        q = jnp.concatenate([rho, rho * u], axis=0)
        for _ in range(LEVELS):
            q = H.coarsen(q)                          # (3, 64, 24) grossier
        fine_uy = u[1, off::step, off::step]          # u_y fin au centre de chaque fenêtre (64,24)
        return q, fine_uy

    @functools.partial(jax.jit, static_argnames=("n",))
    def warm(s, n):
        s, _ = rollout(cpl, s, n); return s

    @functools.partial(jax.jit, static_argnames=("ncol",))
    def coll(s, ncol):
        def b(st, _):
            st, _ = rollout(cpl, st, stride)
            return st, sample(st)
        s, outs = jax.lax.scan(b, s, None, length=ncol)
        return s, outs

    T = D / (ST * U)
    s = init_state(cpl)
    s = warm(s, int(warm_periods * T)); jax.block_until_ready(s["f"])
    s, (qs, fuy) = coll(s, n_snap); jax.block_until_ready(qs)
    return np.asarray(qs), np.asarray(fuy), stride   # qs (n,3,64,24), fuy (n,64,24)


def perc_err(pred, truth, dt):
    """Distance perceptuelle entre deux signaux locaux : (écart fréq dominante, écart amplitude)."""
    band = (0.3 * ST * U / D, 3 * ST * U / D)
    ft, _ = dominant_frequency(truth, dt=dt, band=band)
    fp, _ = dominant_frequency(pred, dt=dt, band=band)
    at, ap = np.std(truth), np.std(pred)
    ef = abs(fp - ft) / ft if (np.isfinite(ft) and ft > 0 and np.isfinite(fp)) else 1.0
    ea = abs(ap - at) / (at + 1e-12)
    return ef, ea


def analyze(qs, fuy, stride):
    n, _, wx, wy = qs.shape
    coarse_uy = qs[:, 2]                                   # (n,wx,wy) q_y grossier = ρu_y ; ~u_y (ρ≈1)
    amp = fuy.std(axis=0)                                  # amplitude locale du signal fin
    active = amp > np.quantile(amp, 0.80)                  # 20 % fenêtres les plus énergétiques
    # clés halo (3×3 grossier) pour mémoïser
    qp = np.pad(qs, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    keys = np.stack([qp[:, :, i:i+wx, j:j+wy] for i in range(3) for j in range(3)], axis=1)
    keys = keys.reshape(n, 9 * 3, wx, wy)
    m = n // 2
    counts = {"memoize": 0, "expert": 0, "descend": 0}
    nact = 0
    for i in range(wx):
        for j in range(wy):
            if not active[i, j]:
                continue
            nact += 1
            truth = fuy[m:, i, j]
            # expert = contenu grossier (reconstruit au mieux par un ROM grossier)
            exp = coarse_uy[m:, i, j]
            ef_e, ea_e = perc_err(exp, truth, stride)
            # mémoïser = réutilisation clé-halo du fin passé (cache = 1ère moitié)
            kt = keys[m:, :, i, j]; kc = keys[:m, :, i, j]
            d2 = ((kt[:, None, :] - kc[None, :, :]) ** 2).sum(-1)
            nn = np.argmin(d2, axis=1)
            mem = fuy[:m, i, j][nn]
            ef_m, ea_m = perc_err(mem, truth, stride)
            mem_ok = ef_m < JND_F and ea_m < JND_A
            exp_ok = ef_e < JND_F and ea_e < JND_A
            if mem_ok:
                counts["memoize"] += 1
            elif exp_ok:
                counts["expert"] += 1
            else:
                counts["descend"] += 1                    # ni mémoïser ni expert -> descendre
    return counts, nact


def main():
    for Ur in [4.5, 8.5]:
        t0 = time.time()
        qs, fuy, stride = collect(Ur)
        counts, nact = analyze(qs, fuy, stride)
        print(f"\n===== Ur={Ur} (point candidat) — {nact} fenêtres actives ({time.time()-t0:.0f}s) =====")
        for a in ("memoize", "expert", "descend"):
            print(f"  {a:8} : {100*counts[a]/max(nact,1):5.1f}%")
        pd = 100 * counts["descend"] / max(nact, 1)
        pr = 100 * (counts["expert"] + counts["descend"]) / max(nact, 1)
        print(f"  -> non-fermable PERCEPTUEL (besoin expert OU descend) = {pr:.1f}%")
        if pd > 5:
            print(f"  => C2 TERNAIRE : {pd:.1f}% des fenêtres énergétiques exigent DESCEND ->"
                  " chirurgie grossier↔fin OBLIGATOIRE (c'est le test complet de l'archi).")
        elif pr > 5:
            print(f"  => C2 BINAIRE perceptuel : l'expert rattrape, descend dormant -> "
                  "composite grossier OK MAIS juger contre le fin.")
        else:
            print("  => non-fermable perceptuel ~0 : C2 VACANT à ce point (router ne paie pas perceptuellement).")


if __name__ == "__main__":
    main()
