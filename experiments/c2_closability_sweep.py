"""Balayage de FERMABILITÉ vs Ur — la mesure qui débloque ou enterre C2 (feedback option 4).

closability = cache validity : mémoïser-clé-halo est valide ssi le halo grossier prédit
l'avance fine. Périodique (lock-in) → halo = clé parfaite (récurrence exacte) → tout fermable
→ C2 vacant. Quasi-périodique (bord de bande) → la phase relative des 2 fréquences est jetée
par le coarsening → halo insuffisant → non-fermable → le routage a une vraie décision.

On balaie Ur à travers la transition, on mesure par fenêtre l'erreur mémoïser plus-proche-clé,
on trace la fraction de fenêtres ACTIVES non-fermables vs Ur :
- PLATEAU GRADUÉ (zone non-fermable croît continûment) → C2 testable, le point d'op est dessus.
- MARCHE BRUTALE (tout fermable -> tout non-fermable, sans intermédiaire) → C2 structurellement
  vacant à ce Re → option 3 (acter la limite) JUSTIFIÉE, prouvée et non supposée.

Réutilise la machinerie G0b (run par Ur) + une seule mesure ajoutée. Pas de re-grille, pas de
changement de Re, G0 intact.
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.harten import windowing as H

U, D = 0.08, 20.0
ST = 0.188
UR_CENTER = 1.0 / ST          # 5.32
K_CENTER = 0.0578
M_EFF = 5.0 * D ** 2 - np.pi * (D / 2) ** 2
LEVELS = 3                    # grille fenêtre = 512/8 × 192/8 = 64 × 24


def collect_coarse(Ur, nx=512, ny=192, stride=40, n_snap=400, warm_periods=35):
    k = K_CENTER * (UR_CENTER / Ur) ** 2
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=100, walls="bounceback", seed_perturb=1e-3)
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=D/2, fill=True)
    spring = Spring(m=float(M_EFF), k=k, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=3, probe_xy=(nx//2, ny//2))

    def cq(state):
        rho, u = fl.macroscopic(state["f"])
        q = jnp.concatenate([rho, rho * u], axis=0)
        for _ in range(LEVELS):
            q = H.coarsen(q)
        return q                                  # (3, 64, 24)

    @functools.partial(jax.jit, static_argnames=("n",))
    def warm(state, n):
        state, _ = rollout(cpl, state, n); return state

    @functools.partial(jax.jit, static_argnames=("ncol",))
    def collect(state, ncol):
        def step(st, _):
            st, _ = rollout(cpl, st, stride)
            return st, cq(st)
        st, qs = jax.lax.scan(step, state, None, length=ncol)
        return st, qs

    T = D / (ST * U)
    state = init_state(cpl)
    state = warm(state, int(warm_periods * T)); jax.block_until_ready(state["f"])
    state, qs = collect(state, n_snap); jax.block_until_ready(qs)
    return np.asarray(qs)                          # (n_snap, 3, 64, 24)


def closable_fraction(qs, rel_thr=0.25):
    """Fraction de fenêtres ACTIVES où mémoïser plus-proche-clé prédit l'avance (fermable).

    key(t) = halo 3×3 de q grossier autour de la fenêtre ; advance(t)=q(t+1)-q(t) de la fenêtre.
    cache = 1ère moitié ; test = 2e moitié ; err = ||advance_test - advance(plus proche clé cache)||.
    fermable si médiane(err) < rel_thr · std(advance). Actives = top variance d'avance.
    """
    n, C, wx, wy = qs.shape
    adv = qs[1:] - qs[:-1]                         # (n-1, C, wx, wy)
    # halo 3×3 par fenêtre (clé), padding réflexif
    qp = np.pad(qs[:-1], ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    keys = np.stack([qp[:, :, i:i+wx, j:j+wy] for i in range(3) for j in range(3)], axis=1)
    keys = keys.reshape(n-1, 9 * C, wx, wy)        # (n-1, 9C, wx, wy)
    m = (n - 1) // 2
    act_std = adv.std(axis=0)                      # (C,wx,wy)
    act_mag = np.sqrt((act_std ** 2).sum(axis=0))  # (wx,wy) amplitude d'avance par fenêtre
    active = act_mag > np.quantile(act_mag, 0.80)  # 20 % de fenêtres les plus actives
    closable = np.zeros((wx, wy), bool)
    errs = np.full((wx, wy), np.nan)
    for i in range(wx):
        for j in range(wy):
            if not active[i, j]:
                continue
            kc = keys[:m, :, i, j]                 # cache clés (m, 9C)
            ac = adv[:m, :, i, j]                  # cache avances (m, C)
            kt = keys[m:, :, i, j]                 # test clés
            at = adv[m:, :, i, j]                  # test avances
            d2 = ((kt[:, None, :] - kc[None, :, :]) ** 2).sum(-1)   # (ntest, m)
            nn = np.argmin(d2, axis=1)
            err = np.linalg.norm(at - ac[nn], axis=1)
            sig = np.linalg.norm(at, axis=1).std() + 1e-12
            errs[i, j] = np.median(err) / sig
            closable[i, j] = errs[i, j] < rel_thr
    frac_closable = closable[active].mean()
    return frac_closable, 1.0 - frac_closable, np.nanmedian(errs[active]), int(active.sum())


def main():
    Urs = [4.0, 4.5, 5.32, 6.5, 7.5, 8.5]
    print(f"Ur_center={UR_CENTER:.2f} (lock-in). Fraction NON-fermable des fenêtres actives vs Ur :")
    print(f"\n{'Ur':>5} {'non-fermable%':>13} {'err_méd':>9} {'#actives':>9}")
    rows = []
    for Ur in Urs:
        t0 = time.time()
        qs = collect_coarse(Ur)
        fc, fnc, emed, nact = closable_fraction(qs)
        rows.append((Ur, fnc, emed))
        print(f"{Ur:>5.2f} {100*fnc:>12.1f}% {emed:>9.3f} {nact:>9}  ({time.time()-t0:.0f}s)")
    arr = np.array(rows)
    nf = arr[:, 1]
    span = nf.max() - nf.min()
    # gradué vs marche : la transition s'étale-t-elle sur plusieurs Ur, ou saute-t-elle ?
    print("\n========== VERDICT FERMABILITÉ ==========")
    print(f"non-fermable : min={100*nf.min():.1f}% (centre) max={100*nf.max():.1f}% (bord/hors)")
    if nf.max() < 0.1:
        print("=> tout fermable partout -> C2 STRUCTURELLEMENT VACANT à ce Re (option 3, PROUVÉE).")
    elif span > 0.15 and 0.1 < nf.max() and (np.sort(nf)[-2] - nf.min()) > 0.05:
        amax = arr[np.argmax(nf), 0]
        print(f"=> PLATEAU GRADUÉ : zone non-fermable croît continûment, pic à Ur≈{amax:.2f}.")
        print(f"   C2 TESTABLE -> point d'opération = Ur≈{amax:.2f} (zone non-fermable maximale).")
    else:
        print("=> MARCHE BRUTALE (tout-ou-rien sans intermédiaire) -> pas de routage gradué")
        print("   à ce Re -> option 3 JUSTIFIÉE (prouvée, pas supposée).")


if __name__ == "__main__":
    main()
