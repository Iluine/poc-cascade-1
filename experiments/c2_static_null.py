"""C2 statique (diagnostic stratifié) — le JUGE : nul de Harten CALIBRÉ + zone de divergence.

Thèse C2 : le halo grossier porte de l'info de routage que le détail seul n'a pas. Elle se
prouve UNIQUEMENT dans la zone où détail et vorticité DIVERGENT — ailleurs (≈90 % convectif
trivial) nul et appris décident pareil. Une moyenne domaine MENT par dilution (faux négatif).

Donc on stratifie. Décision binaire ici (descend dormant : aucun régime non-représentable
à Re=100 2D — acté). Action juste : vortical → expert ; non-vortical → mémoïser.

- Nul CALIBRÉ : meilleur seuil de DÉTAIL (sweep) pour prédire l'action juste (le juge).
- Zone de divergence : fenêtres où le détail se trompe (détail bas mais vortical = cisaillement
  manqué ; détail haut mais lisse = expert gaspillé). On rapporte sa fraction de domaine ET
  l'erreur du nul calibré dedans — c'est là, et là seulement, que le halo peut battre le nul.
- Préview séparabilité halo : le halo distingue-t-il les fenêtres mal classées par le détail ?
  (indique si le routeur appris vaut la peine — construit ensuite si la zone est non-vide).
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout
from cascade.experts import regimes as R

U, D = 0.08, 20.0
ST = 0.188
K = 0.0578
M_EFF = 5.0 * D ** 2 - np.pi * (D / 2) ** 2
WIN = 16   # taille de fenêtre de routage (cellules fines)


def main(nx=512, ny=192, stride=40, n_snap=500, warm_periods=35):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=100, walls="bounceback", seed_perturb=1e-3)
    fl = build_fluid(cfg)
    cx, cy = nx // 3, ny / 2
    body = make_cylinder(cx=cx, cy=cy, radius=D / 2, fill=True)
    spring = Spring(m=float(M_EFF), k=K, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=3, probe_xy=(nx//2, ny//2))
    wx, wy = nx // WIN, ny // WIN

    def per_window(state):
        rho, u = fl.macroscopic(state["f"])
        q = jnp.concatenate([rho, rho * u], axis=0)            # (3,nx,ny)
        qm = q.reshape(3, wx, WIN, wy, WIN)
        halo = qm.mean(axis=(2, 4))                            # (3,wx,wy) — ce que l'appris verra
        detail = jnp.sqrt(((qm - halo[:, :, None, :, None]) ** 2).mean(axis=(0, 2, 4)))  # (wx,wy)
        vmax = jnp.abs(R.vorticity(u)).reshape(wx, WIN, wy, WIN).max(axis=(1, 3))         # (wx,wy)
        return halo, detail, vmax

    @functools.partial(jax.jit, static_argnames=("n",))
    def warm(state, n):
        state, _ = rollout(cpl, state, n); return state

    @functools.partial(jax.jit, static_argnames=("ncol",))
    def collect(state, ncol):
        def step(st, _):
            st, _ = rollout(cpl, st, stride)
            return st, per_window(st)
        st, outs = jax.lax.scan(step, state, None, length=ncol)
        return st, outs

    T = D / (ST * U)
    t0 = time.time()
    state = init_state(cpl)
    state = warm(state, int(warm_periods * T)); jax.block_until_ready(state["f"])
    state, (halo, detail, vmax) = collect(state, n_snap); jax.block_until_ready(vmax)
    halo = np.asarray(halo); detail = np.asarray(detail); vmax = np.asarray(vmax)
    print(f"collecte : {n_snap} snaps × {wx}×{wy} fenêtres = {n_snap*wx*wy} instances ({time.time()-t0:.0f}s)")

    # aplatir en instances (toutes fenêtres × snapshots)
    det = detail.reshape(-1)
    vor = vmax.reshape(-1)
    halo_f = halo.transpose(0, 1, 2, 3).reshape(3, -1).T   # (N,3)

    # vérité-terrain action : vortical (expert) si vorticité > seuil physique (quantile)
    vthr = np.quantile(vor, 0.80)        # 80e percentile = la queue vorticale (sillage/cisaillement)
    vortical = vor > vthr                 # True => expert juste ; False => mémoïser juste

    # NUL CALIBRÉ : meilleur seuil de détail pour prédire 'vortical'
    grid = np.quantile(det, np.linspace(0.5, 0.99, 60))
    best = None
    for t in grid:
        pred = det > t
        acc = np.mean(pred == vortical)
        if best is None or acc > best[1]:
            best = (t, acc, pred)
    dthr, null_acc, null_pred = best
    # erreurs du nul, décomposées
    missed = (~null_pred) & vortical       # détail bas mais vortical : cisaillement MANQUÉ
    wasted = (null_pred) & (~vortical)     # détail haut mais lisse : expert GASPILLÉ
    div_zone = missed | wasted             # zone où le détail seul se trompe
    frac_div = np.mean(div_zone)

    print("\n========== NUL DE HARTEN CALIBRÉ (le juge) ==========")
    print(f"seuil détail optimal = {dthr:.4e}  | seuil vorticité (q80) = {vthr:.4e}")
    print(f"exactitude nul (domaine entier) = {100*null_acc:.1f}%  <- dominée par le trivial")
    print(f"ZONE DE DIVERGENCE (détail seul se trompe) : {100*frac_div:.2f}% du domaine")
    print(f"  cisaillement manqué (détail bas, vortical) : {100*np.mean(missed):.2f}% (=> dérive)")
    print(f"  expert gaspillé   (détail haut, lisse)     : {100*np.mean(wasted):.2f}% (=> compute perdu)")

    # PREVIEW séparabilité halo : le halo distingue-t-il les fenêtres de divergence ?
    # (régression logistique halo -> vortical, exactitude SUR la zone de divergence vs le nul)
    if frac_div > 0.001:
        from numpy.linalg import lstsq
        Xn = (halo_f - halo_f.mean(0)) / (halo_f.std(0) + 1e-9)
        X = np.concatenate([Xn, det[:, None] / (det.std() + 1e-9), np.ones((len(det), 1))], axis=1)
        y = vortical.astype(np.float64)
        w, *_ = lstsq(X, y, rcond=None)         # proxy linéaire (LDA-ish) halo+détail
        halo_pred = (X @ w) > 0.5
        acc_div_halo = np.mean(halo_pred[div_zone] == vortical[div_zone])
        acc_div_null = np.mean(null_pred[div_zone] == vortical[div_zone])  # = 0 par définition
        print("\n========== PREVIEW : le halo aide-t-il dans la zone de divergence ? ==========")
        print(f"exactitude SUR la zone de divergence : nul détail = {100*acc_div_null:.1f}% "
              f"| halo+détail (proxy linéaire) = {100*acc_div_halo:.1f}%")
        signal = acc_div_halo > 0.6
        print("=> " + ("SIGNAL : le halo récupère des fenêtres que le détail rate -> routeur appris vaut la peine"
                       if signal else "pas de signal halo (proxy) -> thèse fragile, à confirmer avec un vrai classifieur"))
        print(f"   (ces fenêtres = {100*frac_div:.2f}% du domaine ; leur poids au verdict = forme DYNAMIQUE)")
    else:
        print("\nZONE DE DIVERGENCE QUASI-VIDE -> le détail seul suffit -> thèse morte à bas coût.")


if __name__ == "__main__":
    main()
