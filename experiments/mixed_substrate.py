"""Septième scène — SPÉCIFIÉE sur papier, avec gates de stationnarité et contrôle gel-coupé.

Spec (une contrainte par échec passé) :
 (a) apport entretenu (inflow soutenu)      (b) front qui pénètre par advection (l'inflow avance)
 (c) commit par PASSAGE DU FRONT : une cellule balayée (was_active) vieillit vers committé,
     INDÉPENDAMMENT de sa vitesse (settling-par-vitesse impossible en débit conservé — falsifié
     à 25× drag : committé=0, débit collé à U). Le repos jamais-balayé ne commit jamais.
 (d) gel progressif dans le temps (1/τ) -> relaxation W=U_front·τ   (e) c jamais feature de routage
 (f)+(g1) zone de relaxation STATIONNAIRE = équilibre de flux activation/commit -> W(t) MESURÉE
 (g2) champ de c spatialement LISSE (sinon le bord committé fabrique du monde-3 artefactuel)

Gates :
 1. stationnarité de W(t) : si W s'effondre (commit>alim, échec 3) ou s'emballe (alim>commit,
    échec 1), la scène n'a pas d'équilibre stable -> rapporter l'instabilité, NE PAS mesurer le monde-3.
 2. monde-3 sous DOUBLE contrôle : sensibilité au binning (physique vs fantôme stat) ET run
    GEL-COUPÉ (physique-advectif vs artefact-de-proxy). monde-3 qui disparaît gel-coupé = fabriqué.
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np
from sklearn.neighbors import NearestNeighbors

from cascade.fluid import build_fluid, FluidConfig
from cascade.harten import windowing as H

U = 0.06
TAU = 1000.0
THR_ACT = 0.020       # seuil de PASSAGE DU FRONT : speed>THR_ACT => cellule balayée (was_active)
THR_COM = 0.008       # seuil "actif" pour la sélection de zone du contrôle gel-OFF (collect_instances)
WIN, KCOARSE = 16, 2


def smooth(c):  # lissage spatial léger du champ de commit (g2)
    s = c
    s = 0.5 * s + 0.125 * (jnp.roll(s, 1, 0) + jnp.roll(s, -1, 0) + jnp.roll(s, 1, 1) + jnp.roll(s, -1, 1))
    return s


def run_scene(freeze_on, nx=512, ny=160, n_snap=22, stride=250):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, inflow=U, diameter=20, reynolds=100,
                      walls="bounceback", seed_perturb=1e-3)
    fl = build_fluid(cfg)
    f0 = fl.equilibrium(jnp.ones((1, nx, ny), jnp.float32), jnp.zeros((2, nx, ny), jnp.float32))
    z = jnp.zeros((nx, ny), jnp.float32)

    @jax.jit
    def step(f, fbuf, fz, c, wasact, t):
        a, b = fl.step(f, fbuf, t); f1 = b
        rho, u = fl.macroscopic(f1)
        speed = jnp.sqrt(u[0] ** 2 + u[1] ** 2)
        wasact = jnp.maximum(wasact, (speed > THR_ACT).astype(jnp.float32))  # (c) front a balayé la cellule
        if freeze_on:
            # commit par PASSAGE DU FRONT : une cellule balayée vieillit vers committé au rythme 1/τ,
            # quelle que soit sa vitesse (settling-par-vitesse impossible en débit conservé, falsifié
            # 25× drag). Le gradient de temps-de-passage donne la zone de relaxation W=U_front·τ.
            c_new = jnp.clip(c + (1.0 / TAU) * (2.0 * wasact - 1.0), 0.0, 1.0)  # +1/τ si balayée, −1/τ sinon
            c_new = smooth(c_new)                                 # (g2) champ lisse
            newly = (c_new > 0.5) & (c <= 0.5)
            fz_new = jnp.where(newly[None], f1, fz)               # fige le snapshot au passage de c=0.5
            f_out = (1.0 - c_new)[None] * f1 + c_new[None] * fz_new
        else:
            c_new, fz_new, f_out = c, fz, f1                      # GEL COUPÉ (contrôle)
        return f_out, a, fz_new, c_new, wasact

    @functools.partial(jax.jit, static_argnames=("n",))
    def adv(state, n):
        return jax.lax.fori_loop(0, n, lambda i, s: step(*s, i), state)

    state = (f0, f0.copy(), f0, z, z)
    us, halos, cs, was = [], [], [], []
    for _ in range(n_snap):
        f = state[0]; rho, u = fl.macroscopic(f)
        q = jnp.concatenate([rho, u], axis=0)
        for _ in range(KCOARSE):
            q = H.coarsen(q)
        us.append(np.asarray(u)); halos.append(np.asarray(q))
        cs.append(np.asarray(state[3])); was.append(np.asarray(state[4]))
        state = adv(state, stride)
    return np.stack(us), np.stack(halos), np.stack(cs), np.stack(was)


def cond_resfrac(X, Y, kk, thr=0.3):
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=min(kk, len(Xn))).fit(Xn)
    _, idx = nn.kneighbors(Xn)
    tot = Y.var(0).sum() + 1e-12
    return float((np.array([Y[ind].var(0).sum() / tot for ind in idx]) >= thr).mean())


def collect_instances(us, halos, cs, select_relax, horizons=(1, 4)):
    nx = us.shape[2]; ny = us.shape[3]; wx, wy = nx // WIN, ny // WIN
    S, Hh, FUT = [], [], {h: [] for h in horizons}
    for ti in range(us.shape[0] - max(horizons)):
        for i in range(wx):
            for j in range(wy):
                sl = (slice(i*WIN, (i+1)*WIN), slice(j*WIN, (j+1)*WIN))
                if select_relax is not None:
                    cm = cs[ti][sl].mean()
                    if not (0.05 < cm < 0.95):
                        continue
                else:
                    uw = us[ti][:, sl[0], sl[1]]
                    if np.sqrt(uw[0]**2+uw[1]**2).mean() < THR_COM:   # gel-off : zone active seulement
                        continue
                uw = us[ti][:, sl[0], sl[1]]; spd = np.sqrt(uw[0]**2+uw[1]**2)
                S.append([float(spd.mean()), float(spd.std())])
                ci, cj = i*WIN//(2**KCOARSE), j*WIN//(2**KCOARSE)
                hh = halos[ti][:, max(ci-1,0):ci+2, max(cj-1,0):cj+2].ravel()
                Hh.append(hh[:27] if hh.size >= 27 else np.pad(hh, (0,27-hh.size)))
                for h in horizons:
                    FUT[h].append(us[ti+h][:, sl[0], sl[1]].ravel())
    return np.array(S), np.array(Hh), {h: np.array(v) for h, v in FUT.items()}


def main():
    t0 = time.time()
    us, halos, cs, was = run_scene(freeze_on=True)
    print(f"scène gel-ON collectée en {time.time()-t0:.0f}s")
    # GATE 1 : stationnarité de W (largeur de la zone de relaxation par snapshot)
    W = [float(((c > 0.05) & (c < 0.95)).sum()) / (us.shape[2]*us.shape[3]) for c in cs]
    comm = [float((c > 0.95).mean()) for c in cs]
    spd = np.sqrt(us[:, 0] ** 2 + us[:, 1] ** 2)                  # (T,nx,ny)
    # région ACTIVE soutenue = balayée, encore mouvante, PAS committée (sinon un domaine MORT —
    # tout figé/quiescent — passe le test de W seul : W stationnaire mais zéro dynamique).
    act = [float(((spd[i] > THR_ACT) & (cs[i] <= 0.95)).mean()) for i in range(len(cs))]
    print(f"committé % : {[f'{x:.2f}' for x in comm[::3]]}")
    print(f"W (frac relaxation) : {[f'{x:.3f}' for x in W[::3]]}")
    print(f"actif % (mouvant, non committé) : {[f'{x:.3f}' for x in act[::3]]}")
    Wmid = np.array(W[len(W)//3:])           # après le transitoire (rampe de W sur ~τ)
    comm_final = comm[-1]; act_final = float(np.mean(act[len(act)//3:]))
    W_ok = Wmid.max() > 0.02
    W_cv = float(Wmid.std() / (Wmid.mean() + 1e-9))
    live = act_final > 0.02                                       # une vraie dynamique persiste
    stationary = W_ok and Wmid.mean() > 1e-6 and W_cv < 0.5 and live
    print(f"W moyen(2/3 fin)={Wmid.mean():.3f} cv={W_cv:.2f} committé_final={comm_final:.3f} "
          f"actif_final={act_final:.3f}  -> {'STATIONNAIRE' if stationary else 'NON STATIONNAIRE'}")
    if not stationary:
        # diagnostic depuis committé + actif + W (un domaine MORT — tout figé/quiescent — passe
        # le test de W seul ; le binaire 'W<0.02 => effondrement' confondait aussi rien/tout commit).
        if not live:
            mode = ("DOMAINE MORT (actif≈0) : tout committé-figé ou quiescent, aucune dynamique "
                    "soutenue -> le freeze a étouffé la source, le front cale (effondrement localisé)")
        elif comm_final < 0.02:
            mode = ("RIEN NE COMMIT (committé≈0) : le front ne fige rien -> tout reste actif "
                    "(échec 1 / emballement, le wake revient)")
        elif comm_final > 0.95:
            mode = ("TOUT COMMIT (committé≈1) : saturation, le domaine fige entièrement -> plus de "
                    "région active (échec 3 / effondrement)")
        elif not W_ok:
            mode = ("ZONE DE RELAXATION TROP MINCE (W<0.02) : transition committé<->actif quasi nulle "
                    "-> séparation triviale, pas de mélange à router")
        else:
            mode = (f"W INSTABLE (cv={W_cv:.2f}>0.5) : équilibre activation/commit présent mais oscillant "
                    "-> pas de largeur stationnaire")
        print(f"=> GATE 1 ÉCHEC : {mode}.")
        print("   Instabilité NOMMÉE (pas 'gros composant' vague) -> ne pas mesurer le monde-3 "
              "ni comparer les routeurs.")
        return

    # GATE 2 : monde-3 sous sensibilité binning + contrôle gel-coupé
    S, Hh, FUT = collect_instances(us, halos, cs, select_relax=True)
    print(f"\ninstances zone relaxation (gel-ON) : {len(S)}")
    if len(S) < 150:
        print("=> trop peu d'instances."); return
    us2, halos2, cs2, _ = run_scene(freeze_on=False)
    S0, Hh0, FUT0 = collect_instances(us2, halos2, cs2, select_relax=None)  # zone active, gel-coupé
    print(f"instances zone active (gel-OFF, contrôle) : {len(S0)}")

    print(f"\n{'horizon':>7} {'kNN':>5} {'monde3% gel-ON':>15} {'monde3% gel-OFF':>16}")
    for h in (1, 4):
        for kk in (12, 24):
            w3_on = cond_resfrac(Hh, FUT[h], kk)
            w3_off = cond_resfrac(Hh0, FUT0[h], kk) if len(S0) > 50 else float('nan')
            print(f"{h:>7} {kk:>5} {100*w3_on:>14.1f}% {100*w3_off:>15.1f}%")
    print("\nLecture : monde-3 PHYSIQUE si il survit gel-OFF ET croît/stable avec l'horizon (advectif).")
    print("monde-3 qui s'effondre gel-OFF = ARTEFACT du proxy de gel (seuil). Sensible au binning = fantôme stat.")


if __name__ == "__main__":
    main()
