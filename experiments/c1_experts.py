"""C1 (complet) — experts isolés sur le substrat qui STRESSE : le flux en LOCK-IN (corps mobile).

Le cylindre fixe était un non-test (cycle limite linéaire = POD+DMD exact par construction).
Ici, corps mobile : la frontière balaie le repère eulérien fixe → structure mobile à grande
largeur de Kolmogorov → une base POD fixe peut casser. C'est le vrai test de représentation.

On compare, par fenêtre fixe :
- WAKE (sillage en aval, hors corps) : régime cisaillement, attendu bas-rang.
- INTERFACE (fenêtre contenant le corps oscillant) : frontière mobile, le test dur.
- SOLIDE : la dynamique de la structure (disp, vel).

Métrique honnête : décroissance d'énergie POD (rang pour 99.9 %) = largeur de Kolmogorov ;
+ rollout 100 T_shed (St, rayon spectral, dérive). Un échec interface parle du DÉCOUPAGE /
de la représentation (n-width), pas forcément de POD+DMD — et c'est un vrai résultat.
"""
import functools, time
import jax, jax.numpy as jnp, numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, spring_from_viv, Spring, Coupled, init_state, rollout
from cascade.harten import windowing as H
from cascade.experts import pod_dmd
from cascade.metrics import dominant_frequency

U, D = 0.08, 20.0
ST = 0.188
F_SHED = ST * U / D
UR = 5.32
K = 0.0578          # ring-down : f_n = f_shed à ce k
M_EFF = 5.0 * D ** 2 - np.pi * (D / 2) ** 2   # m* =5, correction masse interne


def rank_for_energy(S, frac=0.999):
    S0 = S - S.mean(axis=1)[:, None]
    s = np.linalg.svd(S0, compute_uv=False)
    c = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(c, frac) + 1)


def roll_metrics(S, rank, stride, label):
    en = pod_dmd.pod_energy(S, rank)
    rom = pod_dmd.fit(S, rank=rank, dt=float(stride))
    sr = rom.spectral_radius()
    n_roll = int(100 * (D / (ST * U)) / stride)
    roll = rom.rollout(S[:, 0], n_roll)
    finite = bool(np.all(np.isfinite(roll)))
    half = n_roll // 2
    drift = abs(roll[:, half:].std() - roll[:, :half].std()) / (roll[:, :half].std() + 1e-12)
    sig = roll[roll.shape[0] // 2]
    St_r, _ = dominant_frequency(sig, dt=stride, band=(0.3 * F_SHED, 3 * F_SHED))
    St_r = St_r * D / U if np.isfinite(St_r) else np.nan
    r999 = rank_for_energy(S, 0.999)
    # La FRÉQUENCE entre dans le verdict (review 29/07, M14) : St_roll était
    # calculé et imprimé mais jamais jugé — un expert dérivant de 30 % en
    # fréquence passait. Tolérance nommée : ±20 %, la même que G0.
    st_ok = bool(np.isfinite(St_r)) and abs(St_r - ST) / ST <= 0.20
    ok = finite and sr < 1.02 and drift < 0.20 and st_ok
    print(f"  [{label:9}] rang99.9%={r999:>3}  én(r={rank})={en:.4f}  sr={sr:.3f}  "
          f"St_roll={St_r:.4f}  dérive={100*drift:4.1f}%  -> {'ok' if ok else 'CASSE'}")
    return dict(label=label, r999=r999, sr=sr, drift=drift, St=St_r, ok=ok)


def main(nx=512, ny=192, stride=40, n_snap=700, warm_periods=35, rank=30, levels=2):
    cfg = FluidConfig(nx=nx, ny=ny, u_in=U, diameter=D, reynolds=100, walls="bounceback", seed_perturb=1e-3)
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=D/2, fill=True)
    spring = Spring(m=float(M_EFF), k=K, c=0.0, dof_mask=jnp.array([0.0, 1.0], jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=spring, n_forcing_iters=3, probe_xy=(nx//2, ny//2))

    def cq(state):
        rho, u = fl.macroscopic(state["f"])
        q = jnp.concatenate([rho, rho * u], axis=0)
        for _ in range(levels):
            q = H.coarsen(q)
        return q                                  # (3, 128, 48)

    @functools.partial(jax.jit, static_argnames=("n",))
    def warm(state, n):
        state, _ = rollout(cpl, state, n); return state

    @functools.partial(jax.jit, static_argnames=("ncol",))
    def collect(state, ncol):
        def step(st, _):
            st, d = rollout(cpl, st, stride)
            return st, (cq(st), d["disp_y"][-1])
        st, (qs, dys) = jax.lax.scan(step, state, None, length=ncol)
        return st, qs, dys                        # qs (ncol,3,128,48), dys (ncol,)

    T = D / (ST * U)
    t0 = time.time()
    state = init_state(cpl)
    state = warm(state, int(warm_periods * T)); jax.block_until_ready(state["f"])
    state, qs, dys = collect(state, n_snap); jax.block_until_ready(qs)
    qs = np.asarray(qs); dys = np.asarray(dys)
    print(f"collecte lock-in : {qs.shape} snaps, {time.time()-t0:.0f}s  (A_y/D≈{dys.std()*np.sqrt(2)/D:.2f})")

    cgx, cgy = (nx//3)//(2**levels), (ny//2)//(2**levels)   # centre corps en grille grossière
    # fenêtres (grille grossière) : interface autour du corps ; wake en aval
    def win(q, x0, x1, y0, y1):  # q (ncol,3,cx,cy) -> (F, ncol)
        return q[:, :, x0:x1, y0:y1].reshape(q.shape[0], -1).T
    aw = int(np.ceil((D/2 + dys.std()*2 + 4) / (2**levels)))   # demi-largeur interface (grossier)
    S_iface = win(qs, cgx-aw, cgx+aw, cgy-aw, cgy+aw)
    S_wake = win(qs, cgx+3*aw, min(cgx+3*aw+16, qs.shape[2]), cgy-6, cgy+6)

    print("\n========== C1 — experts isolés (lock-in, corps mobile) ==========")
    r_wake = roll_metrics(S_wake, rank, stride, "WAKE")
    r_iface = roll_metrics(S_iface, rank, stride, "INTERFACE")
    # solide : on rapporte la fréquence de disp_y, SEULE chose mesurée ici.
    # (Un `Ssolid = stack(disp, vel)` était construit puis jamais testé —
    # supprimé, review 29/07 M14 : « trivialement bas-rang » est une évidence
    # dimensionnelle (2 DOF), pas le résultat d'un test qui n'a pas tourné.)
    St_solid, _ = dominant_frequency(dys, dt=stride, band=(0.3*F_SHED, 3*F_SHED))
    print(f"  [SOLID    ] DOF=2 (oscillateur)  St={St_solid*D/U:.4f}  (bas-rang par dimension)")

    print("\nLecture : WAKE bas-rang attendu ; INTERFACE = test de largeur de Kolmogorov.")
    print(f"  rang99.9% interface/wake = {r_iface['r999']}/{r_wake['r999']} "
          f"(>>1 => frontière mobile stresse la base POD fixe).")
    verdict = "PASS" if (r_wake['ok'] and r_iface['ok']) else "FAIL (voir interface)"
    print(f"=> C1 : {verdict}")


if __name__ == "__main__":
    main()
