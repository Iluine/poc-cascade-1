"""ÉTAPE (2) — piner la masse interne ajoutée AVANT la boucle VIV (feedback).

Le forçage volumique piège le fluide intérieur (masse ρ_f·V). À a_corps≠0, la réaction
F_IB inclut −(m_interne + m_ajoutée)·a. On l'isole par oscillation IMPOSÉE en fluide
quasi-quiescent (bas KC → quasi pas de lâcher) :

  F_y(t) ≈ −M_eff·a(t) − C·v(t),  avec  M_eff = m_interne + m_ajoutée_physique.

Attendu : M_eff ≈ 2·ρ_f·πR²  (m_interne=ρ_f·V, m_ajoutée=ρ_f·C_a·V, C_a≈1 pour un cylindre).
Conséquence G0b : EDO de structure avec m_eff = m_corps − m_interne (=ρ_f·V mesuré).
"""

import functools, math
import jax, jax.numpy as jnp, numpy as np

from cascade.fluid import build_fluid, FluidConfig
from cascade.coupling.ib_lbm import make_cylinder, Spring, Coupled, init_state, rollout


def main(nx=512, ny=192, D=20.0, A=3.0, Tosc=2000, n_periods=7, iters=3):
    R = D / 2
    rho_V = math.pi * R ** 2  # masse de fluide intérieur théorique (ρ_f=1)
    cfg = FluidConfig(nx=nx, ny=ny, u_in=0.08, inflow=0.0, diameter=D, reynolds=100,
                      walls="bounceback", seed_perturb=0.0)  # fluide quiescent, ν fixé par u_in/Re
    fl = build_fluid(cfg)
    body = make_cylinder(cx=nx//3, cy=ny/2, radius=R, fill=True)
    w = 2 * math.pi / Tosc
    vmax = A * w
    KC = vmax * Tosc / D
    Re_osc = vmax * D / cfg.nu
    print(f"M={body.base.shape[0]} ρ_f·V≈{rho_V:.1f}  A={A} Tosc={Tosc} ω={w:.3e}  KC={KC:.2f}  Re_osc={Re_osc:.1f}")

    def prescribed(t):
        tt = t.astype(jnp.float32)
        disp = jnp.stack([jnp.float32(0.0), A * jnp.sin(w * tt)])
        vel = jnp.stack([jnp.float32(0.0), A * w * jnp.cos(w * tt)])
        return disp, vel

    fixed = Spring(m=1.0, k=0.0, c=0.0, dof_mask=jnp.zeros(2, jnp.float32))
    cpl = Coupled(fluid=fl, body=body, spring=fixed, n_forcing_iters=iters,
                  probe_xy=(nx//2, ny//2), prescribed=prescribed)
    roll = jax.jit(functools.partial(rollout, cpl), static_argnames=("n_steps",))
    N = n_periods * Tosc
    state = init_state(cpl)
    state, d = roll(state, n_steps=N); jax.block_until_ready(d)
    Fy = np.asarray(d["lift"])

    # --- mesure DIRECTE de m_interne au pic de vitesse (fin : t=N=k·Tosc => v=pic, corps centré) ---
    v_peak = A * w  # cos(2πk)=1
    rho_f, u_f = fl.macroscopic(state["f"]); u_f = np.asarray(u_f)
    cx, cy = nx // 3, ny / 2
    xs = np.arange(nx)[:, None]; ys = np.arange(ny)[None, :]
    r2 = (xs - cx) ** 2 + (ys - cy) ** 2
    print("\nmomentum-y cumulé / v_pic par rayon (plateau = m_interne piégée) :")
    prev = 0.0
    for r in [4, 6, 8, 10, 11, 12, 13, 14, 16, 20]:
        m_cum = float(np.sum(u_f[1][r2 < r * r])) / v_peak
        print(f"  r={r:>2}  m(r)/v = {m_cum:7.1f}  (Δ={m_cum-prev:+.1f}, aire disque={math.pi*r*r:.0f})")
        prev = m_cum

    # ajustement F_y = α sin(ωt) + γ cos(ωt) sur périodes entières (saute 1 période de démarrage)
    t = np.arange(N)
    win = slice(Tosc, N)
    s, c = np.sin(w * t[win]), np.cos(w * t[win])
    Amat = np.stack([s, c], axis=1)
    (alpha, gamma), *_ = np.linalg.lstsq(Amat, Fy[win], rcond=None)
    # a(t) = -A ω² sin(ωt) ; -M_eff a = +M_eff A ω² sin -> α = M_eff A ω²
    M_eff = alpha / (A * w ** 2)
    C = -gamma / (A * w)
    m_added_phys = M_eff - rho_V
    C_a = m_added_phys / rho_V
    resid = Fy[win] - Amat @ np.array([alpha, gamma])
    fit_quality = 1 - np.var(resid) / np.var(Fy[win])

    print("\n========== MASSE AJOUTÉE ==========")
    print(f"qualité d'ajustement (R²) : {fit_quality:.3f}  (proche de 1 => F dominé par a et v)")
    print(f"M_eff (coef d'accélération de F_IB) = {M_eff:.1f}")
    print(f"  attendu 2·ρ_f·V = {2*rho_V:.1f}  (interne {rho_V:.0f} + ajoutée physique)")
    print(f"m_ajoutée physique = M_eff − ρ_f·V = {m_added_phys:.1f}  => C_a = {C_a:.3f}  (théorie cylindre = 1)")
    print(f"amortissement C = {C:.2f}")
    print(f"\n=> CORRECTION G0b : m_eff = m_corps − {rho_V:.1f} (ρ_f·V)")


if __name__ == "__main__":
    main()
