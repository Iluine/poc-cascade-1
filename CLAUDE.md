# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Nature du projet — lire avant tout

Ce dépôt n'est **pas** un produit logiciel : c'est un **PoC de recherche pré-enregistré** (« Test
T1 »). Il teste un seul claim (un routeur deux-physiques sur interface FSI contestée bat-il un
routeur de Harten codé en dur, en espace perceptuel, à compute inférieur ?). Le code est
l'instrument ; le **livrable est un verdict honnête**, pas une fonctionnalité.

- **`PREREGISTRATION.md` est le contrat et fait foi.** Seuils et critères figés *avant* le code.
  **Un seuil manqué est un résultat, pas un bug à corriger a posteriori.** Ne jamais déplacer un
  seuil pour faire passer un gate.
- Le **journal des verdicts** (fin de `PREREGISTRATION.md`) est **append-only** : chaque entrée est
  un fait daté. On l'enrichit, on ne le réécrit pas.
- `README.md` donne la carte d'architecture et l'ordre d'exécution gaté.

## Discipline épistémique (load-bearing — prime sur les réflexes d'ingénierie)

Ces règles ont produit les vrais résultats du projet ; les enfreindre fabrique de faux PASS.

1. **G0/G0' valident l'INSTRUMENT ; C1–C4 valident l'ARCHITECTURE. Jamais l'inverse.** Le verdict
   « ça vaut le coup de continuer » vit en C1–C4, jamais à un gate instrument.
2. **JAMAIS de L2 point-à-point comme critère.** La fidélité se juge en **espace perceptuel** (M1–M4 :
   spectre, Strouhal, enveloppe, cohérence de phase). La L2-cache sur-compte (elle « trouve » du
   signal sous le seuil de perception) — elle a failli fabriquer un C2 PASS contaminé. La métrique
   doit voir *ce que le verdict tranche*.
3. **Le moins cher qui peut échouer, d'abord.** Avant tout gros build, écrire la mesure cheap qui
   peut falsifier la testabilité (c'est ce qui a tué C2 pour ~5 min de compute, avant la chirurgie).
4. **Anti-circularité.** Les régimes (`cascade/experts/regimes.py`) sont définis par la PHYSIQUE
   (vorticité / critère-Q / marqueur de structure) — **jamais** par le détail de Harten, sinon le
   routeur nul devient un classificateur parfait par construction et C2 est invérifiable.
5. **Null-first.** Hypothèse nulle (bus minimal `(ρ, ρu)`, DMD brut `stabilize=False`,
   `make_cylinder(fill=True)` sans enrichissement) ; on n'enrichit que si un C condamne, et seulement
   là où l'artefact apparaît.
6. **Ne pas surclamer**, dans aucun sens. Documenter la portée EXACTE d'un verdict (cf. la conclusion
   T1 : verdict réel mais sur 2 sorties de routeur sur 3 — la 3ᵉ, `descend`, n'a jamais été sollicitée).

> Statut au dernier commit : **T1 clos.** Instrument FSI validé (G0 a/b/c) ; **C2 perceptuellement
> vacant à Re=100 2D** (substrat (quasi-)périodique → mémoïser-partout perceptuellement optimal).
> Suite = **T1.5** : trouver un substrat qui SOLLICITE les trois sorties. `experiments/mixed_substrate.py`
> (non commité) est le WIP T1.5. Lire la conclusion de `PREREGISTRATION.md` avant de continuer.

## Commandes

L'environnement est un venv `uv` en Python 3.12. **Toujours appeler les binaires via `.venv/bin/`**
(pas de `python` global) — JAX y est configuré pour le GPU.

```bash
# Setup (cf. README §Setup ; XLB/jax-fem ajoutés ensuite)
uv venv --python 3.12 .venv
uv pip install --python .venv -e .          # installe cascade + deps de pyproject.toml

# Tests (invariants mathématiques — doivent passer net, pas de tolérance arbitraire)
.venv/bin/pytest                            # toute la suite (peskin, harten, regimes, pod_dmd)
.venv/bin/pytest tests/test_peskin.py       # un fichier
.venv/bin/pytest tests/test_peskin.py::test_phi4_partition_of_unity   # un test
.venv/bin/pytest -k partition               # par motif

# Lint
.venv/bin/ruff check .                      # line-length 100, target py312

# Expériences (GPU requis ; chacune a un main() autonome, runs longs ~min)
.venv/bin/python experiments/g0_spring/g0a_strouhal.py    # un gate / claim
.venv/bin/python experiments/c2_action_perceptual.py
```

Les tests activent `jax_enable_x64` (précision double pour vérifier les invariants). Les expériences
tournent en FP32 sur GPU.

## Architecture

### Ce qui est construit vs planifié

`cascade/` annonce 8 sous-modules (README, `cascade/__init__.py`) ; **5 sont implémentés**, les autres
sont des dossiers **vides** (planifiés, non atteints car C2 vacant en amont) :

| Module | État | Rôle |
|---|---|---|
| `cascade/fluid/` | ✅ | Oracle fluide LBM D2Q9 (wrapper du stepper JAX de **XLB**) |
| `cascade/coupling/` | ✅ | `peskin.py` (noyau IB delta-4pts) + `ib_lbm.py` (pas couplé + EDO ressort) |
| `cascade/harten/` | ✅ | MRA dyadique JAX ; `window_features` = ce que voit le routeur |
| `cascade/experts/` | ✅ | `regimes.py` (régimes physiques) + `pod_dmd.py` (surrogates POD+DMD) |
| `cascade/metrics/` | ✅ | `spectral.py` — harnais perceptuel M1–M4 (numpy/scipy, hors graphe) |
| `cascade/solid/` | ⬜ vide | wrapper jax-fem (config poteau, T1.5) |
| `cascade/router/` | ⬜ vide | les 3 sorties ; le routeur nul est codé inline dans les expériences C2 |
| `cascade/accounting/` | ⬜ vide | FLOPs + wall-clock (requis pour C3) |
| `cascade/configs/` | ⬜ vide | configs ressort / poteau |

### Flux de données (graphe différentiable unique)

Tout le pas couplé est en **JAX pur** → un seul graphe différentiable de bout en bout (c'est le levier
§7 pour une éventuelle Phase 2 ; confirmé : `grad/disp_y0=122.9`). Pas couplé (`ib_lbm.coupled_step`) :

```
(ρ,u)=macroscopic(f) → interpolate(u, marqueurs) → F_L=U_corps−u_b (forçage direct)
→ g=spread(F_L·ds)  → injection exact-difference dans f → fl.step (XLB stream+collide+BC)
→ F_hydro=−Σ F_L·ds (Newton 3) → EDO ressort (Euler semi-implicite) → nouvelle position
```

`rollout` empile via `jax.lax.scan`. Le forçage IB est injecté **par-dessus** le stepper XLB (XLB ne
gère qu'une force constante) ; on n'utilise PAS l'`IBMStepper` Warp natif (non-différentiable JAX).

### Conventions invariantes (respecter partout)

- **Layout champs** : `(C, nx, ny)`, **axe 1 = x (sens du courant), axe 2 = y**. Marqueurs lagrangiens
  `(M, 2)` en coordonnées continues.
- **Unités réseau LBM** : `dx = dt = 1`. `ν = c_s²(τ−1/2)`, `c_s²=1/3`, `ω=1/τ`. `Re = u_in·D/ν`.
- **Le routeur ne voit QUE le halo grossier + le détail de Harten** (`window_features`, §6). Les
  régimes/vorticité sont la vérité-terrain que le routeur ne voit jamais.
- **Bus minimal = `(ρ, ρu)`** (hypothèse nulle ; pas de flux conservatif ni de moments).
- **`make_cylinder(fill=True)`** (forçage VOLUMIQUE) est le défaut correct : l'anneau-frontière sur
  corps épais fait un intérieur poreux → portance ×4 (cause racine diagnostiquée). `fill=False` =
  diagnostic seulement. Corps mobile : la masse interne piégée est corrigée via `m_eff = m − ρ_f·πR²`.

### Expériences

`experiments/g0_spring/` = gates instrument (G0 a/b/c, diagnostics force/masse ajoutée).
`experiments/c1_experts.py`, `c2_*.py` = claims d'architecture. Chaque fichier a un `main()` autonome,
un docstring qui énonce le claim/la mesure, et écrit son verdict dans le journal de `PREREGISTRATION.md`.

## Contraintes load-bearing (ne pas casser)

- **`warp-lang==1.10.0` est épinglé et critique.** XLB 0.3.1 importe `warp.utils.ScopedTimer`, déplacé
  en `warp._src.utils` à partir de warp-lang≥1.13. **Ne PAS dé-épingler sans patcher XLB** (inscrit dans
  `pyproject.toml`).
- **GPU cible** : RTX 3050 Ti 4 Go, CUDA 12.8 → `jax[cuda12]==0.10.2` (compatible driver 570).
- **`petsc4py` est différé** (pas de wheel manylinux ; build PETSc échoue). Non bloquant : la config
  **ressort** n'utilise pas jax-fem (EDO masse-ressort en JAX). N'est requis que pour la config **poteau**.
- **`experiments/mixed_substrate.py` (WIP T1.5) dépend de `scikit-learn`** (NearestNeighbors),
  déclaré dans `pyproject.toml` mais hors de la stack cœur — `uv pip install` le tirera.
- **Commentaires, docstrings et messages = en français.** Conserver cette langue dans tout nouveau code.
