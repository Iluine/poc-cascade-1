# T1 — Pré-enregistration (figée avant le code)

> **Statut :** spécification pré-enregistrée. Critères et seuils figés *avant* le code.
> **Un seuil manqué est un résultat, pas un bug à corriger a posteriori.**

Ce fichier est le contrat. Le texte intégral de la spec T1 fait foi ; ce résumé
en fige les invariants opérationnels et tient le **journal des verdicts**.

---

## Claim global unique (§0)

Un routeur évalué par fenêtre de Harten, sur le seul halo grossier, décide entre
réutiliser un opérateur mémoïsé, invoquer un expert de régime, ou descendre d'un
niveau — de sorte que le `F` composite routé reproduise en **espace perceptuel**
un écoulement fluide-structure couplé (sillage **et** interface), sur un **horizon
long**, à un **compute matériellement inférieur** au surrogate fin partout, et
**mieux qu'un routeur de Harten codé en dur**.

## Séparation non négociable (§0)

- **G0 / G0'** valident l'**INSTRUMENT** (l'oracle FSI est-il juste et sharp).
  Ce ne sont **PAS** des verdicts d'architecture.
- **C1–C4** valident l'**ARCHITECTURE** (le routage paie-t-il). Le verdict
  « ça vaut le coup de continuer » vit ici, **jamais** au gate.

## Hors scope (gelé)

Fracture & persistance (T2) ; 3ᵉ physique & transfert 2→3 (T3) ; calibration fine
de la frontière (2)-vs-(3) ; thermique/chimie ; déformation parquée **scalaire**.

---

## Seuils figés

### Gates instrument G0 / G0' (bloquants, §4)
- **G0a** — Strouhal du sillage dans la fourchette publiée du régime, **±15 %**.
- **G0b** — bande de lock-in existe, onset/offset en vitesse réduite **±20 %**.
- **G0c** — convergence maillage : Strouhal + amplitude stables sous raffinement ×2
  (variation **< seuil perceptuel §8**).
- Échec d'un gate → **la config s'arrête là**.

### Métrique perceptuelle (§8) — JAMAIS L2 comme critère
Seuils **provisoires** (à remplacer par JND, pas à défendre) :
- **M1** spectre fluide : pente plage inertielle (K41) **±10 %** / puissance par bande **±15 %**.
- **M2** lâcher : Strouhal routé vs oracle **±10 %**.
- **M3** structure : fréquence dominante **±10 %** / enveloppe d'amplitude **±20 %**.
- **M4** lock-in : cohérence de phase sillage↔structure **> 0.8** sur la bande de lock-in.

### Horizon (§9)
Rollout **≥ 100 périodes de lâcher `T_shed`**. Transitoire écarté ; métriques sur la
fenêtre stationnaire.

### Claims falsifiables (§10), exécutés dans l'ordre, par config
- **C1** — chaque expert seul sur son régime reste sous seuil sur l'horizon. Échec → stop.
- **C2** — routeur appris bat le nul de Harten : à compute égal fidélité strictement
  meilleure, **ou** à fidélité égale compute strictement inférieur. Marge **> bruit
  inter-seeds (≥ 5 seeds)**.
- **C3** — seuil perceptuel atteint à compute **matériellement inférieur** au fin-partout.
  Cible : **facteur ≥ 3** en FLOPs (placeholder ; > 1 avec marge nette suffit).
- **C4** — pas de dérive énergétique structure, pas de pop d'interface, **lock-in (M4)
  reproduit**.

### « Meilleure config » (§11, anti « chiffres qui me plaisent »)
Celle où le routeur appris bat le nul de Harten par la **marge la plus large et la
plus propre (C2)**, ET où la frontière (2)-vs-(3) est la **plus interprétable**.
La config qui rend tout le monde bon (faux positif du ressort) est la **pire**.

---

## Ce que T1 ne prouve PAS (§13)
Même tout vert : uniquement que la boucle routeur-deux-experts tourne, en perceptuel,
sur l'horizon, avec gain compute, sur interface contestée, dans la meilleure des deux
configs. Rien sur fracture/persistance, 3ᵉ régime/transfert, plafond de fermeture,
généralisation hors substrat. **Ne pas surclamer.**

---

# Journal des verdicts

> Append-only. Chaque entrée = un fait daté, pas une interprétation arrangeante.

### 2026-06-26 — Reconnaissance de fondation
- **Dépôt** vierge confirmé (aucun commit).
- **GPU** : RTX 3050 Ti, 4096 MiB, CUDA 12.8, driver 570.211.01 — conforme à la cible spec.
- **AegirJAX : INEXISTANT** (PyPI 404 sur `aegirjax`/`aegir-jax` ; GitHub 0 résultat).
  → **Repli XLB invoqué** (déclencheur §1 « indisponible » pleinement rempli).
  Conséquence : physique fluide shallow-water → Lattice-Boltzmann ; banc sillage
  autour d'obstacle canonique. Features §6 `profondeur/PE` → `densité/vitesse` LBM.
  L'architecture (Harten/routeur/experts/IB/métriques) est **inchangée** (garanti §1).
- **XLB v0.3.1** disponible ; dépend de `warp-lang` + `jax>=0.8.0` ; extra cuda = `cuda13`.
  → on installe XLB nu + `jax[cuda12]==0.10.2` (compatible driver 570). À CONFIRMER :
  backend JAX différentiable de XLB encore fonctionnel en 0.3.1 (critique Phase 2, §7).
- **jax-fem v0.0.12** disponible (jeune, 0.0.x, aucune dépendance déclarée).
- **jax-cuda12-plugin** publié jusqu'à 0.10.2 → pas de conflit GPU/version.

### 2026-06-26 — F0/F1 : fondation installée et vérifiée
- **F0** : `jax[cuda12]==0.10.2`, backend `gpu`, voit `CudaDevice(id=0)`, matmul sur cuda:0. ✅
- **XLB 0.3.1 importe** UNIQUEMENT avec **`warp-lang==1.10.0`** épinglé (1.14 a déplacé
  `warp.utils.ScopedTimer` → `warp._src.utils`, que XLB 0.3.1 importe encore par l'ancien
  chemin). Contrainte load-bearing, inscrite dans `pyproject.toml`.
- **Backend JAX de XLB = PLEINEMENT DIFFÉRENTIABLE à travers le rollout** (test 20 pas,
  D2Q9 BGK, CI bruitée) : `grad/omega=-0.37` fini non nul ; `grad/CI` norme 95.98,
  36864/36864 éléments finis non nuls. → le levier §7 (graphe JAX unique pour Phase 2)
  **tient avec XLB** ; pas besoin de coder le LBM à la main, l'oracle fluide reste importé.
- **XLB 0.3.x embarque un `IBMStepper`** natif (Warp). On ne l'utilise PAS pour la Phase 2
  (couplé Warp, non-différentiable JAX) : on assemble l'IB en JAX par-dessus le stepper JAX
  (cf. F2), conforme à §5 « assemblée en JAX ».
- **Décision oracle solide** : la config **ressort** (1–2 DDL) n'utilise PAS jax-fem — oracle
  = EDO masse-ressort-amortisseur 2-DDL en JAX (VIV d'un cylindre monté élastiquement,
  canonique). jax-fem (+ basix/gmsh/meshio/petsc4py) n'est requis QUE pour la config
  **poteau** (flèche distribuée). Donc petsc4py ne bloque PAS G0 (ressort) — le premier chiffre.
- **jax-fem importe** (couche `jax_fem.problem.Problem` OK). **petsc4py DIFFÉRÉ** : pas de wheel
  manylinux compatible → tentative de build PETSc échoue (`RuntimeError: 256`). Il n'est importé
  que dans `jax_fem/solver.py` (solveur PETSc) ; usage poteau seulement. Options au moment de G0' :
  wheel/conda, import paresseux patché, ou beam-FEM JAX maison. **Non bloquant pour le ressort.**

### 2026-06-27 — F2 : couplage frontière immergée (Peskin) en JAX
- **Noyau de Peskin** (`cascade/coupling/peskin.py`) : delta 4-points, spread/interpolate
  différentiables. **27/27 tests** : partition de l'unité, 1er moment (reproduction linéaire
  exacte), conservation spread/interp, adjointness, différentiabilité /marqueurs. Piège
  NaN « double-where » du sqrt corrigé (safe-sqrt).
- **Oracle fluide** (`cascade/fluid/lbm.py`) : LBM D2Q9 BGK via stepper JAX XLB. BC §2 =
  entrée vitesse (Regularized) + sortie extrapolation (non-réfléchissante) + parois.
  **Parois bounce-back retenues** (freestream pompe la masse : dérive ρ ; bounce-back →
  ρ_moyen ≈ 1.004 stable à t=2000).
- **Pas couplé IB-LBM** (`cascade/coupling/ib_lbm.py`) : forçage direct exact-difference
  spatialement variable + EDO ressort (1–2 DDL, depuis nombres VIV : m*, Ur, ζ). Monolithique,
  sans remaillage. Réaction = −Σ F_L·ds (Newton 3) → le « bus ».
- **Smoke test cylindre fixe** (256×128, Re=100) : **finite**, masse conservée (ρ=1.019),
  **portance oscille** (lâcher s'amorce), traînée +0.133. **Débit 2648 steps/s** →
  **100·T_shed (125k pas) en ~47 s** : horizon §9 largement faisable, même à 5 seeds.
- **Graphe couplé complet DIFFÉRENTIABLE** (fluide+IB+structure) : `grad/disp_y0=122.9`
  fini non nul → levier §7 (Phase 2) confirmé de bout en bout.
- **Fondation (F0–F2) complète.** Tous les inconnus à plus haut risque sont levés.

### 2026-06-27 — F3 (métriques) + G0a : premier chiffre
- **Harnais perceptuel** (`cascade/metrics/`) : M1 (spectre/pente K41), M2 (Strouhal),
  M3 (enveloppe), M4 (cohérence de phase). Estimation de fréquence sub-bin (périodogramme
  zero-paddé + interpolation parabolique) — Welch seul arrondissait trop (St=0.20 au lieu
  de 0.17 sur signal synthétique ; corrigé à 0.0 % d'erreur). **Harten : non encore construit**
  (requis pour le routeur, pas pour G0).
- **G0a — Strouhal (cylindre fixe)** [premier chiffre qui compte] :
  - β=0.104 (ny=192) : **St=0.192**, écart 16.3 % → FAIL vs réf non-confinée 0.165.
  - β=0.05 (ny=400) : **St=0.185**, écart 12.3 % → **PASS** (±15 %). St ↓ quand β ↓
    → confond **confinement confirmé** (parois no-slip ; XLB n'a pas de BC free-slip).
  - Masse conservée (ρ≈1.003), lâcher établi, champ fini.
- **⚠️ DRAPEAU — calibration de force** : Cd≈1.77–1.89 (réf ~1.33) et surtout **Cl_rms passe
  de 0.30 (β=0.10) à 1.18 (β=0.05)** — amplitude de portance ×4 en agrandissant le domaine.
  La FRÉQUENCE est juste, l'AMPLITUDE/force ne l'est pas. Impact : l'EDO de structure (G0b,
  lock-in, C4) est pilotée par cette force. **Diagnostic prioritaire avant G0b** :
  (1) transitoire saturé ? (cycle limite encore en croissance à β=0.05 → std gonflée) ;
  (2) calibration direct-forcing IB (sur-estimation Cd connue, mais ×4 sur Cl est suspect) ;
  (3) nombre d'itérations de forçage (accumulation de la réaction). À isoler une variable
  à la fois. **G0a (Strouhal) PASSE ; la santé d'instrument côté force reste à établir.**

### 2026-06-27 — Test pivot force + diagnostic IB : cause racine
- **Recadrage (feedback)** : pour un substrat VIV, l'instrument n'est validé que si la
  **force** (canal couplage, terme moteur de `m·ẍ+c·ẋ+k·x=F_L`) est pinée, pas seulement
  St (canal fréquence). Le ×4 sur Cl est LE titre, pas une note de bas de page.
- **Test pivot** (β=0.05, 220k pas ≈ 163 T_shed, `g0_pivot_force.py`) : St_local **stable
  à 0.185** sur tout l'horizon (variable rapide verrouillée). Cl_rms **ni ne sature ni ne
  plateaute** : il **module périodiquement entre ~1.05 et ~2.05** (battement ~44k pas ≈ 32
  T_shed). Ni transitoire propre, ni biais constant → 3e issue : **état quasi-périodique à
  amplitude battue**, niveau ×4–6 trop haut. À Re=100 2D le sillage doit être proprement
  périodique → artefact de couplage, pas physique.
- **Diagnostic IB** (`diag_ib_slip.py`) : glissement résiduel à la coque **0.014–0.033**
  (excellent, converge 0.033→0.004 en 6 itérations → pas de bug d'échelle/adjoint), MAIS
  **vitesse intérieure 0.105–0.237·U** → l'intérieur n'est pas solide. **Cause racine :
  IB frontière-seule (anneau) sur corps épais = intérieur creux/poreux** → corps poreux →
  portance excessive + battement.
- **Correctif : forçage VOLUMIQUE** (`make_cylinder(fill=True)`) — marqueurs tapissant le
  disque, u=U_corps imposé dans tout le solide. Une seule variable changée (anneau→disque).
- **À retenir pour G0b/poteau (corps MOBILE)** : le forçage volumique inclut l'inertie du
  fluide intérieur piégé → corriger la réaction par +ρ_f·V·a_corps (masse interne ajoutée),
  sinon la dynamique VIV (lock-in, amplitude) est biaisée. Sans objet pour G0a (fixe).
- **Note horizon (feedback)** : l'amplitude est la variable LENTE, St la rapide. L'horizon
  §9 calibré sur St peut être trop court pour M3/M4 (enveloppe, lock-in) → impact C4.
  À re-vérifier une fois la force pinée.

### 2026-06-27 — Résolution du canal force : point d'opération β=0.10
- **Le « battement » était un artefact de fenêtre glissante** (W≈6.04 périodes non entières +
  harmoniques impaires → scalloping de la RMS). PSD de la portance β=0.05 : pic de lâcher
  propre + harmoniques impaires (3,5,7,9,11×), **aucune énergie basse-fréquence, aucune
  sideband** → pas de mode lent physique. Leçon : fenêtrer en **périodes entières**.
- **Mais la magnitude est réelle** : Cl_rms vraie (std/160k pas) = **1.62** à β=0.05 (×5).
- **Test décisif** (anneau, blocs de 10 périodes entières, 150k pas) :
  - **β=0.10 : Cl_rms = 0.297 ± 0.002** (plat sur 115 périodes), St=0.192, Cd=1.77.
  - β=0.05 : Cl_rms ≈ 1.6 (×5), Cd≈1.89 (à peine changé).
  → **La force n'est PAS fausse partout (pas un bug de formule)**. L'inflation est
  **dépendante de l'obstruction** : Cd (force moyenne) stable, mais l'oscillation
  transverse du sillage s'emballe en domaine HAUT (β bas) — artefact CL/domaine
  (parois bounce-back lointaines), pas la formule de réaction.
- **POINT D'OPÉRATION RETENU : β=0.10** (cylindre anneau, fixe). Validé sur les **3 canaux**
  vs références CONFINÉES (St≈0.18–0.19 ✓ ; Cl_rms≈0.3–0.5 → 0.297 ✓ ; Cd≈1.4–1.5 + biais
  IB connu ~+35% → 1.77 ✓). **Le canal force qui pilote le lock-in (Cl_rms) est pliné.**
  Mon « PASS » antérieur à β=0.05 était le PIÈGE (gagne St-vs-manuel, casse la force).
- **G0a : PASS sur fréquence ET force, à β=0.10.** Critère de gate du feedback rempli
  (Cl_rms plateauté à la bonne valeur, fenêtre stationnaire vérifiée).
- **Conséquence pré-enregistrée (feedback)** : l'instrument porte un biais de **fréquence**
  de confinement (~+16% : St 0.192 vs 0.165 non-confiné). **G0b accordera la structure sur
  f_shed MESURÉE = 1/1302 ≈ 7.68e-4, PAS sur le St de manuel** — sinon la bande de lock-in
  est décalée par construction.
- **Drapeaux ouverts (non bloquants)** : (1) forçage volumique 8× plus lent (50 vs 1065/s)
  → à optimiser avant les runs longs du corps MOBILE ; (2) anomalie β=0.05 non expliquée
  en détail (documentée comme limite CL/domaine) ; (3) Cd +35% = biais IB attendu.

### 2026-06-27 — Validation sur la géométrie RÉELLE + perf au point d'opération
- **Piège évité (feedback)** : mon G0a était validé sur l'ANNEAU (modèle poreux condamné
  dans la même session). Re-mesure obligatoire sur `fill=True` avant tout.
- **Perf chiffrée AVANT les runs longs** (feedback) : à ny=192 (β=0.10), volume(317) =
  **875–1029 st/s ≈ anneau** (le 50/s était la pression mémoire de la grande grille ny=400,
  abandonnée). Budget §9 : ~3 min/run, **~0.5 h pour 5 seeds × 2 configs**. Faisabilité OK
  sur la géométrie correcte.
- **Re-validation 3 canaux sur cylindre SOLIDE (fill=True), β=0.10, 150k pas** :
  **Cl_rms = 0.340 ± 0.001** (plat/113 périodes), St=0.188, Cd=1.840.
  vs anneau : Cl +14 %, St −2 %, Cd +4 %. → écart MODÉRÉ, les deux dans la fourchette
  confinée. **Pas de compensation d'erreurs** (sinon retirer la porosité aurait fait bouger
  d'un facteur). Valeur retenue = solide. **Anomalie β=0.05 NE ressurgit PAS** ici → close.
- **G0a statique : PASS, plíné sur la géométrie réelle (solide)** : Cl_rms=0.340, St=0.188,
  Cd=1.840, stationnaire. Biais assumés : fréquence +14 % (confinement) ; Cd +38 % (IB).
- **Reste avant la boucle VIV (feedback)** : la masse interne ajoutée (canal couplage caché,
  invisible à a=0). Correction : `m_eff = m_corps − ρ_f·πR²`. À VÉRIFIER par oscillation
  imposée en fluide quasi-quiescent (coef d'accélération de F_IB attendu = 2·ρ_f·πR²).

### 2026-06-27 — Test masse ajoutée (oscillation imposée) : m_interne pinée
- `test_added_mass.py` : oscillation imposée y=A·sin(ωt), A=3, Tosc=2000, KC=0.94,
  Re_osc=11.8, fluide quiescent. Ajustement F_y=−M_eff·a−C·v, **R²=0.97**.
- **M_eff (force) = 917**, soit ~1.5× l'attendu 2·ρ_f·V=628 → C_a apparent = 1.9.
- **Mesure DIRECTE de la masse piégée** (momentum-y cumulé/v_pic par rayon, corps centré) :
  à r=10 (rayon marqueurs) = **306 ≈ πR²=314** → **m_interne = ρ_f·V = 314 PINÉE**
  (pas d'inférence circulaire). Le surplus (603) est la masse ajoutée EXTERNE du sim
  (viscale β_S≈20 → C_a~1.4, + confinement, + part calibration IB), non l'intérieur.
- **Résolution G0b** : `m_eff = m_corps − 314`. L'algèbre → `(m_corps+m_ajoutée_sim)·a =
  F_vortex−kx−cv` : la masse ajoutée s'ANNULE, sa valeur exacte n'est pas requise. Pour
  ne PAS conflater masse ajoutée et lock-in (feedback), **Ur défini sur ω_n MESURÉE
  (ring-down en fluide), pas théorique**.
- Recoupement de confiance : Cl_rms de lâcher = 0.34 colle aux réfs → lecteur de force sain
  pour le VIV (sur-comptage ×1.85 donnerait 0.63). Contrainte m_eff>0 ⇒ m*>π/4≈0.785 (OK, m*~5–10).
- **Étape (2) franchie : m_interne pinée, correction validée. Prêt pour G0b.**

### 2026-06-27 — G0b prérequis : ring-down (linéarité + ω_n)
- `g0b_ringdown.py` : cylindre déplacé/relâché en fluide quiescent, ζ_struct=0, m_eff=m_corps−314.
- **Linéarité (feedback)** : f_n par cycle, amplitude 7.4→0.5 (×15) : 7.50e-4→7.61e-4,
  **dérive 1.4 %** → masse entraînée LINÉAIRE → **cancellation de la masse ajoutée EXACTE**.
  Le dernier doute du canal couplé (C_a non-linéaire à grande amplitude) est clos.
- **ω_n pinée** : à k=0.0578, f_n=7.54e-4 ≈ f_shed0 (St_n=0.188) → centre lock-in Ur≈5.3.
- **ζ_fluide ≈ 0.043** (amortissement fluide même à ζ_struct=0) → établissement enveloppe
  ~20–30 périodes → horizon G0b dimensionné + contrôle de saturation par point Ur (feedback).
- **G0b balayage Ur** (`g0b_sweep.py`) : Ur centré sur f_shed MESURÉE (pas manuel : un
  décalage confinement 14 % rentrerait dans ±20 % → PASS contaminé), m*=5, ζ=0, k variable.

### 2026-06-27 — G0b : bande de lock-in → PASS
- Balayage Ur=[4.0 … 8.5], horizon 150k, m*=5, ζ_struct=0, cylindre solide, m_eff=m_corps−314.
- **Bande lock-in : Ur ∈ [4.5, 7.5]** (onset∈(4.25,4.5], offset∈(7.5,8.5)).
  - **Amplitude** : A_y/D saute ×6.7 à l'entrée (0.080→0.536) et chute ×6.7 à la sortie
    (0.301→0.045). Pic A_y/D≈0.54 (amplitude VIV physique).
  - **Entraînement de fréquence** (le vrai lock-in) : dans la bande f_lift quitte f_shed0
    et suit f_n (4.5:1.246 ↘ 7.5:0.806) ; hors bande il revient vers f_shed0 (8.5:0.954≠f_n 0.626).
- **Saturation confirmée** : sat≤0.04 à TOUS les points, y compris le bord offset → bornes
  réelles, pas des amplitudes en montée (feedback sur la variable lente : résolu par horizon long).
- **M4(sillage↔structure)=1.00 partout** (sonde de sillage indépendante) → l'interface
  couple proprement en phase (le « bus » §8 : PASS). MAIS M4 sature à 1 même hors bande
  (les 2 signaux vivent à la fréquence de lâcher) → **M4 n'est PAS un discriminateur de
  bande** ; la bande vient de l'amplitude + entraînement (méthode VIV standard). Honnête.
- **Verdict** : bornes vs théorie VIV [4,8] (faible m*ζ) → **dans ±20 %**, Ur centré sur
  f_shed MESURÉE (biais confinement retiré) → le ±20 % teste la physique, pas le confinement.
  **G0b PASS.**
- **État G0 (ressort)** : G0a (St/Cl/Cd sur solide) ✓ ; G0b (lock-in) ✓ ; **reste G0c
  (convergence maillage ×2)** — d'autant plus pertinent que le D_eff/biais sont des effets
  de résolution : G0c dira s'ils rétrécissent sous raffinement.

### 2026-06-27 — G0c convergence St : MA THÉORIE D'ERREUR RÉFUTÉE (résultat précieux)
- Prédiction falsifiable testée : biais St ∝ 1/D (racine = étalement Peskin). 3 points
  D=20/30/40 à β=0.104 CONSTANT.
- **St(D) = 0.1897 / 0.1940 / 0.1902** (biais +14.9/+17.6/+15.3 %). **NON-monotone, plat
  à ~0.19 ± 1–2 %. Ne converge PAS vers 0.165.** → **prédiction 1/D RÉFUTÉE** (ratio ΔSt
  observé −0.88 vs +0.50 prédit ; Richardson discordant 6.3 %).
- **Interprétation (la 2e source que le feedback prévoyait)** : les biais ont DEUX causes
  distinctes, que le test sépare :
  - **St (fréquence)** : résolution-INDÉPENDANT → **CONFINEMENT** (β-driven, β constant ici).
    Grid-convergé. Ma théorie Peskin était FAUSSE pour la fréquence.
  - **Cl_rms (force)** : résolution-DÉPENDANT, **0.372→0.342→0.296 (−20 % sur ×2)** vers le
    physique. Ma théorie Peskin était JUSTE pour la force. Effective-diameter qui converge.
  - Physique : fréquence = instabilité globale du sillage (convergée) ; force = résolution
    proche-corps. J'avais tout mis sous un seul « D_eff ∝1/D » — incorrect.
- **G0c-St : PASS** (St stable sous ×2 : +0.3 %, ≪ M2 ±10 % ; biais = confinement physique,
  pas sous-résolution). G0b bornes = fréquentielles → reposent sur une fréquence grid-convergée
  → robustes par construction.
- **Drapeau acté** : Cl_rms porte ~20 % de biais résolution à D=20 ; valeur convergée ~0.30
  (D=40). G0a qualitatif tient (force fourchette physique) ; absolu à corriger du biais résolution.
- **Capacité acquise (feedback)** : je sais maintenant quantifier/séparer biais confinement
  (fréquence, fixe) vs résolution (force, ∝~1/D) → distinguera erreur instrument vs architecture
  quand le routeur tournera.

### 2026-06-27 — G0c amplitude au bord → PASS ; G0 (ressort) COMPLET
- `g0c_bandedge.py` : ring-down D=30 cale k=0.0796 pour Ur=4.5 exact (f_n=5.93e-4), puis
  run corps-mobile au bord d'onset.
- **A_y/D : D=20→0.536, D=30→0.528, ratio 0.986 (−1.4 %)** vs taux force statique 0.919 (−8.1 %).
  → **amplitude quasi grid-invariante, converge PLUS VITE que la force** : le couplage AMORTIT
  l'erreur de force (la structure intègre). **Onset TIENT** (Ur=4.5 verrouillé à D=30, sat=0.00).
  Borne G0b NON résolution-dépendante.
- **G0c : PASS** (fréquence convergée + amplitude grid-invariante au bord + onset stable).

## ✅ G0 (RESSORT) COMPLET — instrument couplé VALIDÉ
- **G0a** ✓ : St=0.188 / Cl_rms=0.340 / Cd=1.840 sur cylindre SOLIDE, vs réfs confinées.
- **G0b** ✓ : bande lock-in Ur∈[4.5,7.5], bornes dans ±20 % théorie VIV, Ur centré f_shed MESURÉE.
- **G0c** ✓ : St grid-convergé (+0.3 %/×2) ; amplitude grid-invariante (−1.4 %), onset stable.
- **Modèle d'erreur compris** : fréquence = confinement (fixe +15 %) ; force = résolution
  (∝~1/D, −20 %/×2) MAIS amortie à −1.4 % dans le couplé. Masse interne pinée (314), cancellation
  linéaire exacte, graphe couplé différentiable.
- **Distinction §0 respectée** : G0 valide l'INSTRUMENT, PAS l'architecture. Le verdict
  « ça vaut le coup » vit en C1–C4, pas ici.
- **→ Feu vert pour la CONTRIBUTION : Harten + routeur + experts.**

### 2026-06-27 — Contribution (1) : fenêtrage de Harten construit
- `cascade/harten/windowing.py` : MRA dyadique en JAX (coarsen 2×2 + coefficient de détail),
  `decompose`/`recompose` inversibles, `detail_magnitude` (indicateur de raffinement par
  fenêtre), `window_features` (halo grossier + détail multi-niveaux agrégé — le routeur ne
  voit QUE ça, §6). **9/9 tests** : reconstruction exacte, conservation moyenne, détail nul
  sur lisse, détail ×7.5 sur structure sous-grille vs lisse, somme nulle/bloc, différentiable.
- Propriété actée : le détail capte la variation INTRA-bloc ; un front aligné sur 2^k remonte
  au niveau grossier → d'où l'agrégation multi-niveaux dans `window_features`.
- Reste de F3 : accounting FLOPs/wall-clock (petit, requis pour C3) — différé jusqu'à C3.

### 2026-06-27 — Contribution (2) : régimes définis par la PHYSIQUE (anti-circularité)
- **Piège évité (feedback critique)** : définir les régimes par les bandes de détail de Harten
  rendrait C2 invérifiable (le routeur nul = seuil de détail serait un classificateur de
  régime parfait par construction → cible = baseline). Les régimes sont donc définis par la
  **vorticité / critère-Q + marqueur de structure** — grandeurs que le routeur NE VOIT PAS.
- `cascade/experts/regimes.py` : CONVECTIVE (|ω|<seuil) / SHEAR (|ω|≥seuil ou Q>0) / INTERFACE
  (proximité corps). Seuil calibré par quantile de |ω| (pas présupposé). **6/6 tests**, dont
  `test_regime_diverges_from_harten_detail` : rotation solide lisse = SHEAR (vorticité) mais
  détail Harten bas → le détail ne définit pas le régime → C2 mesure une vraie hypothèse.
- Nuances actées (feedback) : (a) POD sur les FLUX aux faces directement (pas état-puis-
  différencié, sinon fuite d'interface) ; (b) snapshots = oracle G0, biais d'oracle (confinement
  +15 %, résolution) acté → C1 teste « expert vs ORACLE », pas « expert vs vraie physique » ;
  (c) un échec C1 peut parler du DÉCOUPAGE (prémisse BTX, 1er test domaine) autant que des briques.

### 2026-06-27 — RECADRAGE (feedback) : bus minimal, cap sur C2
- **Dérive corrigée** : un bus à 9 moments MRT « inversible/exact » était la conservation
  retirée plus tôt, ressuscitée — sur-ingénierie déguisée. « Exact » ≠ critère (plausible, pas
  réel ; personne ne mesure la fuite d'un moment à un raccord).
- **Bus MINIMAL = (ρ, ρu)** = hypothèse nulle. PAS de flux conservatif, pas de moments, pas de
  ghosts. POD+DMD directement sur l'état. Annule la nuance (a) ci-dessus (POD sur flux) — c'était
  la même sur-spécification.
- **La richesse de représentation est une décision du ROUTEUR**, pas un choix global a priori.
  Π/enrichissement seulement si **C4 condamne** le bus pauvre, et seulement là où l'artefact
  perceptuel apparaît. Enrichissement chirurgical, jamais par principe.
- **Priorité = C2** (routeur appris bat le nul de Harten) — le seul juge du verdict projet.
  G0 était la route nécessaire ; tout détour loin de C2 est du treadmill. Experts minimaux →
  C1 → C2, au plus près du critère perceptuel.

### 2026-06-27 — Contribution (3) : engine experts + C1 (PASS faible, honnête)
- `cascade/experts/pod_dmd.py` : POD+DMD sur bus pauvre (ρ,ρu), `stabilize=False` (null-first).
  4/4 tests (récupère fréquences, rollout borné, prédiction un-pas).
- **C1-fluide cylindre FIXE = NON-TEST (feedback)** : cycle limite = système linéaire autour
  de l'orbite → POD+DMD exact par construction. Le 0.9995/rang20 = oracle diffusif masqué ;
  c'est le SUBSTRAT (Re=100 2D) qui est trop facile. Refait sur le substrat couplé.
- **C1 complet (lock-in, corps mobile, A_y/D≈0.52)** : 3 experts (wake/interface/solid) tous
  bas-rang, stables sur 100 T_shed, sr=1.000, dérive 0 %. **n-width frontière mobile mesuré :
  interface 12 modes vs wake 7 (×1.7)** — réel mais modeste, le descend n'est pas forcé.
- **PASS sur la lettre (aucun expert ne dérive), confirmation FAIBLE sur l'esprit** : sr=1.000
  partout = même le lock-in couplé reste un cycle limite périodique à Re=100 2D → POD+DMD
  quasi-exact par construction → C1 ne stresse PAS la représentation non-linéaire (faudrait
  Re plus haut). Le vrai juge = C2 (composition + routage). Ne pas surclamer C1 en preuve BTX forte.
- **C2 — invariants de design actés (feedback)** : (a) nul de Harten CALIBRÉ sur les mêmes
  données que l'appris (meilleure politique f(détail seul) vs meilleure f(halo+détail)) — sinon
  on mesure « entraîner vs pas entraîner » ; (b) verdict = DOMINATION de front de Pareto
  fidélité-compute, pas comparaison de 2 points ; (c) coder le nul calibré EN PREMIER (le juge).

### 2026-06-27 — C2 vacant au lock-in périodique ; closability = cache validity
- **Diagnostic statique** (`c2_static_null.py`, nul calibré) : zone de divergence détail/vorticité
  = 4.61 % du domaine (dilution confirmée — la moyenne 95.4 % la noie). MAIS j'avais pris un
  raccourci (vorticité = action juste) que le feedback rejette (l'action = erreur de
  représentation, pas vorticité) → diagnostic en partie artefactuel. Faute à moi.
- **Le mur réel, plus grave** : `mémoïser` = réutiliser l'opérateur en cache, clé = halo grossier.
  Or C1 a montré `sr=1.000`, 0 % dérive → lock-in saturé = **cycle limite parfaitement
  périodique** → le halo **recurre à l'identique chaque période** → **mémoïser-partout
  trivialement quasi-optimal** → les 2 routeurs convergent → **C2 NE DISCRIMINE RIEN**.
  C'est le **§11 « tout le monde bon = la pire config »**, au niveau du substrat. Même cause
  (périodicité) que la faiblesse de C1. Trouvé pour ~5 min de compute, AVANT la chirurgie
  dynamique : le « moins cher qui peut échouer » a falsifié la TESTABILITÉ, pas la thèse.
- **RECADRAGE (feedback) : closability = cache validity.** mémoïser-clé-halo valide ⟺ le halo
  grossier est une clé de cache suffisante. Périodique → clé parfaite → tout fermable. Quasi-
  périodique (bord de bande) → la phase relative des 2 fréquences est jetée par le coarsening
  → halo insuffisant → non-fermable → **le routage a une vraie décision**. Le bord de bande
  EST le premier endroit où la frontière fermable-vs-non-fermable devient non-vide.
- **Action (option 4) : balayer Ur à travers la transition, mesurer la fraction non-fermable
  par fenêtre** (`c2_closability_sweep.py`) → la **mesure empirique de fermabilité** que tout
  l'arc théorique réclamait. Plateau gradué → C2 testable, point d'op = pic non-fermable.
  Marche brutale → C2 structurellement vacant à ce Re → option 3 (acter la limite) PROUVÉE.
  Pas de re-grille, pas de changement de Re, G0 intact.

### 2026-06-27 — Balayage de fermabilité : C2 DÉBLOQUÉ (transition graduée)
- `c2_closability_sweep.py` : fraction non-fermable des fenêtres actives vs Ur :
  Ur=4.0:18.9% / 4.5:18.6% / 5.32:6.2% / 6.5:2.9% / 7.5:6.8% / 8.5:17.3%.
- **Transition GRADUÉE (U-shape), pas une marche** → **C2 TESTABLE** au bord de bande /
  hors-lock-in (Ur≈4.0–4.5 ou 8.5, ~19 % non-fermable). Mélange ~80/20 fermable/non →
  routage a une vraie décision graduée. Point d'op localisé EMPIRIQUEMENT (pas deviné).
- Mécanisme : non-fermabilité ∝ 1/(réponse lock-in). Lock-in profond = mono-fréquence =
  périodique = fermable ; hors-bande = f_n & f_shed coexistent = quasi-périodique = non-fermable.
  **La fermabilité EST la frontière (2)-vs-(3), mesurée empiriquement** (acquis théorique).
- Caveat : absolu (~19 %) peut être gonflé par cache fini (12 périodes) ; la TENDANCE
  (bords ≫ centre) est robuste, suffit pour localiser le point d'op.
- **Conséquence build** : descend dormant (C1 : tout représentable) → composite C2 = GROSSIER
  SEUL (mémoïser-ou-expert par fenêtre) → AUCUNE chirurgie populations↔macro. Plus simple que craint.

### 2026-06-28 — VERDICT C2 : perceptuellement VACANT (option 3 prouvée en perceptuel)
- **3 corrections feedback (une seule : la métrique doit voir ce que le verdict tranche)** :
  juger contre le FIN (pas grossier) ; mesurer expert-vs-DESCEND (pas binaire présupposé —
  le bord réveille le descend) ; en distance PERCEPTUELLE (pas erreur de cache L2). J'avais
  les trois à côté ; non corrigées → faux PASS fabriqué.
- `c2_action_perceptual.py` : par fenêtre énergétique, mémoïser/expert/descend en distance
  perceptuelle (fréq ±10 %, ampl ±20 %) contre la vérité FINE, aux bords (Ur=4.5 et 8.5).
- **Résultat : memoize 100 %, expert 0 %, descend 0 % aux DEUX bords.** Perceptuel non-fermable
  = **0 %**, alors que cache non-fermable = 18.6 %/17.3 %. **DIVERGENCE TOTALE cache↔perceptuel** :
  les erreurs de phase non-fermables au cache sont SOUS le JND → invisibles. mémoïser-partout
  est perceptuellement optimal partout.
- **VERDICT (prouvé, pas supposé)** : à Re=100 2D, sur tout le balayage lock-in, **C2 est
  perceptuellement VACANT**. Le substrat (choisi pour la tractabilité de G0) est, par sa
  (quasi-)périodicité perceptuelle, incapable de tester la thèse de routage au sens qui compte.
  **Option 3, prouvée en perceptuel.** Cause racine unique = périodicité (a affaibli C1, vidé
  C2, et rend le bord sous-JND). Router ne paie que sur dynamique perceptuellement IRRÉDUCTIBLE
  (Re plus haut / chaotique), hors ce Re.
- **Ce que les 3 corrections ont sauvé** : juger en L2-cache aurait « trouvé » 18.6 % de signal
  et fabriqué un C2 PASS contaminé (router aide sur des fenêtres que personne ne perçoit). La
  mesure cheap perceptuelle a tué le fantôme AVANT la chirurgie. Discipline « jamais L2 » vindiquée.

---

## 🏁 T1 — CONCLUSION (2026-06-28)

**Le verdict, à la portée EXACTE (un vrai chiffre, pas étendu au-delà de ce qu'il couvre) :**

> ⚠️ **Portée (correction critique) : T1 a exercé 2 sorties sur 3.** La décision routeur s'est
> effondrée en UNE sortie : **mémoïser 100 %, expert 0 %, descend 0 %**. Donc T1 a localisé la
> condition de validité de la sortie **CACHE/EXPERT** (axe : mémoïsabilité perceptuelle) — PAS
> "du routage". La sortie **DESCEND — cœur de la thèse architecturale** (closability, frontière
> (2)-vs-(3), « raffiner ici ») — n'a produit **AUCUNE donnée** : le substrat (tout-représentable,
> C1 : tout bas-rang) ne l'a **jamais sollicitée**. Zéro activation ≠ « descend ne paie pas » =
> « descend non interrogé ». **Verdict réel et honnête, mais sur une FRACTION de l'archi.**
>
> Sur l'axe testé : **router (cache→expert) ne paie que si le contenu perceptuellement pertinent
> est non-mémoïsable clé-halo = non-récurrence perceptuelle > JND.** Propriété de l'archi, mesurée
> proprement (cache 18.6 % → perceptuel 0 % à Re=100). À Re=100 2D (quasi-périodique), non remplie
> → C2 perceptuellement vacant SUR CET AXE → pas de verdict fabriqué (§13).

**Acquis solides de T1 :**
1. **Instrument FSI validé et réutilisable** (G0 a/b/c) — oracle couplé différentiable, modèle
   d'erreur séparé (confinement fréquence vs résolution force, force amortie dans le couplé).
2. **Pipeline complet et testé** — IB-Peskin (27 tests), Harten (9), régimes physiques
   anti-circulaires (6), POD+DMD (4), harnais perceptuel.
3. **Acquis théorique transférable** : closability-cache ≠ closability-perceptuelle ;
   c'est la seconde qui gouverne. **Mesurée, pas supposée** (divergence 18.6 % → 0 %).
4. **La méthode a fonctionné quand elle devait** : pré-enregistration + « jamais L2 » +
   « le moins cher qui peut échouer » + « la métrique voit ce que le verdict tranche » ont,
   ensemble, transformé un faux PASS à 18.6 % en un vrai « pas ici ». La machine a refusé de mentir.

**Hors scope T1, reporté à T1.5 (décision NEUVE, séparée) :** config poteau (G0'), Phase 2
différentiable, C3/C4 — tous en aval de C2, donc moot à ce substrat (même vacance perceptuelle).

**La question T1.5 (reposée DURE) :** PAS « substrat produisant de la non-récurrence perceptuelle »
(ça ne réveille que l'EXPERT) — mais **« quel substrat SOLLICITE LES TROIS SORTIES »** :
contient du mémoïsable + du **représentable-non-mémoïsable** (réveille l'expert) + du
**NON-représentable au grossier** (réveille le DESCEND). Le descend ne s'active que sur du contenu
qu'aucun ROM bas-rang ne tient à la résolution grossière → dynamique localement à haute dimension.
- **Piège à éviter** : le forçage apériodique à Re=100 casse la récurrence (réveille l'expert) mais
  reste bas-rang (C1) → **descend toujours dormant** → on re-teste 2/3 en croyant tester le tout.
- **Conséquence** : le chaos (Re élevé) est **suffisant pour l'expert** et **peut-être NÉCESSAIRE
  pour le descend** — c'est le régime où la dimension locale dépasse la capacité d'un expert
  grossier. T1 ne pouvait pas poser cette question (substrat tout-bas-rang) ; T1.5 doit la poser.
- **Méthode** : utiliser l'instrument de fermabilité pour vérifier, sur tout substrat candidat,
  que les TROIS sorties s'activent (taux memoize/expert/descend non dégénéré) AVANT de juger C2.

**Critère TEMPS-RÉEL (C3 redéfini — l'objectif est le jeu, pas un ratio abstrait) :**
- C3 ÉTAIT le claim temps-réel ; moot en T1 pour la même raison que C2 (composite vacant →
  « compute du routé » = lire un cache = trivial). T1 ne pouvait pas le produire.
- **Succès = franchissement de budget-frame (SEUIL), pas ratio de FLOPs** : le routage gagne son
  existence quand le fin-partout VIOLE le budget-frame et que le routé y RENTRE, à plausibilité
  égale (sous JND). Si le fin-partout tient déjà → routage plausible mais SANS ENJEU.
- **L'overhead du routeur compte dans le coût du routé** (extraction features + réseau + décision
  par fenêtre). Sinon mémoïser-partout (zéro décision) bat le routage en TEMPS → vacance par une
  autre porte. Le routeur doit être radicalement moins cher que la décision qu'il évite.
- **Baseline T1 mesurée (2026-06-28)** : fin-partout couplé 512×192 = **1.33 ms/pas, 111 MB,
  tient 60 fps confortablement** (13 pas/frame, lâcher à 0.56 Hz naturel). → **AUCUN enjeu
  temps-réel sur ce substrat** : 3e angle confirmant la vacance T1.
- **L'enjeu temps-réel est un axe d'ÉCHELLE, pas (que) de Re** : le fin-partout dépasse le budget
  quand le domaine est grand (échelle-jeu, 3D), à tout Re. Chaos → enjeu *plausibilité* ;
  échelle → enjeu *temps-réel* ; partiellement orthogonaux (les deux exigent la non-mémoïsabilité).
- **Mémoire : travail (bornée, T1.5, mesurable — cache mémoïsation, bases POD, état actif) ⊥
  persistante (non bornée, T2 = le mur fracture/persistance).** Ne pas confondre.

**Substrat T1.5 valide = (a) perceptuellement non-récurrent + (b) assez grand pour que le
fin-partout dépasse le budget-frame + (c) localement non-fermable au-delà d'un expert HONNÊTE
(capacité raisonnable, pas sous-dimensionné — sinon faux réveil du descend, C1 retourné) +
(d) spatialement SPARSE — domaine majoritairement inerte/settled avec front mince actif.**

**(d) SPARSITÉ — le 4e angle de vacance, le plus structurel (mesuré 2026-06-28) :**
- L'économie = **facteur S = 1/(fraction PERCEPTUELLEMENT IRRÉDUCTIBLE f_p)** — ne tourner fin
  que là où mémoïser/grossir échoue *perceptuellement* (PAS où c'est dynamiquement actif).
  Le routage ne franchit le budget que si **S > G** (G = facteur de dépassement du fin-partout).
  Substrat dense → S petit → fenêtre d'enjeu `fin>budget≥routé` quasi-vide, à TOUTE échelle.
- **⚠️ Correction de mesure (feedback)** : le seuil « std de vitesse » mesure l'activité DYNAMIQUE,
  qui SUR-COMPTE (comme la L2-cache) — ex. la relaxation derrière un front est dynamiquement active
  mais perceptuellement éphémère/mémoïsable. La vraie sparsité est **perceptuelle** (f_p).
- **Substrat VIV** : 42 % d'activité DYNAMIQUE (seuil 10 %) = **type de scène dense** (observation de
  type valide : un sillage entretenu n'a pas de bulk qui s'éteint). MAIS le verdict opératoire est
  perceptuel : **f_p ≈ 0 %** (C2, périodique → mémoïser-partout) → vacant. Les 42 % dynamiques ne
  condamneraient que dans le contrefactuel non-récurrent (S~2.4×) ; ici C2 condamne d'abord.
- **Scènes-jeu (inondation ~10 %, feu ~5 %) = HYPOTHÈSE NON MESURÉE, à TESTER — pas une
  démonstration.** Intuition dynamique posée à côté d'un fait mesuré (le 42 % du wake) ≠ preuve.
  Le f_p réel peut diverger DANS LES DEUX SENS : la zone de relaxation derrière le front (largeur ∝
  temps de relaxation NON mesuré) peut densifier ; l'écart dynamique↔perceptuel (relaxation
  mémoïsable) peut sparsifier. Inconnu jusqu'à mesure.
- **Filtre de faisabilité T1.5 (pendant temps-réel du balayage de fermabilité)** : sur toute scène
  candidate, **mesurer f_p (perceptuellement irréductible), PAS la fraction dynamique, AVANT tout
  oracle** — la mesure perceptuelle qui a tué le fantôme, appliquée à la sparsité. f_p > ~10-20 %
  (S < 5-10×) → écarter. **Mesurer f_p, ne pas présupposer.**
- **Note de portée** : la non-fermabilité qui réveille le descend = **non-localité d'information**
  (halo local insuffisant), dont la haute dimension chaotique n'est qu'un cas particulier suffisant
  — chercher d'abord le bas-rang non-local (advection d'amont). Enjeu temps-réel = **sparsité ×
  échelle**, pas échelle seule. Mémoire : même gouvernance (le routage borne la mémoire de travail
  à l'actif → même chiffre, f_p (perceptuel, pas dynamique), décide temps ET mémoire de travail).

**(e) LE JND — pinner le perceptuel, sortir de l'attracteur (feedback, le vrai générateur) :**
- L'attracteur « proxy sur-compte → perceptuel corrige » a frappé 3× (L2-cache @C2 ; vitesse
  dynamique @sparsité). Le GÉNÉRATEUR n'est pas un réflexe à surveiller — c'est que le perceptuel
  n'est pas calculé, il est DÉFINI, et mon JND est un placeholder (±10 %/±20 %, « ne pas défendre »).
  Tant qu'il est placeholder, « perceptuellement irréductible » n'a pas de référent fixe → chaque
  f_p hérite de l'arbitraire du seuil. **La sortie de l'attracteur = pinner le JND, pas surveiller le proxy.**
- Le wake passait car ses chiffres étaient aux EXTRÊMES (0 %/42 %, robustes à tout JND). L'inondation
  sera en ZONE GRISE → le verdict dépendra du JND non pinné → l'attracteur frappe une 4e fois.
- **Geste T1.5 (gravé) : f_p AVEC analyse de sensibilité au JND** — f_p en fonction du seuil sur sa
  plage plausible. C'est l'analyse de sensibilité que je fais pour β/D/seeds, JAMAIS appliquée au
  seul paramètre dont dépend TOUTE la métrique perceptuelle. f_p petit sur toute la plage → enjeu
  robuste, go. f_p traverse la fenêtre de décision quand le JND varie → verdict INDÉTERMINÉ →
  **pilote JND BLOQUANT, pas optionnel.** Ne pas croire f_p avant d'avoir borné sa sensibilité.
- **En réserve (ne pas construire encore)** : le JND-jeu n'est pas constant — distance d'observation,
  caméra, attention. f_p est donc un CHAMP fovéa-dépendant, et la sparsité perceptuelle EXPLOITABLE
  > f_p à JND uniforme (la majorité du domaine est loin → JND élevé → mémoïsable même si active).
  C'est la fovéa observer-centrée / le LOD, revenant comme MULTIPLICATEUR de sparsité — l'axe qui
  pourrait faire passer l'inondation de « limite » à « enjeu réel », mais seulement si mesuré.

**(f) STRUCTURE — routage ⊗ persistance (résultat de ROADMAP, feedback) :**
- **C2 (plausibilité) et C3 (économie) ne sont PAS séparables.** Sur le wake, le routé reproduisait
  le monde (mémoïser-partout) PARCE QU'il n'y avait rien à router — même cause que C3 vide. « Router
  améliore la plausibilité » et « router économise » = deux LECTURES d'un seul événement (une fenêtre
  où mémoïser échoue et la bonne action réussit).
- Le routage a une décision graduée SEULEMENT sur un domaine **MIXTE** (mémoïsable + non-mémoïsable
  coexistant). Le wake n'avait pas de mélange (tout actif → décision triviale, effondrée). **Le
  mélange vient de la PERSISTANCE** : le settled est mémoïsable, le front non. Sans persistance, pas
  de mélange, pas de décision — pour C2 COMME C3. (Option « C2 seul sans persistance » = piège
  renommé : actif-partout = wake à l'envers.)
- **Donc le routage ENTIER dépend de la persistance.** INVERSION DE DÉPENDANCE : persistance =
  PRÉREQUIS du routage, pas sa suite. La pré-enregistration T1(routage)→T2(persistance) avait l'ordre
  inversé pour la VALEUR du routage. Acter l'inversion ; pas besoin de re-dériver toute la roadmap.
- **Test réel de l'architecture = sur un domaine mixte : le routeur appris distingue-t-il le committé
  NON-TRIVIAL de l'actif mieux que le nul, et ce gain se lit-il EN PLAUSIBILITÉ ET EN COMPUTE ?**
  Un seul test, deux lectures inséparables — le premier vrai test, sur le terrain où l'archi vit.
- **Garde anti-circularité (récursive)** : le committé ne doit PAS être trivialement lisible depuis le
  détail (sinon le nul-détail le détecte = circularité, C2 vacant). Le committé doit porter une
  STRUCTURE fine (relief figé = HAUT détail) tout en étant INERTE (activité nulle) → le détail seul ne
  distingue pas committé-figé de front-actif ; le halo (vitesse≈0) le peut. Divergence physique, pas construite.

**Ne pas surclamer dans aucun sens** : ni « le routage est mort » (faux), ni « il suffit de monter
le Re » (non mesuré), ni « T1 a testé l'architecture » (faux : 2 sorties sur 3, descend non interrogé,
et — résultat (f) — domaine non mixte donc routage jamais sollicité du tout).

---

## 🔬 T1.5 — Journal (substrat sollicitant les trois sorties)

### 2026-06-30 — Substrat mixte par persistance : mécanisme front-passage → BARRAGE (obstacle traversant-spécifique) ; cap (D) inondation

Construction d'un substrat MIXTE (mémoïsable settled + non-mémoïsable actif ; le mélange vient de la
persistance, cf. (f)) dans le rig LBM (`experiments/mixed_substrate.py`), gaté 2 étages (GATE 1
stationnarité de W ; GATE 2 monde-3 sous binning + gel-coupé).

- **Settling par physique RÉFUTÉ** (drag uniforme, 3 points jusqu'à 25× baseline) : committé=0 à tous,
  débit mouillé collé à U, frac_active MONTE avec le drag (0.96→0.98). Conservation de masse ⇒ débit
  ≈ U partout en canal traversant ; un puits de qdm uniforme ne settle rien (construit densité/pression
  pour maintenir le débit). Settling-par-vitesse impossible ici.
- **Mécanisme (1) — commit par PASSAGE DU FRONT** (découplé de la vitesse : cellule balayée vieillit
  vers committé au rythme 1/τ) : GATE 1 a d'abord dit STATIONNAIRE (committé 22 %, W 6.6 %, cv 0.01)
  — **artefact**. Profil spatial : committé = l'INLET (x<0.28), figé à v≈0, front calé, **domaine MORT**
  (actif=0.004 partout). Le commit fige d'abord l'inlet (continûment actif) → région committée v≈0 =
  OBSTRUCTION → en débit conservé elle fait BARRAGE → source étouffée, front calé, domaine mort.
- **Tenaille du freeze (dans cette géométrie)** : figer à **v≈0** → halo-distinguable (garde
  anti-circularité OK) MAIS barrage ; figer à **v≈U** → pas de barrage MAIS halo-invisible (garde
  violée). Pas de 3e voie dans un canal traversant.
- **Classifieur GATE 1 durci** (acquis net, réutilisable) : le binaire `W<0.02 ⇒ effondrement` mentait
  (confondait rien-commit / tout-commit, tous deux W≈0) ET laissait passer le domaine-mort (W
  stationnaire, zéro dynamique). Désormais 4 modes nommés depuis (committé, actif, W) : domaine-mort /
  rien-commit / tout-commit / W-instable.

**Énoncé ÉTROIT (le seul qui se transfère)** : *dans une géométrie mono-champ, incompressible, à flux
traversant conservé, une région inerte committée est nécessairement une obstruction.* Load-bearing =
**mono-champ / incompressible / traversant**. **PAS** « la persistance est structurellement dure »
(deux échecs ≠ théorème). Le barrage est une PRESCRIPTION : le canal traversant était le mauvais vase.

**Deux échappatoires, dans les scènes que Cascade vise, aucune dans le LBM-traversant** :
(α) **surface libre / accumulation** (inondation) — le settled ne barre pas, le volume s'empile (niveau
monte), pas de conflit de flux par section ; v≈0 physique, pas fiat.
(β) **champ persistant séparé** (sédiment/charbon) — l'inerte n'est pas le médium qui s'écoule, 2
quantités conservées distinctes. (Mais à Re=100 le dépôt est esclave du flux périodique → mémoïsable →
résout le barrage, PAS la vacance ⇒ confirme que la **non-récurrence** est la contrainte liante.)

**Deux contraintes liantes pour les 3 sorties** : (i) anti-barrage, (ii) anti-vacance-périodique (tueur
de C2 à Re=100). Aucune option mono-contrainte ne suffit. **Cap T1.5 = (D) inondation shallow-water sur
terrain** (surface libre + front transitoire non-récurrent = (i) ET (ii)), via la brique validée
shallow-water-rom (MUSCL wet/dry = le re-test SHARP dû depuis le caveat de diffusivité d'oracle).
Mapping 3 sorties : memoize = flaques settled (v≈0, h tracke le terrain = haut détail) ; expert = front
wet/dry non-récurrent ; descend = front × terrain fin non-fermable au grossier. **Conditionné** au filtre
de faisabilité (f_p perceptuel + sensibilité JND + 3 taux non-dégénérés) AVANT tout router, et au
re-sharpening MUSCL (sinon front sur-diffusé « paraît bas-rang » → re-tue le descend en silence). 3/3
non garanti ; 2/3 (memoize/expert) reste une avancée stricte sur la décision effondrée du wake.

### 2026-06-30 — Recadrage par le BUT (vérifier si Cascade est une solution) + contrat pré-enregistré de la règle de dépôt

**Recadrage de posture (le but force la falsification, pas la construction)** : l'objectif n'est pas
« tester le router » — c'est **vérifier si Cascade** (un état-monde `z`, un opérateur routé `F`, image+son
readouts, plausible + PERSISTANT + horizon infini au budget-jeu) **est une solution**. Vérifier = falsifier
(le test le moins cher qui peut tuer), pas ingénier vers le succès. Le livrable n'est pas « le router
marche » (binaire) mais une **carte de viabilité** : où le routage paie (sparse + non-récurrent + sous
pression de budget + non-local) vs vacant. Les vacances (C2 wake, barrage) sont des points de carte, pas des
échecs à contourner. Trois axes de tueur : budget (router, payoff prouvé nulle part), **persistance**
(le mur le plus profond, à peine touché), plausibilité (**JND non pinné** → aucun axe perceptuel mesurable).

**Métrique = plausibilité, JAMAIS exactitude.** Le détour gate-0 (front 2nd-ordre ? convergence ?) était la
mauvaise métrique : exactitude-de-solveur, la dérive que l'arc tue. « 1er-ordre au front » n'est disqualifiant
QUE si le trait perdu était perceptuellement présent. L'instrument doit être perceptuellement adéquat, pas exact.

**Mesure (1) = test-tueur de l'axe PERSISTANCE** (la scène doit être PERSISTANTE, pas stationnaire — sinon
on photographie un ressaut a-historique = VIV redux, et `z`-qui-grossit-avec-l'histoire n'est jamais sollicité,
le piège T1 par la porte « scène propre »). Trois issues : (1) trivial-fermable [vacant, wake déguisé] /
(2) structuré-mais-irréductible [mur mémoire, persistance morte] / (3) structuré-ET-non-trivialement-fermable
[le mélange existe]. **(2) et (3) ne sont en jeu QUE si la règle de dépôt est path-dependent** ; une règle
instantanée-locale ne laisse que (1), maquillée.

**Contrat pré-enregistré de la règle de dépôt (gravé AVANT le code — c'est lui qui empêche (1) de se mentir) :**
- **Défaut latent** : `commit(x)=g(h(x),terrain(x))` est terrain-pilotée/haut-détail/v≈0/halo-distinguable
  (passe l'anti-circularité détail) MAIS fermable-halo trivialement (info = terrain connu + profondeur dans
  le halo). Issue 1 déguisée en issue 3 ; verdict « fermable » juste, conclusion fausse.
- **4e garde (manquante)** : halo-distinguable, pas détail-trivial, ET **pas halo-trivial** (= pas
  reconstructible depuis terrain-statique-connu + profondeur-locale SANS l'histoire).
- **Règle** : champ **SÉDIMENT séparé** `s(x)` (échappatoire β, pas gel-de-`h` α qui re-barre + reste
  instantané), loi **PATH-DEPENDENT** (type Exner : dépôt/reprise par cisaillement), modifie `b_eff` (front
  suivant ré-ancre sur terrain+histoire). `s` à v≈0 → ne barre pas (lac-au-repos couvre déjà le settled).
- **Critère pré-enregistré (falsifiable en 2 runs)** : deux séquences de pulses produisant le **même état
  instantané (h,u) en x** doivent produire des dépôts `s(x)` **différents**. Même dépôt → règle
  instantanée-locale → (1) vacant par circularité (axe halo, pas détail).
- **Hypothèse architecturale load-bearing** : la fermeture a accès LIBRE au terrain fin STATIQUE (asset
  autorisé, stocké, jamais régénéré) → structure terrain-dérivée « gratuite », seule l'histoire compte.
  Tient pour terrain statique ; basculerait pour terrain dynamique/destructible.
- **Critère ≡ mesure** : fermabilité de (1) = cache **clé-(halo dynamique instantané + terrain fin connu)**
  qui échoue. La clé DOIT inclure le terrain fin (sinon sur-compte la structure-terrain en non-fermabilité) et
  être INSTANTANÉE (sinon un cache-de-trajectoire ferme l'histoire et fake la fermabilité). Les 2 erreurs de
  clé = les 2 façons de mentir.
- **Goldilocks bloquant** (comme le pilote-JND) : la loi a un taux dépôt/reprise. Mal réglé → tout se dépose
  (empreinte du 1er pulse = quasi-instantané, issue 1) ou rien ne tient (domaine-mort). Mesurer
  `f_fermable(taux)` avec sensibilité, pas un seul réglage, avant de croire (1).

→ Build : scène pulses-sédiment path-dependent + mesure de fermabilité clé-(halo+terrain) du `s` accumulé ;
BC soutenues (pour le ressaut-ancré = trait descend, mesure 2) **différées derrière (1)**.

### 2026-06-30 — (1) premier chiffre : l'histoire S'ÉVAPORE au readout (C2-bis prouvé) + critère de richesse RE-spécifié en readout

**Critère path-dependent (verrou 1) : PASSE**, robustement. `corr(s_A,s_B)=0.39` sans binning (ordre de pulses
inversé → 61 % du dépôt vient de l'histoire), corroboré par resfrac 67 % (binning-sensible, habillage). La règle
intègre l'histoire. La path-dependence est une propriété de la RÈGLE → espace-`s` est le bon espace pour CE verrou.

**Premier chiffre de (1) — l'histoire mesurée en READOUT (pas distance-`s`)** : à taux faible (`s`~2 % du relief),
`corr(R_A,R_B)=1.0000`, **0 % supra-JND à JND≥5 %** sous ombrage RASANT (révèle le relief — garde anti-platitude
satisfaite). **61 % d'histoire en espace-`s` → ~0 % en readout.** C2-bis prouvé sur la PERSISTANCE : le 61 % est
le 18,6 % cache-L2 de C2, →~0 % perceptuel. Cause : un dépôt à 2 % noyé par le terrain à 98 % en pixels — la
non-linéarité du readout (thèse LOD) décide, pas la magnitude de `s`. Mesurer la fermabilité en distance-`s`
aurait fabriqué 61 % de non-fermabilité fantôme. **Verdict réel mais conditionnel au TAUX** (Goldilocks).

**Correction du proxy de RICHESSE (même trou, déplacé d'un espace)** : « `s[P1..Pn]` vs `s[P1]` » vit en espace-`s`
→ des changements de `s` s'évaporent au readout (qu'on vient de prouver). Richesse-en-`s` nécessaire, PAS suffisante ;
elle doit monter en readout, comme la fermabilité. Une fois en readout, richesse (b) et survie (a) FUSIONNENT en un
seul objet `ΔR`. `corr(s_A,s_B)` abandonnée (proxy d'avant-readout, aveugle à la saturation).

**DÉFINITION PRÉ-ENREGISTRÉE de `f` (critère de richesse, falsifiable en 3 runs : 2 ordres + simultané)** :
  `f(taux) = fraction du domaine où ‖readout(s[ordre]) − readout(s[simultané même-masse])‖ > JND`,
  readout = relief ombré incidence rasante, avec sensibilité-JND sur la plage plausible.
- Évaporation (taux faible) → `ΔR` sub-JND partout → `f≈0`.
- Saturation (taux fort) OU dépôt terrain-déterminé (ordre n'importe pas) → `readout(ordre)≈readout(simultané)` →
  `f≈0`. **Le terrain-déterminé s'annule dans `s_ordre−s_simultané` → `ΔR` isole l'HISTOIRE, pas le terrain :
  anti-circularité (« pas halo-trivial ») INTÉGRÉE à la définition, pas un test séparé.**
- Seul un dépôt dont l'ORDRE laisse une trace perceptible → `f>JND` = issue 2/3 réelle.
- Référence = simultané-même-masse (order-free), PAS « vs premier pulse » (privilégierait P1). C'est le critère
  original « même-état → dépôts différents » remonté en readout, débarrassé du choix de référence.
- UNE courbe `f(taux)` + sensibilité-JND : maximum franc = intégration visible ; rampe sub-JND sur toute la plage =
  sub-JND/trivial partout. Robustesse lue sur une courbe, pas une intersection (a)∩(b) bruitée (qui fabriquerait un
  Goldilocks en zone grise = attractor #4, JND non-pinné).

**Séquencement (cheapest-falsifiable)** : `f(taux)` D'ABORD (3 runs, pas de cache). `f≈0` partout → (D) vacant sur
l'axe persistance, point de carte, verdict réel SANS clé de cache. `f>JND` robuste à un taux → histoire
perceptiblement persistante → ALORS seulement la fermabilité clé-(halo+terrain) en distance-readout sépare issue 2
(mur mémoire) de issue 3 (fermable). La clé de cache est différée **derrière `f`**.

### 2026-06-30 — VERDICT persistance-sédiment + ÉNONCÉ DE CARTE : le routage paie seulement sur du perceptible HORS-CLÉ

**Balayage `f(taux)` (def pré-enregistrée, `f(A−B)` sans confond schedule)** : `s/relief` = 2/6/20/45 % →
`f(A−B)` = 0.7/3.7/7.0/11.4 % (JND 2 %), 0/1.2/4.4/7.7 % (JND 5 %). Monotone montant, **aucune saturation**.

**Verdict NET (pas gris)** : persistance-par-sédiment **sub-JND sur tout le régime plausible** (≤~20 % relief :
`f` petit pour TOUT JND raisonnable → JND-ROBUSTE), perceptible seulement au dépôt **lourd** (45 % relief = le
sédiment enterre le terrain = **re-terraforming**, plus de la persistance-par-dépôt). La dépendance-JND (×2) est
**entièrement du côté non-plausible** de la frontière de plausibilité du dépôt. **Point de carte robuste, pas indécis.**

**PAS de pin-JND ici (piège, pas détour)** : pinner le référent le plus structurant de l'arc contre une mesure
auto-bordée (1 substrat/règle/géométrie/readout/angle, taux plafonné) = tirer un invariant d'un point unique. Et
treadmill pur : le pin ne déplacerait le verdict que dans le coin déjà marqué hors-plausible ; là où le verdict vit
il est déjà JND-robuste. Le JND attend un readout digne de le porter.

**DEUX POINTS DE CARTE, CAUSE COMMUNE** :
- Wake (budget) : vacant car PÉRIODIQUE → tout récurrent → le halo (clé) tient la récurrence → mémoïsable.
- Sédiment (persistance) : sub-JND car le dépôt visible est dominé par le TERRAIN STATIQUE (dans la clé) →
  l'histoire visible est noyée par de la structure que la clé contient déjà gratuitement.
- **Commun : le contenu perceptiblement pertinent est porté par quelque chose que la clé (halo+terrain) tient DÉJÀ.**
  Le routage ne paie que sur du perceptible **HORS-CLÉ** ; les deux substrats mettent presque tout le perceptible
  DANS la clé. **Thèse readout-LOD retournée** : la non-linéarité qui jette le détail lointain (cadeau compression)
  est la même qui noie l'histoire sous le terrain (routeur vacant).

**Raffinement — ISSUE 1, PAS issue 2 (le mur reste non testé)** : les deux vacances sont des ISSUE 1 (in-clé,
trivialement fermable), PAS le mur-mémoire. Le sédiment sub-JND est BONNE nouvelle pour l'engine : le LOD jette
l'histoire invisible → mémoire bornée → horizon infini cheap, SANS routeur. Séparation acquise : **engine-viable**
(in-clé = cheap, pas de mur) ≠ **routeur-paie** (exige hors-clé). Le mur-mémoire (tueur le plus profond) exige du
persistant hors-clé, perceptible ET irréductible → **non testé**.

**ÉNONCÉ DE CARTE (premier de l'arc — condition de viabilité NOMMÉE, pas cherchée à tâtons)** : *le routage est
vacant partout où le perceptible est porté par la clé (périodicité, relief statique) ; reste à tester s'il paie là
où le perceptible est porté par de l'HISTOIRE DYNAMIQUE HORS-CLÉ.* Atteint par ÉVIDENCE (deux vacances même-cause),
pas par « montons le Re » (réconfort). Prochain test = `ΔR(ordre−simultané)` sur un trait **FLUIDE** (pas déposé),
dont le contenu readout-visible n'est PAS réductible à terrain+présent — **simultanément** le test routeur-paie ET
le premier vrai test du mur-mémoire. `f≈0` là aussi → cause commune = **propriété du PARI** (résultat majeur).
`f` franc + JND-robuste → premier lieu où le routage vit → là le JND mérite son pin.
**Caveat de scope** : « hors-clé + persistant » exige un fluide qui NE settle PAS vers un état order-indépendant →
dynamique soutenue / non-récurrente (BC soutenues, régime non-récurrent) = nouveau substrat. Et « non-récurrent mais
bas-rang » pourrait retomber in-clé (bas-rang = halo-fermable) → le vrai hors-clé pourrait exiger du haut-rang.

### 2026-06-30 — Distinction PÉRIODIQUE / BAS-RANG / ORDER-INDÉPENDANT (le wake les fusionnait) — (1) re-scopé

**Correction de prédiction** : « bas-rang → in-clé → f≈0 » n'est vrai que pour le bas-rang **AUTONOME**. Un cycle
limite est bas-rang ET order-indépendant (l'attracteur oublie le transitoire → in-clé) ; le wake fusionnait
périodicité et basse dimension car à Re=100 elles arrivaient ensemble. Elles se séparent : un écoulement **bas-rang
NON-AUTONOME** peut garder dans son état présent une trace de l'ordre du forçage (mode amorti à τ_relax long,
recirculation lente) — order-DÉPENDANT tout en étant bas-rang. L'info distinguant A de B vit dans une **phase lente
que le coarsening jette** (mécanisme du bord-de-bande C2, mais PERSISTANT) → **hors-clé bas-rang**. C'est le trou de
la carte ; « bas-rang retombe in-clé » l'excluait par construction. Le bon découpage = périodique ≠ bas-rang ≠
order-indépendant, pas la fusion que le wake a vendue.

**(1) re-scopé — 3 causes de `f≈0` à séparer (sinon zéro creux)** : (a) order-indépendant (cycle limite, oublie),
(b) order-dépendant mais trace sub-JND (sédiment-bis, LOD noie), (c) géométrie sans mémoire lente assez longue. La
mesure qui les sépare, EN ESPACE-ÉTAT (légitime : propriété de la dynamique, pas du readout) :

  **`corr(état_A(t), état_B(t))` après le dernier pulse divergent — la COURBE sur tout l'horizon :**
  - remonte vers 1 → order-INDÉPENDANT (attracteur) → `f≈0` = « le pari exige la non-récurrence vraie », EARNED
    (pas supposé) → build chaos justifié par évidence.
  - plateau < 1 → order-DÉPENDANT (mémoire d'ordre réelle) → ALORS `ΔR(ordre−sim)` au readout sépare sédiment-bis
    (mémoire noyée → common cause 3e fois : le pari meurt sur le readout-LOD, pas la récurrence) de
    premier-hors-clé-RÉEL (mémoire perceptible + bas-rang → retourne la prédiction, meilleur résultat possible).
  - **COURBE pas point** : plateau<1 (mémoire vraie) vs rampe→1 (mémoire qui s'efface, persistance illusoire à
    horizon plus long). La forme est le verdict, comme la saturation au balayage de taux.

**Conséquence** : on ne build le haut-rang/chaos QUE si `corr` remonte à 1 en bas-Re soutenu. Si la mémoire d'ordre
existe déjà en bas-Re, le pari exige peut-être juste de la **mémoire lente bas-rang** (bien moins chère, instrument
validé) — l'hypothèse que le wake a masquée. Symétrique de « ne pas sauter au chaos par réconfort » : ne pas
présupposer que seul le chaos porte du hors-clé.

### 2026-06-30 — Le BUT coupe les géométries fluides : mémoire-effaçable ≠ persistance ; le test = f(READOUT), pas f(géométrie)

**Coupe (par le but)** : chercher la mémoire d'ordre dans un mode fluide lent (bord-de-bande, recirculation) est une
dérive — dispositifs de LABO, hors-jeu, choisis pour réutiliser l'instrument cheap (mode-ingénieur par la porte
« réutilise l'instrument »). Et physiquement disqualifié : un mode lent bas-Re est **AMORTI** (τ_relax fini) →
`corr(t)` **RAMPE→1**, ne plateau pas (dissipatif autonome → relaxe vers l'attracteur). Mémoire effaçable = transitoire
lent, PAS persistance à horizon infini. `plateau<1` strict exige forçage-jamais-coupé (mémoire dans le forçage =
in-clé via pulses-présents) OU multistabilité/hystérésis (état IRRÉVERSIBLE). Donc le mode fluide lent = **4e forme
d'in-clé** (esclave du forçage connu).

**DISTINCTION GRAVÉE** : mémoire-EFFAÇABLE (mode fluide amorti, hors-jeu, `corr`→1) ≠ mémoire-NON-EFFAÇABLE
mal-rendue (sédiment + mauvais canal). La persistance à horizon infini exige un changement IRRÉVERSIBLE
(commit/seuil/bascule), pas une phase lente. Le wake a vendu la fusion périodique/bas-rang ; ne pas laisser le
sillage vendre la fusion mémoire-lente/persistance.

**RE-DÉSIGNATION** : j'ai DÉJÀ la mémoire non-effaçable (sédiment, `corr(s_A,s_B)=0.39`, le dépôt ne relaxe pas = être
committé). Le trou n'est pas « un mode fluide hors-clé » mais **« un READOUT où cette mémoire déjà-non-effaçable
devient perceptible sans enterrer le terrain »**. Le sédiment a échoué par le CANAL (relief ombré = terrain+dépôt
même canal optique), pas par manque de mémoire (61 %). Un seul readout testé, le pire.

**Le but comme OUTIL** (« image ET SON sont des readouts ») : canaux non testés où `s` module ce que le terrain ne
porte PAS — albedo/matériau du dépôt, turbidité de colonne d'eau, signature sonore lit-fluide. Non-linéaires en `s`,
DÉCOUPLÉS du terrain = la classe que la thèse readout-LOD prédit comme le bon levier.

**TEST re-désigné (cheaper que (1), plus proche du but que (2))** : garder le sédiment path-dependent + `ΔR(ordre−sim)`,
balayer le **CANAL de readout**, pas la géométrie. `f(readout)` : relief (fait, ≈0 plausible) / albedo / turbidité / son.
- Un canal fait survivre les 61 % >JND → **premier hors-clé, in-game, mémoire vraie, horizon infini**.
- Aucun canal → **common cause 3e fois, STRUCTURELLE** : le pari meurt sur le readout-LOD (pas récurrence ni terrain) —
  le résultat le plus profond de l'arc, touchant directement le `z→readouts` au cœur de la phrase.
- Implémentation : turbidité-propre exige un champ sédiment SUSPENDU (pas tenu) → commencer par l'**albedo** (`s`→
  couleur saturante, découplé du relief), buildable depuis le `s` existant. Son APRÈS (JND auditif non pinné),
  seulement si l'albedo montre que le canal est le levier.

### 2026-07-01 — PREMIER HORS-CLÉ candidat : la mémoire d'ordre SURVIT dans le canal albedo (le canal était le levier)

`f(readout)` sur le sédiment existant (mémoire d'ordre non-effaçable, `corr(s_A,s_B)=0.39`), 2 canaux, taux plausibles :
- **relief ombré** (terrain-couplé) : ≈0 (noyé par le terrain) — déjà acté, le C2-bis.
- **albedo** (`1−exp(−s/S_HALF)`, terrain-découplé) : `f` = 5–10 % aux mêmes taux, JND-robuste.
- **Sensibilité S_HALF (0.005→0.20, ×40)** : `f_albedo` reste **3.5–10 % ≫ f_relief (1.2 %)** sur TOUTE la plage
  (ne s'effondre à aucun bout) → survie **PAS un réglage**, robuste au JND ET au paramètre matériel S_HALF.

**REFRAME** : le « hors-clé » n'était pas une propriété de la DYNAMIQUE (mode fluide) mais du **READOUT (canal)**. La
mémoire était dans le sédiment committé depuis le début ; le relief la jetait (compétition optique avec le terrain),
l'albedo la garde (découplé). Le levier du routage est le **CANAL**, pas seulement le substrat — ce que `z→readouts`
prédisait, manqué par mon unique readout (Lambert) par construction.

**ANTI-SUR-REVENDICATION** : « perceptible-hors-clé EXISTE » ≠ « le routage PAIE ». Reste (a) **fermabilité
clé-(halo+terrain) DANS le canal albedo** : mémoire reconstructible du halo (issue 3, paie + persistance fermable)
ou pas (issue 2, MUR mémoire) — path-dependence l'implique non-fermable, à MESURER, et c'est le **premier vrai test
du mur-mémoire** (tueur le plus profond) sur un objet perceptible réel ; (b) **pression de budget** (fin-partout vs
frame, axe orthogonal intouché). Caveats : 1 terrain/règle/géométrie/canal, mais le relatif (albedo≫relief, robuste
JND+S_HALF) est intra-scène.

**POINT DE CARTE — premier POSITIF de l'arc** : il existe du perceptible-persistant-hors-clé (mémoire d'ordre du
sédiment, canal albedo) → **premier endroit où le routeur pourrait avoir un job**. Prochain test = fermabilité
clé-(halo+terrain) en distance-readout-albedo : SÉPARE issue 2 (mur) de issue 3 (paie). Gate `f>JND` robuste enfin
tenu → la clé de cache, différée jusqu'ici, a enfin un objet qui mérite d'être mesuré.

### 2026-07-01 — Test du MUR-MÉMOIRE corrigé : `f(k)` profondeur de mémoire perceptible (PAS complexité-de-champ)

**Trou de la fermabilité-instantanée** : la mémoire d'ordre étant non-fonction de (état+terrain) par le critère
path-dependent DÉJÀ établi, la fermabilité-clé-instantanée retournerait « non-fermable » TRIVIALEMENT — re-confirme la
path-dependence, ne teste pas le mur. **Trou de « complexité albedo sature-t-elle »** (ma 1re reformulation) : un champ
borné (albedo plafonne) sur grille finie a complexité bornée QUELLE QUE SOIT l'histoire → « sature » = borne-de-grille,
pas closure. Faux-NON-mur. La complexité-de-champ sature trivialement.

Le mur = « l'information mutuelle passé↔readout-FUTUR croît-elle » : combien du passé retenir pour que le readout
présent soit correct. Closure = résumé borné `c`, readout reconstructible de `c`, `c` ne croît pas avec l'histoire.

**Métrique : `f(k)` = profondeur de mémoire perceptible** = `ΔR-albedo(readout présent)` entre deux histoires
IDENTIQUES sur les `k` derniers pulses, DIVERGENTES avant (même multiset, ordre du préfixe inversé).
- `f(k)→0` (k croît) : divergence ancienne oubliée → CLOSURE (issue 3, horizon infini viable, thermalisation-borne-le-
  buffer mesurée).  - `f(k)>JND` pour k grand : ancienne divergence encore perceptible → MUR (issue 2).
- Mesure une DIFFÉRENCE → **ne sature PAS par grille** (deux champs saturés peuvent l'être différemment). Garde
  anti-faux-mur INTÉGRÉE : étalement-sans-rétention → `f(k)→0` auto (l'étalement EST une closure). Réutilise ΔR-albedo
  validé, zéro instrument neuf.

**Trois gardes gravées** : (1) lire la COURBE `f(k)` — décroît-vers-0 / plateau-sous-JND (closure) / plateau-sur-JND
(mur) ; JND-albedo a enfin son référent (gagné à l'étape S_HALF-robuste). (2) Taux **fort-plausible** (20 % relief) —
le mur vit où la mémoire est tenace ; même point d'op que la présence perceptuelle. (3) Horizon infini : le verdict est
une **PENTE EXTRAPOLÉE** (`f(k)` sommable → closure / divergente → mur), PAS une valeur à k fini (sinon on rejoue « ça
sature » sur horizon trop court = le piège horizon-length de la conclusion T1). La pente est mesurable sur k fini.
Cheapest-first : `f(k)` à 2-3 k, voir si la courbe descend, avant le long horizon.

### 2026-07-01 — `f(k)` mesuré : PAS de mur-mémoire (closure-leaning, ROBUSTE à la géométrie)

`f(k)` cheapest-first (T=10, k=1/4/7/10, taux fort-plausible, JND-albedo), descend à JND 5 % :
- géométrie OVERLAP (pulses même bande, recouvrement = facile pour closure) : 8.5 → 1.9 → 0.5 → 0 (mémoire ~3 pulses).
- géométrie SÉPARÉE (grille 2D = point TENACE, chaque région garde son histoire) : 9.7 → 4.3 → 2.2 → 0 (mémoire
  ~6-7 pulses, plus lent MAIS toujours monotone descendante). [j'avais d'abord testé l'overlap seul = géométrie facile ;
  garde auto-attrapée → re-test au point tenace].

**Verdict cheapest-first : `f(k)` descend aux DEUX géométries → pente downward → closure-leaning, PAS de mur.** La
profondeur de mémoire est bornée (géométrie-dépendante, ~3-7 pulses) → l'état persistant se résume au passé récent →
**horizon infini viable à mémoire bornée** (thermalisation-borne-le-buffer, mesurée). Le tueur le plus profond
(mur-mémoire) est ÉVITÉ sur ce substrat, robuste à la géométrie. Caveats : cheapest-first (4 k) ; verdict RIGOUREUX
sommable-vs-divergent exige le long horizon (T grand, pente reste-t-elle géométrique). 1 taux/substrat.

**Implication routeur** : closable-borné → engine viable, mais le routage ne PAIE que si la profondeur de mémoire est
HÉTÉROGÈNE (router par-région : stocke-fin où profond, mémoïse où peu) ; sinon « stocke les N récents » (nul trivial)
suffit → routeur vacant sur la persistance. Questions ouvertes : hétérogénéité de la profondeur + pression de budget.

**CARTE DE VIABILITÉ (état, 2026-07-01)** — 2 vacances + 2 positifs :
- Budget (wake) : VACANT (périodique, in-clé).
- Persistance-perceptibilité (sédiment, canal relief) : sub-JND (in-clé terrain) ; MAIS canal **albedo** → **out-of-key
  perceptible EXISTE** (positif 1).
- Persistance-mur (f(k)) : **PAS de mur, closable-borné** (positif 2, robuste géométrie).
- → **l'ENGINE (persistant, horizon infini, mémoire bornée, contenu perceptible hors-clé) looks viable sur ce
  substrat.** Le payoff du ROUTEUR (C2 : appris bat nul) reste l'OUVERT : closable-trivial → nul « stocke récent »
  suffit ; le routeur ne paie que sur **hétérogénéité de profondeur + pression de budget**, non encore mesurées.

### 2026-07-01 — RECADRAGE : les substrats testés sont des COUPLAGES DOUX ; le pari Cascade vit dans le couplage RAIDE

**Le sédiment-flood était le premier substrat bi-physique COUPLÉ** (pas « une 2e physique ») : fluide shallow-water +
champ persistant sédiment, échange par loi de dépôt path-dependent, `b_eff=b0+s` rétroagissant sur le fluide = la
structure exacte de F-multiphysique (deux conservés distincts, terme d'échange, un qui persiste). Les 3 gardes passées
(hors-clé existe, pas de mur, baseline tient) sont les 3 gardes du multiphysique-au-budget. **Mais c'est le couplage le
plus DOUX** : deux physiques lentes, échange quasi-unidirectionnel (feedback `s→b_eff` faible, mesuré). **L'engine tient
PARCE QUE le couplage est doux.**

**Difficulté NON-ADDITIVE du pari** : N physiques couplées ≠ N coûts additionnés, mais un espace-PRODUIT où chaque
INTERFACE entre régimes peut devenir non-fermable. `resfrac` mesure ça. Sur mono-physique : « le routeur a-t-il un job ».
Sur un COUPLAGE : si l'interface produit de l'irréductibilité locale que le fin doit résoudre et que l'IA pourrait
router = **la thèse centrale de Cascade**. Le multiphysique qui casse le budget = feu×fluide×structure×fracture, échanges
RAIDES bidirectionnels, échelles à plusieurs ordres, interfaces où une physique CRÉE le régime de l'autre. **On n'y est pas.**

**ÉNONCÉ DE CARTE (plus juste que « engine-oui/routeur-non »)** : l'engine atteint le but sur **couplage doux**, sans
routeur (le LOD suffit quand le couplage ne crée pas d'irréductibilité). Le pari — l'IA rend le multiphysique-RAIDE
temps-réel — **n'est PAS testé** : aucun substrat n'a de couplage assez raide. Ni « routeur mort » ni « Cascade prouvé » :
engine validé sur le régime FACILE du pari ; le régime DUR (sa raison d'être) reste devant. Routeur vacant partout car
testé seulement où le couplage est trop doux — wake (trop périodique), sédiment (terrain-dominé), couplage-doux : la
tractabilité EST la douceur qui vide le test. **L'axe manquant = RAIDEUR DE COUPLAGE, pas Re ni échelle seule.** « Monter
le Re » = mono-physique-plus-dur (le réflexe récurrent). F est routé PARCE QU'il est multiphysique : le routage n'a de
raison d'être que parce que des physiques distinctes exigent des experts distincts et que tout résoudre fin sur le
couplage dépasse le budget. Cause commune de la session unifiée : le hors-clé que le routeur route = le contenu
d'INTERFACE de couplage, et seul un couplage raide le produit.

**Distinctions (anti-sur-revendication symétrique)** : « hors-clé existe » (albedo) ≠ « routeur a un job » (resfrac) ;
« engine suffit sur couplage doux » ≠ « routeur inutile » (1 substrat ≠ le pari). Le couplage raide a sa propre trappe
de tractabilité (low-Re/périodique → re-doux) → design = prochaine question paper-grade. **Ordre (fin avant moyen)** :
engine-au-budget-sur-couplage-doux = 1er rang ; resfrac = 2e rang. resfrac sur le couplage doux mesure le PLANCHER :
>0 → routeur a un job sur le couplage le plus faible ; ≈0 (probable) → trop doux → **désigne la raideur comme l'axe**,
« construire un couplage raide » passant d'intuition à nécessité DÉMONTRÉE.

### 2026-07-01 — resfrac PLANCHER : POSITIF sur le couplage doux (surprise vs prédiction), mais SANS enjeu-budget

`resfrac(order_memory | halo)` par fenêtre de Harten, désambiguïsé : le 1.03 initial était sous-alimenté (72 fenêtres) ;
à **182 fenêtres committées** il se stabilise à **0.67–0.85** robuste (WIN 2/4/8, k variés). order_memory = signal PUR
(solveur déterministe, `|albA−albA'|=0`). **→ le hors-clé n'est PAS prédictible du halo (~75 %) → le nul de Harten ne peut
pas le router → le routeur a un JOB LATENT même sur le couplage DOUX.** Contredit la prédiction (≈0). Cause : l'order_memory
(différence d'ordre, path-dependent) n'est pas co-localisé avec le dépôt halo-visible (coarse-`s` = total, pas la
différence d'ordre → halo aveugle là).

**Caveats (le faux-positif = l'erreur symétrique)** :
- Conditionnel au jeu de features du halo (nul plus riche → resfrac plus bas) ; path-dependence garantit `resfrac>0`
  (plancher), la magnitude 0.75 (même avec coarse-`s` dans le halo) est le surplus.
- Valeur-vs-localisation : resfrac-sur-valeur borne la marge du routeur (qui décide par localisation), ne l'égale pas.
- **DÉCISIF (le (f)) : AUCUN enjeu-budget.** Fin-partout tient la frame sur ce petit substrat → « job SANS enjeu » : le
  routeur pourrait router, rien ne l'y oblige. `resfrac>0 ≠` routage PAIE sans pression de budget.

**Lecture réconciliée au recadrage** : le plancher est plus HAUT que prédit — le job latent existe même au couplage le
plus faible → **le pari est en meilleure posture qu'attendu.** Mais l'enjeu (budget) n'est pas sur ce petit substrat doux.
`resfrac>0` RENFORCE le plancher du pari **sans établir le payoff** ; le payoff exige l'enjeu = couplage RAIDE / échelle.
Le recadrage tient (la raideur reste l'axe), désormais avec un **plancher mesuré positif sur le doux**.

**CORRECTION (le fantôme C2-cache à l'ENTRÉE de resfrac)** : le 0.71 brut était sur `mean|albA−albB|` **non-seuillé** —
le risque (c) réel n'était pas le bruit numérique (écarté par déterminisme) mais l'order-memory perceptuellement NULLE
(la garde « la métrique voit ce que le verdict tranche », appliquée aux 3 sorties mais PAS à l'entrée). Re-mesure sur
`order_memory` **supra-JND** (fraction où `|albA−albB|>JND-albedo`) : **resfrac tient à 0.60–0.66** (JND 2/5/10 %, 116–151
fenêtres, supra sur 9–14 % du domaine), vs 0.71 brut → **ne s'effondre PAS**. Le hors-clé est **perceptible**, pas invisible.

**Verdict corrigé** : job latent **perceptible RÉEL** sur le couplage doux. La prédiction « doux → resfrac≈0 » ne tient
pas. **Cause commune NUANCÉE** : forme resserrée (hors-clé = irréductibilité d'interface de couplage) HOLD — le doux l'a
produit *parce qu'il EST un couplage* ; mais « SEUL le raide le produit » = FAUX (même le doux en produit du perceptible).
**→ l'axe discriminant du PAYOFF n'est pas la raideur, c'est l'ENJEU-BUDGET.** Le doux a le job, pas les stakes. Caveats
tenus : conditionnel aux features du halo (0.62 = borne sup) ; valeur-vs-localisation ; **décisif : aucun enjeu-budget**
(job perceptible SANS stakes ≠ « routage paie »). Plancher vindiqué comme **perceptible**, axe budget en discriminant.

**Garde de SORTIE (job existe ≠ job paie) — la sparsité du hors-clé** : le routeur gagne en compute économisé, pas en
valeur non-prédite ; il n'économise que si le non-fermable est SPARSE (concentré, le reste mémoïsable). resfrac (magnitude)
ne donnait pas la structure spatiale. Mesure (WIN=2, JND-albedo) : **f_p = 18 % du domaine exige du fin → S = 1/f_p = 5.5** ;
**concentration : 90 % de la masse supra-JND dans 14 % des fenêtres, Gini=0.86, Moran=0.56** (clusterisé/front, PAS diffus).
La suspicion-diffus (qui aurait tué le payoff même avec enjeu) est **mesurée-écartée** : le hors-clé du couplage doux est
un **front sparse**. → Les 3 choses (sparsité / raideur-la-produit / budget) se recollapsent en 2 : sparsité FAVORABLE
(mesurée S=5.5), donc le discriminant du payoff est bien **l'enjeu-budget** — gagné, pas supposé. La raideur n'est PAS
requise pour le job (le doux l'a, sparse) ; elle augmenterait S (interface plus mince) et G (dynamique plus rapide) = la
MAGNITUDE du payoff, pas son existence. Caveats : `f_p=18%` dépend du seuil de routage (concentration seuil-robuste) ;
S=5.5 à ce point d'op.

**Discriminant final** — et CORRECTION (`S>G` cache deux fuites, mauvaise unité) : `S=5.5=1/f_p` est géométrique
(économie SPATIALE). Le critère n'est PAS `S>G` (approximation overhead-nul, interdite par la conclusion T1). En entier :
**routé tient ⟺ `C_routé < budget < C_fin`**, `C_routé = f_p·C_fin + (1−f_p)·C_memo + C_overhead`. Deux termes que `S>G`
jette, tous deux rabotant S vers le BAS :
- **C_overhead** : nul de Harten (features + décision) × 182 fenêtres — à S=5.5, pas du second ordre (T1 : « routeur
  radicalement moins cher que la décision qu'il évite »).
- **C_memo** : mémoïser ≠ gratuit (lecture cache + reconstruction). C_memo~5 %·C_fin → `S_eff≈1/(0.18+0.82×0.05)≈4.5`.
→ **`S_geom=5.5` = plafond géométrique** (concentration seuil-robuste : Gini 0.86 / Moran 0.56 — solide ; niveau
seuil-dépendant). **`S_eff = C_fin/C_routé net < 5.5, NON MESURÉ.** La moitié-C2 réellement mesurée est S_eff, pas S_geom.

**Prochain chiffre = le COUPLE `(S_eff, G)`, pas G seul.** S_eff = la mesure surprenante (exige C_overhead + C_memo,
mesurables ICI sur le nul-de-Harten que la spec impose de coder en premier — C_overhead gratuit) ; G = extrapolation
coût-fin (arithmétique baseline, peu d'info — `G(2D)` paie 313²–735², `G(3D)` 46³–81³ ; à l'échelle-jeu 3D G≫S → exige
hors-clé razoir-mince = front codim-1 = **la raideur revient comme nécessité à l'échelle**, via S, pas via le job).
Critère gravé : `C_routé(net) < budget < C_fin`, jamais `S>G`.

### 2026-07-01 — PREMIER BUILD DE F : `(S_eff, G)` mesuré ; `S_eff` dépend de la granularité de routage

Premier vrai build du routeur de l'arc (nul de Harten + memoize coarse+terrain + profil FLOPs, JIT-indépendant) sur le
sédiment-flood. **Contrôle interne Q1 PASSE à 93.8 %** (fenêtres fermables reconstruites sous-JND par le memoize) →
`f_p` et `C_memo` cohérents (6 % d'optimisme résiduel mineur). FLOPs/cell comptés : `C_fin=310`, `C_memo=6`, `C_overhead=9`.

**`(S_eff, G)`** : à WIN=4 (`f_p=37%`) → **`S_eff=2.45`** ; à WIN=2 (`f_p=18%`, la granularité du S_geom gravé) →
**`S_eff=4.45`**. `G(64²)=0.042 ≪ 1`. La correction overhead+memo est confirmée : `S_geom=5.5 → S_eff=4.45` à WIN=2
(rabot ~1.0).

**Trou exposé par le build (réel) : `f_p` — donc S_eff — dépend de la GRANULARITÉ de routage** (taille de fenêtre de
Harten + seuil needs-fine). Un front mince contamine plus de fenêtres-en-fraction quand la fenêtre grossit → S baisse.
**`S_eff ∈ ~2.5–4.5`** selon WIN ; le `S_geom=5.5` gravé était le coin le plus fin/optimiste. Le routeur réel a une
**granularité optimale** (fin = S haut mais C_overhead haut) — un knob non encore optimisé.

**Lecture (anti-sur-revendication, double)** : `G≪1` → budget PAS en jeu ici → `(S_eff, G)` est l'**économie nette du
routeur là où rien ne l'oblige**, PAS le verdict Cascade. Ni « le routage paie » (aucun enjeu), ni « S trop petit »
(granularité non optimisée, échelle non testée). Le verdict attend `G>1` (échelle où fin-partout casse la frame) —
non atteignable sur le matériel actuel → extrapolation, pas clôture. **C'est le premier S_eff net du livrable, sur le
premier substrat qui a un job ; il dit l'économie, pas la viabilité.**

### 2026-07-01 — `f_p(L)` MESURÉ (le terme caché de l'extrapolation) : quasi-codim-1 → la fenêtre s'ouvre, mais via le LOD-fovéa

**Le danger corrigé** : `G(L)` est arithmétique (coût-fin ∝ cellules), mais le critère est la FENÊTRE `C_routé<budget<C_fin`,
et `C_routé` contient `f_p`, qui NE scale PAS comme G. J'avais écrit deux choses contradictoires sur `f_p(L)` (« ~invariant »
au resfrac ; « razor-mince → décroît » au codim-1) — une hypothèse déguisée en calcul qui décide le verdict Cascade.
**Mesuré** (même physique self-similaire, dx fixe, monde ×4, `f_p` en CELLULES physiques pas fenêtres) :
`f_p = 7.08 / 4.50 / 2.40 %` à `L = 10/20/40` → **`f_p ∝ L^(-0.78)`** : quasi-codim-1 (α=0.78, loin du filling α=0),
léger épaississement `w ∝ L^0.22`. L'intuition « invariant » RÉFUTÉE. `S=1/f_p ∝ L^0.78` croît → scaling MESURÉ, pas décrété.

**Extrapolation de la fenêtre (α mesuré + hypothèses affichées)** : `S(L)∝L^0.78` mais `G(L)∝L^d` (d=2/3) → G croît PLUS
VITE. Fenêtre `1<G<S` ferme à `L/L₀ = S₀^(1/(d−α))` : `S₀≈3` → 2D ~2.5×L₀, 3D ~1.7×L₀. **Étroite → SANS LOD-fovéa, le
routage ne paie pas à la vraie échelle-jeu** (G∝L^d dépasse S∝L^0.78).

**→ Le LOD-fovéa est le mécanisme LOAD-BEARING** (pas un bonus) : il borne le fin à la VUE de l'observateur (taille fixe),
pas au monde → l'échelle effective reste dans la fenêtre quelle que soit la taille du monde. Dans la fovéa, routage paie à
`S≈3` (doux, mesuré) ; un couplage RAIDE y monterait S (front plus mince, α→1).

**VERDICT DE VIABILITÉ (gagné par chaîne mesurée, pas clôture)** : Cascade tient au budget-jeu **ssi** (a) LOD-fovéa borne
le fin à la vue — l'architecture l'a (distance = plafond LOD) — ET (b) le routage paie dans la fovéa — mesuré `S≈3` sur le
couplage doux. **Les deux mécanismes du pari sont présents et l'un est mesuré.** Restent à mesurer : le budget-fovéa absolu
(fine-fovéa > frame ?) et le gain de S par raideur. Première fois de l'arc que la moitié-C3 a un **scaling `f_p(L)` mesuré**
au lieu d'un décret — le verdict ne sort plus du choix d'hypothèse mais de la physique.

### 2026-07-01 — CORRECTION : (a) n'est PAS acquis-par-l'architecture ; budget-fovéa mesuré ; le pin JND/fovéa = dernier load-bearing

**Inversion corrigée** : j'avais gravé (a) « LOD borne le fin à la vue » comme ACQUIS (architectural) et (b) `S≈3` comme le
maillon mesuré. C'est inversé : (b) est mesuré ; (a) « le fin-fovéa TIENT LA FRAME » est **NON mesuré** (borné ≠ sous budget ;
intention architecturale ≠ mesure). Le LOD-fovéa fait DEUX choses (j'en avais gravé une) : (i) borne le fin à la fovéa ; (ii)
**sparsifie la périphérie** (JND croît avec l'excentricité → loin, presque tout mémoïsable). Mon `S≈3` était à JND UNIFORME.

**Budget-fovéa mesuré (`S_eff` sous champ-JND-de-distance, balayé sur `r_fovea` — perceptuel, sensibilité pas choix)** :
`r_fovea/rmax = 1.0/0.3/0.15` → `S_eff = 1.8/2.3/3.9` (périphérie JND ×1/×3/×6). **Le LOD multiplie bien S (mécanisme ii
confirmé) mais ~2× sur la plage** — pas libérateur. Et **`S_eff` DÉPEND de `r_fovea`, perceptuel NON-PINNÉ** → (a) hérite de
l'arbitraire JND/fovéa = le référent que tout l'arc a manqué. Choisir `r_fovea` pour que le budget passe = fabriquer le verdict.

**ENDPOINT HONNÊTE** : la chaîne est mesurée-favorable de bout en bout SAUF un maillon — le **pin perceptuel JND/fovéa** — et
ce n'est pas un trou de plus : c'est le référent que ce substrat **computationnel ne peut pas produire** (exige un modèle
perceptuel / étude). Engine viable + routeur job-sparse (`S≈2.5–4.5`) + `f_p∝L^{-0.78}` (codim-1) + LOD multiplie S (~2×) :
tout mesuré, tout favorable. Le verdict ABSOLU se réduit à *le JND/fovéa est-il tel que S(LOD-multiplié) batte G(échelle-fovéa)*.
**Le couplage RAIDE a une justification neuve, mesurée** : un S plus haut (front plus mince, α→1) rend le verdict **ROBUSTE au
pin non-fait** (passe à `r_fovea` plus large) — pas « plus de marge » mais **l'assurance contre le seul référent non-pinnable ici**.

### 2026-07-01 — CORRECTION D'INVERSION : la fenêtre se FERME (le LOD-bornage est le pilier) + r_fovea bordé par l'acuité → penche positif

**Inversion corrigée (j'avais coché la gravure favorable de deux qui se contredisent)** : j'avais gravé « la fenêtre s'ouvre ✓ »
alors que j'avais MESURÉ qu'elle se FERME (`G∝L^d > S∝L^0.78`, ferme à ~1.7–2.5×L₀). Le codim-1 n'ouvre pas la fenêtre, il
ralentit la fermeture. **Donc le verdict ne tient pas sur le scaling — il pend ENTIÈREMENT au LOD-fovéa = le pilier, pas une
cerise** (sans lui : défavorable, pas neutre). Et le pilier du LOD n'est PAS le ×2 périphérique (bonus, facteur constant vs
exposant divergent) — c'est le **BORNAGE D'ÉCHELLE** : la fovéa fixe `L_eff` donc fixe `G` → sort de la course `G∝L^d` =
changement de régime. Et le raide n'échappe pas au pin : `α→1 < d` → ferme toujours à l'échelle non-bornée, déplace seulement
le seuil `r_fovea`, toujours conditionnel au pin. (« robuste au pin » était l'optimisme de « la fenêtre s'ouvre ».)

**(1) le pin n'est PAS hors-scope** : « au budget d'un jeu » EST perceptuel → changement d'instrument substrat→référent
perceptuel (comme AegirJAX→XLB, diffusif→MUSCL). **Bornage `r_fovea` par acuité** : haute-acuité fovéale ~2°, FOV jeu ~90° →
`r_fovea/vue ≈ 0.022`. Balayage : `r_fovea=0.15→S_eff=3.9`, `S_eff∝1/r_fovea²` → à `r_fovea≈0.022`, `S_eff` en dizaines. Pilier
vu juste : zone fine `≈(0.022·cellules_vue)²` = **bornée par l'ÉCRAN, pas le monde** → coût-frame screen-bounded, hors `L^d`.

**VERDICT DIRECTIONNEL (pas clôture)** : au `r_fovea` honnête (petit), pilier-bornage en régime FORT → coût-fine screen-bounded
ET petit → **penche POSITIF**, et **le raide est probablement DÉCORATIF** (le doux suffit à S_eff élevé sous petite fovéa).
**Conditionnel** au dernier terme : budget absolu écran = `C_fin × cellules_fovéa` vs frame, exige la résolution-fovéale
(modèle rendu/caméra). Caveats order-of-magnitude (acuité, mapping r_fovea↔angle, résolution-fovéale), pas cloués. **Endpoint
non atteint : il manque le budget-écran absolu (caméra), pas une finition — mais la direction, pour la 1re fois, penche positif
sur le pilier bordé par la physique perceptuelle, pas par décret.**

### 2026-07-01 — CORRECTION ARITHMÉTIQUE : le budget-écran n'est PAS directionnel-positif — INDÉTERMINÉ en 3D-volumétrique

**Erreur (sens favorable, biais+faute renforcés)** dans l'estimation `~340 MFLOP/s, tient par 10⁴×` : (1) **falaise-pas-rampe** —
j'ai binarisé l'acuité (2° fin / reste gratuit) alors qu'elle décroît CONTINÛMENT (mécanisme ii, ~50% à 2.5°, ~20% à 10°) →
bien plus de cellules partiellement-fines que les 14k de la falaise ; (2) **2D-pas-3D-volumétrique** — la fovéa est un FRUSTUM,
`cellules-fines ≈ r²×profondeur`, et pour eau/fumée/feu (les readouts de Cascade) PAS d'occlusion → rayon traverse le volume →
profondeur ~10³ → `14k → ~10⁷`. **Le pilier-bornage borne l'échelle LATÉRALE (écran), PAS la profondeur du volume fin** →
screen-bounded vrai en 2D, FAUX en 3D-volumétrique (le cas de Cascade par sa thèse même).

Les deux corrections **consomment exactement le 10⁴×**. → **Budget INDÉTERMINÉ à l'OoM en 3D-volumétrique**, bascule sur la
**profondeur du frustum fin** — terme qu'aucun proxy sédiment ni estimation 2D ne produit.

**Conséquence** : (a) le F-build n'est pas « confirmer le positif » mais **MESURER le seul terme qui peut basculer** (profondeur
de frustum fin volumétrique sous champ-JND-continu), qui penche moins bien qu'estimé → construire F sur **le cas qui peut le
FALSIFIER** (volumétrique, profond, observé en perspective), PAS un monde 2D/peu-profond qui passe (= faux-PASS par choix de
substrat, une dernière fois). (b) Le raide n'est PAS « probablement décoratif » (l'estimation qui le rendait tel vient de tomber)
— il est **décoratif-OU-nécessaire selon la profondeur de frustum** : `α→1` mincit le front → réduit la profondeur fine traversée
par le rayon → attaque le terme indéterminé. **Un seul inconnu — profondeur de frustum fin volumétrique — gouverne ET le verdict
budget ET le besoin du raide.** L'endpoint : pas « tout favorable, build pour confirmer », mais « dernier terme indéterminé en 3D,
penche-possiblement-négatif, seul F tournant sur le cas falsifiant le tranche ».

---

### 2026-07-01 — Dynamique-z : le contrôle β-max A TIRÉ (extension passive triviale) ; re-design Boussinesq/Ra, critère gravé AVANT build

**Fait** : le build passif (`dyn_z.py`, scalaire advecté par vélocité posée, balayage β prévu) a été soumis au
contrôle pré-enregistré « β-max doit casser le washout AVANT tout balayage ». **Le contrôle a tiré** :
Δpixel(β=4.0) = 0.0% → l'extension-z était TRIVIALE (coarse-z ne diverge pas du fin même à couplage violent)
→ le balayage aurait mesuré zéro PAR CONSTRUCTION (faux-favorable structurel, l'analogue du domaine-mort).
Balayage NON exécuté. Diagnose : un scalaire PASSIF n'a aucune rétroaction fin-z→horizontal ; la projection
lave toute différence-z qui ne rétroagit pas. (Corollaire honnête : le sédiment sans flottabilité est
PHYSIQUEMENT sans moteur de structure-z — sa trivialité était correcte, pas un artefact.)

**Le résultat conceptuel du tour (posé par la diagnose, retourné par Romain)** : le SEUL canal par lequel la
dynamique-z peut coûter au budget est la rétroaction fin-z → structure horizontale projetée. La structure-z
qui ne rétroagit pas est lavée par la projection → gratuite quelle que soit sa richesse. La question n'est
pas « la dynamique-z est-elle chère » mais **f_p-dynamique-projetée** = fraction de la structure-z fine dont
la rétroaction horizontale dépasse le JND APRÈS projection — le 3ᵉ f_p (après f_p-latéral et
f_p-readout-profondeur≈0).

**Critère gravé AVANT le build suivant (leçon miroir de l'artefact-3 métrique-÷L)** : la projection est un
amortisseur de la divergence DYNAMIQUE, pas seulement du readout. Divergence-volume ≠ divergence-projetée,
et c'est la projetée qui touche le budget. → **Ra\* (et tout β\*/κ\* de dynamique-z) se mesure sur
Δpixel-PROJETÉ sous rollout, JAMAIS sur Δ-volume** — sinon faux-DÉFAVORABLE (divergence-volume que la
projection efface avant le pixel). La bonne quantité est toujours ce qui atteint le pixel.

**Design pré-enregistré (aucun point posé par l'exécutant ; les deux gardes symétriques)** :
- Substrat : **panache Boussinesq, tranche x-z 2D-verticale** (le moins cher qui possède la rétroaction
  fin-z→horizontal par construction physique). 3D différé jusqu'à ce que la tranche tranche.
- **Échelle-z ÉMERGENTE, non tunée** : elle sort de l'instabilité (mode le plus instable ← Rayleigh),
  personne ne la pose. C'est la garde anti-biais sur le DOF neuf.
- Paramètre balayé : **Ra** (Ra bas → structure large, coarse-z résout ; Ra haut → plumes sous-maille).
  Sortie : **Ra\*** = seuil où la rétroaction projetée passe le JND, confronté au Ra tabulé des media réels
  (fumée chaude/flamme : Ra haut ; sédiment : Ra~0).
- Mesure : **Δpixel-projeté(t)** entre run coarse-z et run fin-z (coarse initialisé = z-coarsening du fin),
  lecture de forme : linéaire = rétroaction sans borne = le budget paie la profondeur (défavorable) ;
  plateau = thermalise/sature = closure sur l'axe z-dynamique (cheap).
- **Contrôle AVANT balayage, corrigé** : à Ra-max, Δpixel-PROJETÉ doit casser le washout. S'il ne casse
  pas → « la projection amortit même la convection violente » = dynamique-z cheap POUR DE VRAI (résultat
  fort et favorable, gagné contre le bon contrôle). S'il casse → le balayage a un signal, Ra\* mesurable.
- **Neutralité imposée dans les deux sens** : ni « les media actifs sont probablement pas cheap »
  (pessimisme-sur-correcteur, miroir du ~10³) ni washout-readout extrapolé (l'intégrale-le-long-du-rayon
  moyenne pareil le passif et l'actif). Ra\* peut tomber des deux côtés ; le balayage seul tranche.

---

### 2026-07-01 — Boussinesq/Ra : contrôle cassé, balayage mesuré ; 7ᵉ knob (Δ₀ jumeaux) tué par la garde-en-taux ; verdict de la dynamique-z par rapport de pentes

**Contrôle Ra-max (10⁷)** : Δpixel-projeté final = 18 % (κ=4, JND=5 %) → CASSE → le balayage avait un signal.
**Balayage (Δpixel-projeté ponctuel, plateau t∈[10,25] retournements)** : Ra\* encadré [3e4,1e5] (mâchoire
transparente J=2 %) à [3e5,1e6] (absorbante J=5 %). Forme = PLATEAU (pas de mur-z) mais plateau HAUT
(60–80 % pixels >JND). Media actifs réels (fumée/flamme Ra≳1e6) au-dessus ; sédiment (Ra~0) en-dessous.

**7ᵉ knob attrapé (Romain) — dans la garde elle-même** : la magnitude de perturbation des jumeaux (1e-6)
décidait le verdict (t_c−t_j = (1/λ)·ln(Δ₀_coarse/Δ₀_jumeaux) : 1e-3 → « sur-compte » favorable, 1e-12 →
« signal réel » défavorable). Le « plancher de chaos = 0.0 % » du run précédent ne voulait rien dire.
**Garde rebâtie sans magnitude : comparer des TAUX, pas des trajectoires** — λ = pente log des jumeaux
(propriété de l'écoulement), discriminant = pente(Δ coarse-fin projeté) vs λ_proj :
≈λ → chaos intrinsèque (le coarse déplace la CI sur l'attracteur ; la mémoïsation ne bat pas ça non plus) ;
>λ ou saut non-exponentiel → destruction structurelle = coût réel du coarse-z.

**Mesuré (λ en 1/retournement ; invariance au Δ₀ DÉMONTRÉE : 1e-6 et 1e-8 → pentes identiques aux 3 Ra)** :

| Ra | λ_vol jumeaux | λ_proj jumeaux | pente coarse-fin proj | lecture-taux |
|---|---|---|---|---|
| 1e5 | 0.423 | 0.335 | 0.165 | ≤λ → chaotique-intrinsèque |
| 1e6 | 0.161 | 0.294 | 0.320 | ≈λ → chaotique-intrinsèque |
| 1e7 | 0.360 | 0.434 | **1.160** | **>>λ (×2.7) → structurel** |

- L'écoulement est chaotique aux 3 A (λ>0) ; l'« amortissement du chaos par la projection » n'est PAS
  uniforme : λ_vol−λ_proj = +0.09 (1e5) mais −0.13 (1e6), −0.07 (1e7) — la projection n'amortit pas le TAUX
  de chaos à haut Ra. En revanche elle amortit le SAUT structurel de volume : d_vol(coarse-fin) saute
  immédiatement à ~0.15 (aucune fenêtre exponentielle, structurel en volume) que la projection lave à 1.7e-4.
- **M1/M3 statistiques du film projeté (Ra=1e5, 1e6)** : luminance moyenne identique (≤0.1 %), puissance par
  octave coarse/fin ∈ [0.82, 1.09], enveloppe temporelle ratio 0.98/1.05 → **le film coarse-z est
  statistiquement le même film**. Le plateau ponctuel 60–80 % était de la DÉCORRÉLATION, pas de
  l'implausibilité — le Δpixel ponctuel-contre-contrefactuel sur-comptait, rejouant la leçon L2/C2 sur l'axe
  temporel (personne ne voit jamais le run fin).

**Portée EXACTE du verdict** : à Ra ≤ 1e6, la dynamique-z est cheap AU SENS FORT (divergence coarse-fin au
taux du chaos intrinsèque + statistiques perceptuelles conservées = le coarse-z produit une AUTRE réalisation
plausible du même film). À Ra = 1e7, signal super-λ (×2.7) = destruction structurelle possible — NON tranché :
(a) le fin lui-même y est sous-résolu (Nz=64, caveat gravé d'avance → le chiffre est suspect dans les deux
sens), (b) fenêtre de fit courte (48 pts), (c) M1/M3 non mesurés à 1e7. La frontière structurelle vit dans
(1e6, 1e7] et les media cibles (fumée/flamme) chevauchent cette zone. Prochain pas falsifiant le moins cher :
M1/M3 à Ra=1e7 + refaire 1e7 à fin plus résolu (Nz=128) pour tester si le super-λ est physique ou artefact
d'instrument. Arbitrage de la fourche ponctuel-vs-statistique : les deux mesures pointent la même lecture,
mais elle reste à valider par Romain (pas de fermeture unilatérale côté favorable).

---

### 2026-07-01 — Validation d'instrument à Ra=1e7 : le super-λ PERSISTE à fin doublé + M1 distordu → la frontière structurelle est PHYSIQUE, dans (1e6, 1e7]

**Même design gravé, fin Nz=128 (vs 64), coarse Nz=64, ratio de coarsening inchangé (2), Ra=1e7** :
- λ_proj jumeaux = 0.339 ; pente coarse-fin projetée = 0.973 → **ratio 2.87** (vs 2.67 à Nz=64).
  Le super-λ N'EST PAS un artefact de sous-résolution : il persiste, même légèrement renforcé, quand le fin
  double. La destruction structurelle par le coarse-z à Ra=1e7 est physique.
- **M1 à 1e7 (241 trames)** : octaves coarse/fin = 0.63 / 1.25 / 0.79 / 0.93 / 0.86 / 0.66 / 0.63 — hors de
  la bande ±10 % vue à 1e5/1e6. La bande dominante (kx=1, 39 % de la puissance) perd 37 %, kx 2-3 gagne 25 % :
  le coarse-z DÉCALE la distribution des tailles de plumes projetées → **film statistiquement DIFFÉRENT**.
  M3 (enveloppe) reste ~1 (0.97) : le scintillement survit, c'est la GÉOMÉTRIE spectrale qui casse.

**Les deux mesures indépendantes (taux ET statistiques) convergent — cette fois côté DÉFAVORABLE** :
à Ra=1e7 le coarse-z ne produit plus « une autre réalisation plausible » mais un film aux mauvaises tailles
de structures, divergeant 2.9× plus vite que le chaos. Symétrie avec Ra≤1e6 (les deux mesures convergeaient
côté cheap) : le discriminant fonctionne dans les deux sens, il n'est pas un instrument-à-confirmer.

**Verdict de la dynamique-z (portée exacte)** : TERME À DEUX CÔTÉS AVEC FRONTIÈRE. Cheap au sens fort pour
Ra ≤ 1e6 ; coût réel (structurel, budget paie la profondeur fine) à Ra = 1e7, frontière dans (1e6, 1e7].
Les media cibles du pari raide (fumée chaude Ra~1e8-1e10, flamme Ra~1e6-1e8) sont AU-DESSUS ou À CHEVAL →
pour eux, le coarse-z seul ne suffit pas ; le sédiment (Ra~0) reste cheap. Limites de portée : tranche 2D
x-z, Boussinesq seul, ratio de coarsening 2, M1 à une seule mâchoire (κ=4). L'implication architecturale
(la fovéa-LOD doit-elle inclure z ? est-ce le rôle du `descend` du routeur ?) = arbitrage Romain, pas fermé ici.

---

### 2026-07-01 — Trois prises sur le verdict dynamique-z ; seuils gravés AVANT les mesures qui suivent

**Prise petite (gravée comme MISS, pas comme arrondi)** : le critère M1 gravé est ±15 % de puissance par
bande. À Ra=1e6, kx 32-63 : coarse/fin = 0.82 → **MISS** (3 points sous le seuil ; idem kx=64 à 1e5 : 1.35).
Le rapport précédent l'a emballé dans un PASS (« ∈ [0.82, 1.09] ») — motif « rigueur habillée », noté.
Analyse à produire (mesure, pas exemption rétroactive) : contraste-Weber porté par la bande (χ = RMS_bande/⟨L⟩)
et Δχ fin-coarse vs JND=2 %. Deux issues possibles, énoncées avant le chiffre : Δχ ≥ JND → le miss est
perceptuellement réel, le verdict « statistiquement même film » à 1e6 est ENTAMÉ ; Δχ ≪ JND → le miss est réel
au sens du critère mais sous-perceptuel → limite DU CRITÈRE (bandes non-porteuses sur-comptées par le ±15 %
non pondéré), à corriger dans le critère pour les usages FUTURS, jamais rétroactivement.

**Prise moyenne — le discriminant n'avait ni seuil ni barre d'erreur (runs uniques, lecture post-hoc).
SEUIL GRAVÉ MAINTENANT, avant toute mesure multi-graines** : N ≥ 4 graines (θ₀), par graine r = pente(Δ
coarse-fin projeté)/λ_proj(jumeaux). Verdict : **STRUCTUREL si mean(r) − 2·SEM > 1 ET mean(r) ≥ 1.5 ;
CHAOTIQUE-INTRINSÈQUE si mean(r) + 2·SEM < 1.5 ; sinon INDÉTERMINÉ** (pas de lecture à l'œil). Le 1.5 est
conventionnel et gravé avant les graines ; la zone indéterminée est le prix de l'honnêteté.

**Prise grande — l'expérience ne distingue pas les deux façons de payer.** Le « coarse-z » mesuré est un
solveur TRONQUÉ NAÏF ; le niveau grossier de l'architecture est F-avec-expert, potentiellement corrigé
sous-maille. La frontière mesurée est donc une BORNE SUPÉRIEURE sur « où le grossier a besoin d'aide », pas
une preuve que l'aide = résolution (miroir de la leçon oracle-diffusif : un coarse trop naïf rend le fin
artificiellement nécessaire). Le détail M1 ferme UNE branche : kx=1 perd 37 % → l'erreur est à l'échelle
RÉSOLUE → une closure de READOUT conditionnée sur des invariants grossiers faux hérite du faux (la licence
Kolmogorov ne sauve pas ce régime au readout). Mais il ouvre l'autre : erreur résolue CAUSÉE par la troncature
sous-maille = exactement ce qu'une closure DYNAMIQUE au niveau grossier répare peut-être.

**Test gravé (le moins cher qui peut échouer, AVANT toute spec fovéa-z)** : coarse FERMÉ vs coarse naïf,
Ra=1e7, instrument validé (fin Nz=128, ratio 2), closure = Smagorinsky bête (Cs=0.17, Δ=√(dx·dz), Pr_t=1,
ν_t clampé pour stabilité), même discriminant (jugé au seuil gravé ci-dessus), mêmes M1/M3.
Lectures : ratio s'effondre sous le seuil intrinsèque ET kx=1 revient dans ±15 % → **le job appartient à la
sortie EXPERT** (meilleur opérateur grossier, coût quasi-coarse), descend au chômage ; ratio tient contre une
closure honnête → **descend garde le job**, escalade licenciée par un test qui pouvait la tuer.

**Gardes de portée** : (a) treadmill — NE PAS raffiner Ra\* intra-décade (aucune décision n'en dépend : les
media cibles sont au-dessus quoi qu'il arrive) ; (b) croissance-vs-saturation au-delà de 1e7 = chère, GATÉE
derrière le test closure ; (c) le volumique saute structurellement aux TROIS Ra — seule la projection (κ=4,
une mâchoire) sauve 1e5–1e6 : **le « cheap » vit dans le readout, pas dans z** ; Ra\* n'est PAS un scalaire
du médium mais un champ fovéa-dépendant (cohérent avec la réserve JND-champ gravée) — ne pas graver
« Ra\* constant ». (d) Descend-comme-spec-fovéa-z : hypothèse LÉGITIME, conclusion ILLÉGITIME tant que le
coarse fermé ne l'a pas ratée. Concession actée en sens inverse (Romain) : Boussinesq donne au `descend` son
premier job physiquement localisé — c'est le premier résultat qui donne à la 3ᵉ sortie autre chose qu'une
existence de principe.

---

### 2026-07-01 — Résultats P1/P2/P3 : le coarse FERMÉ reste structurel à 1e7 (descend garde le job) ; RÉVISION : 1e6 devient INDÉTERMINÉ sous barres d'erreur ; les MISS M1 sont sous-perceptuels (limite du critère)

**P1 (le MISS 0.82)** : contraste-Weber par bande — TOUS les Δχ sont ≪ JND=2 % (max 0.42 % à 1e6, kx 2-3 ;
somme quadratique ~0.5 %). Les MISS ±15 % (dont de NOUVEAUX apparus sur ce run : kx 2-3 = 1.35 à 1e6) sont
réels au sens du critère et sous-perceptuels au sens Weber → **limite DU CRITÈRE actée** : le ±15 % par bande
non pondéré sur-compte les bandes non-porteuses ; pour les usages futurs, M1 se juge sur Δχ vs JND (χ = 
RMS_bande/⟨L⟩), le ±15 % reste diagnostic. Jamais rétroactif : les MISS restent des MISS au journal.
**Note d'instrument découverte en passant** : les ratios de bande single-run sont INSTABLES (kx 2-3 à 1e6 :
1.09 sur 31 trames → 1.35 sur 240 trames, même graine, même trajectoire) et le spread inter-graines de kx=1
est énorme (0.50–1.12 à 1e6 ; 0.12–2.31 à 1e7 fermé) → toute lecture M1 par bande exige des barres d'erreur.

**P2 (discriminant au seuil gravé, 4 graines)** :
| Ra | coarse | r (±2·SEM) | verdict gravé |
|---|---|---|---|
| 1e6 | naïf | 1.17 ± 0.55 | **INDÉTERMINÉ** |
| 1e6 | fermé | 1.19 ± 0.32 | INDÉTERMINÉ |
| 1e7 | naïf | 2.85 ± 0.46 | STRUCTUREL |
| 1e7 | fermé | **2.36 ± 0.69** | **STRUCTUREL** |

**RÉVISION d'un verdict antérieur (la prise moyenne avait raison)** : « cheap au sens fort à Ra ≤ 1e6 » ne
tient plus tel quel. Le single-run r=1.09 à 1e6 était de la chance de graine (graine 23 → r=1.85, λ=0.252).
État honnête de la frontière : 1e7 STRUCTUREL robuste (4/4 graines, tient au spot-check Nz=128 : r=2.53) ;
1e6 = statistiques perceptuelles conservées (Δχ ≪ JND) mais TAUX INDÉTERMINÉ ; 1e5 jamais multi-grainé
(single r≈0.5). La zone indéterminée est réelle, pas un échec du seuil — c'est lui qui l'a exposée.

**Test closure (gravé au tour précédent, verdict rendu)** : le Smagorinsky bête NE fait PAS s'effondrer le
super-λ à 1e7 (2.85 → 2.36, toujours STRUCTUREL, mean−2·SEM = 1.67 > 1). La clause kx=1 « revient dans
±15 % » n'est PAS claimable : moyenne 1.00 par ANNULATION d'un spread 0.12–2.31 (P3 : le fermé répare kx=1
0.63→0.88 mais sur-dissipe le haut du spectre, 0.53/0.23 — signature Smagorinsky classique).
→ **Le `descend` GARDE le job à Ra=1e7** — l'escalade architecturale est licenciée par un test qui pouvait
la tuer. PORTÉE : contre CETTE closure (Smagorinsky bête, celle nommée au test gravé) ; une closure apprise
reste un contender non testé — c'est une mesure future LÉGITIME mais non due (le test gravé est rendu),
à gater derrière une décision qui en dépendrait réellement (anti-treadmill).

---

### 2026-07-01 — ARC B-bis rendu : ZONE GRISE (une cellule) — le canal statistique ne confirme PAS le structurel de 1e7 ; « les deux mesures convergent côté coût » est DÉGRADÉ ; marges fines en titre : Cs=0.17 → +0.17, Cs=0.23 → −0.27 vs 1.5

**Canal TAUX (balayage Cs, 4 graines appariées, format R3)** : naïf r=2.85±0.46 (marge vs 1.5 : +0.89) ;
Cs=0.10 : 2.73±0.42 (+0.81) ; Cs=0.17 : 2.36±0.69 (**+0.17, fine**) ; Cs=0.23 : 2.14±0.91 (**−0.27** —
STRUCTUREL sous la règle 0a [mean−2·SEM=1.23>1 ET mean≥1.5] mais l'intervalle CHEVAUCHE le seuil 1.5).
Tendance honnête : la marge s'amincit MONOTONEMENT avec Cs, et la graine 42 s'effondre à Cs fort
(r=1.33 à 0.17, r=0.81 à 0.23) — le spread CROÎT avec la closure. Le structurel-au-taux tient au balayage
Cs sous la règle gravée, mais pas confortablement : c'est une conclusion à marge fine, pas un plancher.

**Canal STATISTIQUE (Δχ, 4 graines, ~240 trames/graine, porteuses gravées avant lecture = kx 1-1, 2-3,
4-7, 8-15)** : TOUT est sub-JND robuste à JND ∈ [2,4] % — naïf ET fermé (max : kx=1 naïf 0.94±0.45 %,
fermé 0.70±0.65 %). Une seule cellule grise : **kx=1 à JND=1 %** (mean+2·SEM = 1.39 %/1.35 % > 1 %, ni
supra ni sub robuste — barres trop larges, pas signal). M3 : naïf 0.91±0.08, fermé 0.94±0.08 (~1).
**Verdict par la règle gravée : ZONE GRISE** (Bb2 exigeait sub robuste jusqu'à JND=1 % partout ; raté sur
la seule cellule kx=1@1 %). En substance, penche Bb2 : rien de supra-JND nulle part, à aucun JND.

**Item 2 — dégradation gravée (annoncée avant lecture, confirmée)** : le FAIL M1 du naïf à 1e7
(kx=1 = 0.63 en puissance, single-run) se dégrade en **Δχ = 0.94±0.45 % = sub-JND à 2 %** multi-graines.
« Les deux mesures indépendantes convergent côté coût à 1e7 » (entrée du 2026-07-01, validation Nz=128)
est RÉVISÉ : **seul le canal TAUX porte le verdict structurel à 1e7** ; l'écart M1 était réel en puissance
et sous-perceptuel en contraste-Weber. Le structurel de 1e7 reste robuste (4/4 graines, balayage Cs,
spot-check 128) mais il est un fait de DYNAMIQUE (super-λ) sans contrepartie mesurée sur les axes
perceptuels actuels (contraste spatial stationnaire, enveloppe).

**Conséquences (règle de l'addendum appliquée)** : B2 reste NON-DÉFINITIF — structurel-taux-seul, stats en
zone grise. La spec fovéa-z reste SUSPENDUE (Bb1 seul licenciait pleinement). La question « qu'est-ce que
le super-λ détruit que le contraste-Weber ne voit pas ? » (candidats : cohérence temporelle des structures,
géométrie des transitoires) est la sortie naturelle vers l'Arc C — mais la règle gravée ne la déclenchait
que sur Bb2 STRICT : posée ici comme CANDIDATE, arbitrage Romain. Lever la cellule grise kx=1@1 % coûterait
des graines supplémentaires — dépense non due (anti-treadmill) sauf si l'arbitrage en dépend.
Appariement consigné : même θ0 par graine, coarse naïf/fermé = même zdown, même base fine.

---

### 2026-07-01 — Correction de règle (Cs=0.23 = INDÉTERMINÉ), regravure canal taux « 2/3 + tendance », requalification du discriminant juge→détecteur, pré-enregistrement ÉTAGE 1

**P1 — Correction d'étiquette (erreur de règle dans le tableau précédent)** : sous la référence R3 (le seuil
est 1.5, jamais 1), structurel = IC entièrement au-dessus de 1.5. À Cs=0.23, mean−2·SEM = 1.23 < 1.5 →
la cellule est **INDÉTERMINÉE**, pas « STRUCTUREL (0a) mais ». Canal taux REGRAVÉ : **structurel sur 2 Cs
sur 3** (naïf +0.89, Cs=0.10 +0.81, Cs=0.17 +0.17 — marge fine), **indéterminé à Cs=0.23**, graine 42 sous
1 à Cs fort (0.81 : le fermé y diverge MOINS vite que le chaos). La clause B2-v1 exigeait le structurel
pour TOUTES les valeurs de Cs → **non satisfaite**. EN TITRE : la tendance r(Cs) est MONOTONE DÉCROISSANTE
(2.85 → 2.73 → 2.36 → 2.14) — une closure bête à peine mieux réglée continuerait probablement de raboter.
Germano dynamique et closure apprise restent gatés (même garde anti-treadmill : ne redeviennent vivants
qu'au chiffrage d'une spec fovéa-z).

**P2 — REQUALIFICATION GRAVÉE : le discriminant passe de JUGE à DÉTECTEUR.** B-bis a établi que l'implication
« super-λ ⇒ destruction perceptuelle » est coupée sur tous les axes disponibles (contraste-Weber spatial,
enveloppe). Le discriminant pente-vs-λ reste un fait de dynamique robuste et un instrument bon marché — mais
son statut est désormais TRIPWIRE : il dit OÙ regarder, il ne rend plus de verdict perceptuel. Aucun verdict
futur ne cite « structurel-au-taux » comme équivalent de « perceptuellement cassé ». Cellule kx=1@JND=1 % :
NON achetée en graines — se résout gratuitement quand l'Arc C pinne le JND (≥2 % → moot ; ~1 % → relire les
données existantes). Dépense gravée CONDITIONNELLE.

**P3 — ÉTAGE 1 pré-enregistré (mesure de DIFFÉRENCE d'ensembles, pas de perception ; l'hypothèse rivale est
ouverte : le super-λ pourrait n'être que de la vitesse de décorrélation vers le MÊME ensemble, y compris
temporel — jamais mesuré comme ensemble).** Protocole gravé AVANT lecture :
- Données : T=100 retournements (~1440 trames stationnaires t≥10 ; 240 trames vérifiées trop courtes pour
  les basses fréquences), 4 graines appariées, fin vs naïf vs fermé (Cs=0.17), Ra=1e7, Nz=64.
- Axes (gravés) : (A1) spectre k-ω du readout projeté — par octave-kx porteuse : centroïde ω̄ et largeur σ_ω ;
  (A2) vitesse de phase c\*=ω̄/k̄ par octave [ADAPTATION gravée avec justification : la vitesse d'ascension
  est invisible par construction dans un readout projeté-le-long-de-z ; sa trace visible est la dérive
  latérale] ; (A3) distribution de puissance par octave-ω des séries temporelles de pixels ;
  (A4) temps de décorrélation τ (autocorrélation 1/e) par octave-kx filtrée.
- Statistique (gravée, appariée par graine) : d_s = stat_coarse,s − stat_fine,s ; un axe TIRE ssi
  |mean(d)| > 2·SEM(d) ET signe cohérent sur ≥3/4 graines (garde comparaisons-multiples : 4 axes × 4 octaves
  × 2 coarses = 32 cellules, ~1-2 faux positifs attendus à 2σ sans la clause de signe). Rapporter la carte
  complète, pas seulement les cellules qui tirent.
- Verdicts (gravés, les deux nets) : DIFFÉRENCE TROUVÉE → le descend récupère un job nommé et mesuré ;
  l'axe entre dans l'Arc C (étage 2, JND temporel) qui décide de la perceptibilité. PAS DE DIFFÉRENCE (aux
  barres) → le descend est au chômage à 1e7 PAR MESURE — excellente nouvelle pour le budget (coarse-z +
  readout suffiraient jusqu'à 1e7 inclus, la frontière remonte ou disparaît). **Avertissement bi-directionnel
  gravé** : la traction anti-descend (exécutant) ET la traction pro-budget (l'attracteur de l'autre bord)
  existent toutes deux ; les seuils ci-dessus sont figés, la lecture ne les déplacera dans aucun sens.
- Étage 2 (perception) : GATÉ derrière un étage 1 positif.

---

### 2026-07-01 — ÉTAGE 1 : DIFFÉRENCE TROUVÉE — les ensembles temporels diffèrent ; le descend a un job NOMMÉ ET MESURÉ : la largeur spectrale temporelle des grandes structures. Marge en titre : naïf kx=1 → −36.8 % ± 8.6 %, 4/4 graines (4.3× la barre)

**Cellules qui tirent (règle gravée : |mean|>2·SEM ET ≥3/4 même signe)** — 3 cellules, TOUTES sur le même
axe A1 σ_ω (largeur spectrale en ω par octave-kx), TOUTES du même signe (coarse plus ÉTROIT), sur les
octaves les plus porteuses :
| coarse | octave | d (±2·SEM) | marge |
|---|---|---|---|
| naïf | kx 1-1 | **−36.8 % ± 8.6 %** | 4/4, 4.3× |
| fermé | kx 1-1 | −22.0 % ± 14.5 % | 4/4, 1.5× |
| fermé | kx 2-3 | −15.3 % ± 13.3 % | 4/4, 1.15× |

**Ce n'est PAS le motif des comparaisons multiples** (qui serait : cellules isolées, axes dispersés, marges
minces) : un seul axe, un seul signe, cohérent naïf/fermé, dont une cellule à 4.3× la barre. Signature
physique lisible : **les grandes structures projetées du coarse-z évoluent trop lentement/trop
régulièrement** — moins de résolution-z → moins de disruption fine → les panaches vivent trop longtemps,
scintillent dans une bande trop étroite. Contrôle anti-artefact dans la carte : c\* (A2) à kx=1 naïf =
−2.4 % (inchangé) → le rétrécissement de σ_ω n'est PAS un décalage Doppler d'advection, c'est une vraie
perte de largeur de bande. N'ont PAS tiré : A2 (vitesses de phase), A3 (distribution ω des pixels), A4
(τ — estimateur bruité mais même direction : fermé kx=1 +206 %, cohérent avec σ_ω plus étroit).

**Conséquences (verdicts gravés d'avance, appliqués)** :
- L'hypothèse rivale (« le super-λ n'est que de la décorrélation vers le MÊME ensemble ») est **RÉFUTÉE sur
  l'axe temporel** : les ensembles diffèrent, par mesure appariée multi-graines.
- **Le descend a un job nommé et mesuré** : restaurer la largeur spectrale temporelle des grandes échelles
  projetées (la « respiration » des panaches). Le tripwire (super-λ) pointait au bon endroit — statut
  détecteur confirmé utile.
- La closure bête RÉDUIT le déficit à kx=1 (−36.8 → −22.0) mais l'ÉTALE vers kx 2-3 (−15.3 tire pour le
  fermé ; naïf y était à −10.7, 3/4, sans tirer) — cohérent avec sa signature (sur-dissipation déplacée).
- **Étage 2 (gaté, dû à l'Arc C)** : l'axe « largeur de bande temporelle des grandes structures » entre dans
  le harnais JND comme nouvel axe de stimuli ; le JND temporel décide si −37 % (naïf) / −22 % (fermé) est
  perceptible. AUCUN verdict perceptuel n'est rendu ici : étage 1 est une mesure de différence d'ensembles.
- La question candidate « qu'est-ce que le taux détruit que le contraste-Weber ne voit pas ? » a maintenant
  une réponse mesurée : la cohérence temporelle des structures (le candidat nommé dans l'addendum v2) —
  elle passe de candidate à MESURÉE, en attente de son JND.

---

### 2026-07-01 — ARC V pré-enregistré (gravé avant tout run) : le fin Nz=64 est-il convergé sur σ_ω, l'axe qui rend le verdict de l'étage 1 ?

**Question** : le −36.8 % (déficit σ_ω, coarse32 vs fin64, kx=1) est provisoire tant que le fin de référence
n'est pas convergé sur CET axe. Leçon oracle-diffusif : le mécanisme nommé (moins de disruption fine →
panaches trop persistants) s'applique au fin sous-résolu comme au coarse.

**Protocole (v3, transcrit)** : σ_ω par octave, fin Nz=128 vs fin Nz=64, 4 graines (7, 11, 23, 42),
T=100 retournements, fenêtre stationnaire t≥10, bornes d'octaves gravées inchangées, critère apparié
inchangé (|mean(d)| > 2·SEM ET ≥3/4 même signe). **Choix d'exécution gravés** : appariement par
θ0_64 = zdown(θ0_128) (même grande échelle d'IC, analogue exact de la construction coarse) ; les séries
L(x,t) de TOUS les runs sont ARCHIVÉES en .npz (correction de dette : deux protocoles de suite ont supposé
des archives inexistantes). Ra=1e7, κ=4, mêmes solveur/rendu. Aucun autre run dû dans cette session.

**Claims (v3, transcrits)** :
- **V1 (fin convergé)** : d(σ_ω) fin128-vs-fin64 sub-critère sur kx=1 ET kx=2-3 → le −36.8 % est gravé
  DÉFINITIF comme amplitude du déficit.
- **V2 (fin non convergé)** : une cellule tire → le chiffre-titre est REMPLACÉ par le déficit re-mesuré à
  la référence Nz=128. Note de structure gravée : « coarse » ≡ run à demi-résolution-z dans cet instrument,
  donc la cellule fin64-vs-fin128 EST le déficit ratio-2 à la référence 128 — le remplacement ne coûte
  aucun run. Sens attendu du mécanisme : « vrai déficit ≥ −36.8 % » — on grave ce que la mesure dit,
  pas le mécanisme.
- Dans les deux cas : l'étage 2 (C-temporel, session parallèle) lit son point final à l'amplitude V-validée.
**Dépenses refusées (v3)** : σ_ω@1e6 (conditionnel, gaté) ; tout autre run dans cette session.

---

### 2026-07-01 — ARC V rendu : V2 — fin Nz=64 NON convergé sur σ_ω ; chiffre-titre REMPLACÉ : le déficit ratio-2 à la référence 128 est −19.1 % ± 12.6 % (kx=1, 4/4, marge fine 1.5×) — et il RÉTRÉCIT quand la référence double, à l'INVERSE du sens attendu du mécanisme

**Mesure (protocole gravé, appariement zdown, archives .npz complètes)** : d(σ_ω) fin64-vs-fin128 :
kx=1 : **−19.1 % ± 12.6 %, 4/4 → TIRE** ; kx 2-3 : +5.3 % ± 16.5 % (2/4, sub-critère) ; kx 4-7 et 8-15
sub-critère. → **V2** : le fin Nz=64 n'était pas convergé sur l'axe qui rend le verdict de l'étage 1.

**Application du claim gravé** : le chiffre-titre −36.8 % (coarse32 vs fin64) est REMPLACÉ par
**−19.1 % ± 12.6 %** — la cellule fin64-vs-fin128, qui EST le déficit ratio-2 à la référence 128 (note de
structure gravée d'avance). Point de synchronisation pour l'étage 2 (C-temporel) : **la lecture finale
Ct1/Ct2 se fait à −19.1 % sur la psychométrique** (dans la plage balayée [−10 %, −60 %]).

**Le chiffre inconfortable, en titre comme dû — dans les DEUX sens** :
1. Le sens attendu du mécanisme était « vrai déficit ≥ −36.8 % ». La mesure dit l'inverse : le déficit
   ratio-2 RÉTRÉCIT quand la résolution absolue double (−36.8 % à 32/64 → −19.1 % à 64/128). Gravé tel quel.
2. Conséquence ouverte, non tranchable dans cette session : la référence 128 n'est PAS montrée convergée
   (le déficit non nul à 64-vs-128 le prouve par récurrence). La suite (−36.8, −19.1, …) tend vers des
   déficits plus petits à résolution croissante — l'hypothèse « le déficit σ_ω est en partie un artefact de
   sous-résolution ABSOLUE (les deux runs de l'étage 1 étaient sous-résolus) » est désormais VIVANTE. Elle
   affaiblirait le job du descend si la suite converge vers ~0 ; elle le préserve si elle converge vers un
   plancher non nul. Un point 128-vs-256 la trancherait — NON DU (aucun autre run dans cette session, gravé),
   et c'est une dépense à arbitrer par Romain AVANT que l'étage 2 ne lise son point final : lire la
   psychométrique à −19.1 % n'a de sens que si −19.1 % est tenu pour l'amplitude réelle, ce que la
   non-convergence de la référence rend incertain dans le sens favorable-au-budget.
3. Marge fine : 1.5× la barre (−19.1 ± 12.6). Sous G-4, à moins d'un écart-type du critère → sensibilité :
   la cellule tient à 2·SEM et 4/4 signes, mais une 5e graine pourrait la faire basculer — rapporté, pas caché.

**G-5 vérifié** : V2 ne retire pas la justification de l'étage 2 (le balayage paramétrique [−10, −60] a été
conçu pour couvrir tout déplacement du chiffre par V — les sessions restent parallélisables). Il DÉPLACE le
point de lecture et ajoute la question de convergence ci-dessus à l'arbitrage. Session Boussinesq : plus
rien de dû ; archives en place pour toute relecture.

---

### 2026-07-02 — ARC W0 pré-enregistré (gravé avant l'analyse) : pouvoir statistique du point 128-vs-256

**Question W (v4, transcrite)** : D(Nz) = déficit ratio-2 à la résolution d'opération — loi de puissance
(le déficit s'éteint : descend = arbitrage de coût) ou plancher non nul (déficit irréductible : descend
obligatoire au-dessus du JND temporel) ? Deux points existent (−36.8 % @32/64, −19.1 % @64/128) ;
ratio 0.52 compatible D ∝ Nz^−p, p≈0.95 — COMPATIBLE, PAS FITTÉ. W1 fournirait le 3e point.

**Prédictions jumelles (v4, gravées avant tout chiffre)** : W-loi → D(256) ≈ −10 % ; W-plancher →
D(256) ≈ −19 %. Pouvoir requis pour les distinguer : **2·SEM ≤ 4.5 %** sur la cellule 128-vs-256.

**W0 (zéro run)** : (a) bruit par graine recalculé depuis les archives .npz (cellule 64/128 exacte,
par graine) + le 8.6 % (2·SEM) du journal pour 32/64 ; (b) n\* = graines nécessaires (SEM ∝ 1/√n) ;
(c) coût RÉEL chiffré par benchmark de timing (200 pas extrapolés — solveur numpy CPU, pas GPU : on
chiffre l'instrument tel qu'il tourne) + vérification mémoire (inv_p à Nz=256 ≈ 34 Mo ×2, OK a priori).
**Gate d'achat (v4)** : n\* ≲ une nuit → W1 en fond ; sinon « 128-vs-256 non discriminable au bruit
actuel » gravé — un run qui ne peut pas trancher fabrique de la confiance, options remontées à Romain.
Note de neutralité : T plus long réduit σ² ∝ 1/T mais coûte ∝ T — le produit n\*·coût est ~invariant ;
si le gate échoue en graines, il échoue aussi en T (à consigner si c'est le cas, pas d'échappatoire cachée).

---

### 2026-07-02 — W0 rendu : « 128-vs-256 NON discriminable au bruit actuel » — gate d'achat FERMÉ (19.4 h au bruit best-estimate vs ~10 h) ; W1 non lancé. En titre aussi : la graine 7 donne d ≈ −2.4 % — la cellule 64/128 contient une graine quasi nulle

**(a) Bruit par graine (archives, cellule 64/128 exacte)** : d = −2.4 % (graine 7), −20.6 %, −20.4 %,
−33.1 % → σ par graine = 12.6 %. Le chiffre inconfortable en titre : UNE graine sur 4 est quasi NULLE —
la fragilité anticipée sous G-4 à l'Arc V est confirmée au niveau des données brutes (le −19.1 % ± 12.6 %
tient au critère, mais son support inter-graines est hétérogène). Cellule 32/64 (journal) : σ = 8.6 % ;
le bruit CROÎT avec la résolution (8.6 → 12.6), donc l'estimation pour 128/256 par la cellule voisine
(12.6 %) est la best-estimate, et probablement encore optimiste.

**(b) n\*** : 32 graines au bruit best-estimate (12.6 %) ; 15 au bruit optimiste (8.6 %, contredit par la
tendance). **(c) Coût réel (benchmark 200 pas, numpy CPU)** : Nz=256 = 32.6 min/run, Nz=128 = 3.8 min/run
→ paire 36.3 min → total 19.4 h (n\*=32) à 9.1 h (n\*=15, optimiste).

**Verdict du gate (règle gravée « ≲ une nuit ~10 h »)** : **INFAISABLE au bruit best-estimate** —
9.1 h n'est atteignable que sous l'hypothèse de bruit que la tendance mesurée contredit. Gravé :
« 128-vs-256 non discriminable au bruit actuel ». W1 NON lancé — un run qui ne peut pas trancher entre
−10 % et −19 % fabriquerait de la confiance. Note d'invariance (gravée d'avance, vérifiée applicable) :
T plus long n'échappe pas (σ² ∝ 1/T, coût ∝ T, produit invariant).

**Options remontées à Romain (arbitrage, aucune n'est exécutée ici)** : (1) réduction de variance par
appariement plus serré — piste concrète issue des données : la dispersion vient de graines entières
(−2 vs −33), pas du bruit de fenêtre ; un appariement par ENSEMBLE de conditions initiales par graine
(moyenner d sur k sous-fenêtres temporelles longues par graine) pourrait réduire σ sans nouveau run de
mesure — à chiffrer sur archives AVANT toute décision ; (2) accepter 2 nuits (19.4 h) si la décision de
spec le vaut ; (3) porter l'incertitude D(Nz_op) ∈ [loi, plancher] dans la spec (issue INDÉTERMINÉ-loi
assumée sans le 3e point). Statut session Boussinesq : PLUS RIEN DE DÛ — W0 clôt la file v4 côté Boussinesq.

---

## Addendum Arc A — Manche 1 (fermeture état-complet) : §A0–§A5 (gravé le 2026-07-03, AVANT toute mesure)

### §A0 — Arc A : structure gravée (deux manches, trois cellules)

**Hiérarchie des claims (gravée avant mesure) :**
- **Registre-commis = claim EXISTENTIEL.** C'est lui que le but exige : non-contradiction du
  témoignage émis, ledger borné par le budget d'observation — pas par le volume du monde.
- **État-complet = SÉLECTEUR d'architecture.** Il départage invariants-dans-`z` vs ledger.
  Son FAIL ne tue rien.

**Emboîtement logique :** état-complet PASS ⇒ registre-commis PASS (un résumé qui régénère
déterministiquement tout le champ persistant reproduit a fortiori toute observation émise,
revisites comprises). Contraposée : commis FAIL ⇒ état-complet FAIL.

**Grille des issues, pré-interprétée (aucune relecture a posteriori) :**

| Cellule | Issue | Conséquence gravée |
|---|---|---|
| 1 | état-complet PASS | Fermeture dans `z` : architecture invariants-régénérables. Manche 2 court-circuitée par implication — **sur CE substrat uniquement**. |
| 2 | état-complet FAIL, commis PASS | Mur mémoire sur l'état, pas sur le témoignage. Architecture **ledger obligatoire** ; le différenciateur du projet survit. |
| 3 | commis FAIL | Thèse morte sous ses deux formes. Verdict existentiel négatif. |

**Ordre d'exécution :** manche 1 = état-complet (instrumentation existante, le moins cher).
Manche 2 = registre-commis, construite **seulement si** manche 1 FAIL ou INDÉTERMINÉ-porté.

**Pré-étiquetage (bloquant) : un FAIL de la manche 1 est NON-EXISTENTIEL.** Il sélectionne
la cellule 2-ou-3 à départager ; il ne tue rien. Personne ne le relira autrement.

### §A1 — Claim manche 1 (figé)

Il existe un résumé **grossier**, **déterministe-régénérable**, du champ persistant
(sédiment) tel que :
- **(i)** la régénération reproduit le **readout** (turbidité/albédo, contraste Weber Δχ)
  **sous JND**, instantanément **et** après rollout avant — fermeture **sous la dynamique**,
  pas fermeture de texture ;
- **(ii)** la taille minimale du résumé **k\*(L) sature** avec la longueur d'historique L.

### §A2 — Opérationnalisation (figée)

- **Substrat** : le substrat sédiment/shallow-water validé (historique path-dependent mesuré
  ~61 %, canal readout survivant = turbidité/albédo, resfrac 0.60–0.66). **Réutiliser tel
  quel. Ne pas re-valider le générateur. Ne pas re-régler la sédimentation.**
- **Histoires** : L ∈ {L₀, 2L₀, 4L₀, 8L₀} épisodes, où L₀ = longueur d'un arc épisodique déjà
  en usage dans le substrat (fixé en Task 0 depuis l'existant, PUIS gravé au journal avant le
  premier run). **≥ 5 seeds par L.**
- **Compresseur (famille figée)** : coarse-graining Harten du champ persistant au niveau
  ℓ ∈ {1, 2, 3} + invariants scalaires conservés (masse déposée totale, et par sous-domaine
  4×4). Taille du résumé = la variable balayée.
- **Régénérateur (figé)** : opérateur déterministe conditionné **uniquement** sur le résumé.
  Toute stochasticité est seedée par **hash(résumé)** — aucune seed libre, aucun état caché,
  aucune vue du champ vrai ni de l'historique.
- **Mesures** :
  - **M-A1 (instantané)** : Δχ(readout(régénéré), readout(vrai)) vs JND.
  - **M-A2 (dynamique, bloquant)** : rollout T_fwd = 1 épisode depuis le champ régénéré vs
    depuis le champ vrai, **même forçage, même seed de forçage** ; Δχ des readouts le long du
    rollout vs JND. Sans M-A2, un PASS est une texture, pas un état.
  - **M-A3 (déterminisme, bloquant)** : double régénération → identité (tolérance relative
    1e-12). **La fidélité distributionnelle est interdite comme voie de succès.**
- **k\*(L)** = plus petit résumé tel que M-A1 ∧ M-A2 sous JND pour la **médiane des seeds**,
  dispersion inter-seeds rapportée, **aucune seed > 2×JND**.

### §A3 — Verdicts (figés)

- **PASS (fermeture)** : pente de k\*(L) sur la moitié supérieure de la plage de L compatible
  avec 0 (IC 95 % inter-seeds contient 0), **ET** k\*(L_max) ≤ **10 % de la taille du champ
  persistant fin** (plafond anti-trivialité — sinon « stocker tout » passe ; placeholder
  nommé, à ne pas défendre).
- **FAIL (mur)** : pente positive > bruit inter-seeds sur toute la plage, sans plateau.
- **INDÉTERMINÉ** : IC chevauchant, ou verdict basculant sur la plage JND. C'est un
  **résultat**. Options pré-écrites : (a) étendre à 16L₀ **une seule fois** ; (b) porter
  l'incertitude et ouvrir la manche 2.
- **Analyse de sensibilité JND OBLIGATOIRE** (geste T1.5 gravé) : k\*(L) recalculé sur la
  plage plausible du JND placeholder ; verdict retenu seulement s'il est stable sur toute la
  plage, sinon INDÉTERMINÉ.

### §A4 — Gardes anti-PASS-fabriqué (bloquantes)

1. Métrique en **espace readout uniquement** — jamais L2 sur l'état (discipline établie ;
   le 18.6 % → 0 % est au dossier).
2. **M-A2 bloquant** : fermeture sous la dynamique, pas de PASS-texture.
3. **M-A3 bloquant** : déterminisme structurel, seed = hash(résumé).
4. **Plafond anti-trivialité** sur k\* (§A3).
5. **Anti-fuite** : le régénérateur ne reçoit RIEN d'autre que le résumé — imposé par
   signature et vérifié par test.

### §A5 — Ce que la manche 1 ne prouve PAS (§13 local)

- **Un PASS** ne dit rien : des autres écritures persistantes (fracture, dégâts,
  empreintes) ; de la 3D ; de l'échelle au-delà du domaine testé ; du routage/fovéa ; du
  registre commis au-delà de l'implication logique — qui ne vaut que sur CE substrat.
- **Un FAIL** est celui de **cette famille de compresseurs** (Harten-coarse + invariants
  scalaires), pas de toute fermeture possible — doublement non-existentiel (cf. §A0).
- **Le JND reste un placeholder** : tout verdict est conditionné à la sensibilité §A3.

---

### 2026-07-03 — Task 0 Arc A (reconnaissance) : le substrat sédiment N'EXISTE PLUS sous forme de code ; arbitrage = reconstruction DURABLE dans pocPhysicator + amendement re-validation

**Reconnaissance (négative, vérifiée à fond)** : le code du substrat sédiment validé (loi de dépôt
path-dependent, forçage pulses, `b_eff=b0+s`, readout albedo `1−exp(−s/S_HALF)`, instruments ΔR/f(k))
n'existe **nulle part sur cette machine** — vérifié dans pocCascade2phys ET pocPhysicator (worktree,
toutes branches, tout l'historique git), dans tous les transcripts de session, et dans /tmp. Seuls les
verdicts (entrées 2026-06-30 → 2026-07-01 ci-dessus) survivent. Cause : le code vivait en scratchpad
/tmp ; **reboot machine 2026-07-03 19:49 → /tmp purgé**. Dommage collatéral consigné : les **archives
Arc V** (`v_fin128_s*.npz`, `v_fin64zd_s*.npz`) sont perdues — l'option (1) de W0 (« chiffrable
gratuitement sur les archives ») exige désormais de régénérer les 8 runs (~30–60 min CPU).

**Ce qui existe et se réutilise** : `pocPhysicator` (main, v2.5) = brique shallow-water 2D 64×64
validée — solveur MUSCL 2ᵉ ordre well-balanced positivity-preserving mouillé/sec
(`src/solver_wetdry.py::simulate_wetdry_o2`, oracles Thacker/Ritter/Stoker), terrains paramétrés
(`src/terrains.py`), rendu heatmap (`src/render.py`). **Manquent** (à reconstruire depuis les specs
gravées ci-dessus) : couche sédiment, forçage épisodique, readout albedo, instrument Δχ Weber.

**Arbitrage Romain (2026-07-03)** : option 2 — **reconstruire le substrat, DURABLEMENT** : code commité
(git), plus aucun artefact load-bearing en /tmp (leçon du reboot).

**AMENDEMENT §A2 (gravé AVANT le build, conséquence de l'arbitrage)** : la clause « Réutiliser tel
quel. Ne pas re-valider le générateur. » est insatisfiable (l'objet n'existe plus). Elle est remplacée
par : **le substrat reconstruit DOIT être re-validé contre les critères DÉJÀ gravés au journal, au même
point d'opération, SANS re-réglage** — paramètres fixés avant les runs de re-validation :
  (a) critère path-dependent (2 runs, ordre de pulses inversé → dépôts différents ; référence :
      corr(s_A,s_B)≈0.39, resfrac ~67 %) ;
  (b) survie du canal albedo : f_albedo ≫ f_relief, robuste sur S_HALF ∈ [0.005, 0.20]
      (référence : 3.5–10 % vs 1.2 %) ;
  (c) f(k) monotone décroissante aux deux géométries (overlap et séparée).
**Un critère manqué = STOP et remonter** (le substrat reconstruit n'est pas celui qui a été validé) —
pas de re-réglage pour le faire passer. L'esprit anti-tuning de la clause d'origine est conservé ;
seule la lettre (impossible) est amendée.

**Décision de dépôts** : code + tests + `outputs/arcA/` dans **pocPhysicator** (branche dédiée) ;
contrat + journal restent ICI. Chaque entrée de journal Arc A citera les commits pocPhysicator.

**Proposé au point d'arrêt Task 0 (EN ATTENTE de validation Romain — non gravé comme fixé)** :
- **L₀ = 10 pulses** — l'arc épisodique déjà en usage au journal (f(k) mesuré à T=10 pulses ;
  profondeur de mémoire mesurée ~3–7 pulses → la saturation de k\*(L) est discriminable dès 2L₀).
- **Plage JND (albedo) = [2 %, 5 %]** — les deux valeurs en usage au journal pour f. L'instrument
  Δχ Weber de l'addendum (χ = RMS_bande/⟨L⟩, cf. R1) est à construire sur le canal albedo ; la plage
  JND s'applique à Δχ.

### 2026-07-03 — Point d'arrêt Task 0 LEVÉ (validation Romain) : L₀ et plage JND FIXÉS ; specs d'instrument et de substrat gravées AVANT le build

**FIXÉ (validé par Romain au point d'arrêt)** : **L₀ = 10 pulses** → L ∈ {10, 20, 40, 80}, ≥ 5 seeds
par L. **Plage JND = [2 %, 5 %]** (sensibilité §A3 balayée sur cette plage).

**Instrument Δχ-albedo (gravé avant toute mesure)** — deux mesures, deux usages, pas d'échange :
- **f (re-validation uniquement, réplique du journal)** : fraction du domaine où |ΔA|/⟨A⟩ > JND,
  A = albedo. C'est la définition d'origine des entrées 2026-06-30/07-01 ; elle sert aux critères (a)/(b)/(c).
- **Δχ par bande (M-A1/M-A2, §A1)** : spectre spatial radial de A, octaves k ∈ {1, 2-3, 4-7, 8-15,
  16-31} cycles/domaine ; χ_b = RMS(bande)/⟨A⟩ ; bandes PORTEUSES = ≥ 10 % de la puissance AC (règle
  R1) ; Δχ_b = |χ_b(régénéré) − χ_b(vrai)| ; **critère = max sur les bandes porteuses vs JND**.
  Aucun verdict sur bande non-porteuse (leçon R1).

**Specs de reconstruction du substrat (gravées avant le build, amendement 2026-07-03)** :
- **Épisode** = pulse d'eau (monticule gaussien, centre seedé, amplitude fixe) sur terrain sec fixe
  (asset statique connu, même terrain pour TOUS les runs) → relaxation wet/dry (`simulate_wetdry_o2`,
  N_settle pas fixés) → assèchement (h remis à sec, `s` persiste). Le forçage d'un run = la séquence
  seedée des centres de pulses.
- **Loi de dépôt (type Exner, gravée)** : proxy de cisaillement θ = u²+v² sur cellules mouillées ;
  ds/dt = k_d·h·1[θ<θ_c]·(1−θ/θ_c) − k_e·s·1[θ>θ_c]·(θ/θ_c−1) ; rétroaction b_eff = b0 + s au
  solveur À CHAQUE épisode (le pulse suivant voit le dépôt).
- **Point d'opération (gravé au journal d'origine, repris tel quel)** : taux fort-plausible —
  s_max ≈ 20 % du relief à L₀ ; S_HALF = 0.05 au point d'op, robustesse balayée sur [0.005, 0.20].
- **Calibration UNIQUE autorisée** : k_d est fixé par UNE recherche préalable pour atteindre le point
  d'op gravé (≈20 % relief à L₀), PUIS gelé AVANT les runs de re-validation (a)/(b)/(c). Toute autre
  retouche de k_d/k_e/θ_c après la première mesure de re-validation = interdite (un critère manqué
  = STOP et remonter).
- **Référence « simultané même-masse »** pour f : run tous-pulses-à-la-fois, s final rescalé à la
  masse totale du run ordonné.

Exécution : code + tests + outputs dans pocPhysicator, branche `arc-a-etat-complet` ; chaque étape
citée ici avec ses commits.

### 2026-07-04 — GATE de re-validation Arc A : FAIL 2/3 — le substrat reconstruit N'EST PAS le substrat validé (corr(s_A,s_B) = 0.9929 vs seuil ≤ 0.7, référence d'origine 0.39) ; manche 1 STOPPÉE avant toute mesure

**Mesures (seuils figés au plan AVANT exécution, application mécanique, n = 10 épisodes/histoire)** :
- **(a) path-dependence : FAIL.** corr(s_A, s_B) = **0.9929** (seuil ≤ 0.7 ; l'original mesurait 0.39,
  soit « 61 % du dépôt vient de l'histoire »). Le dépôt reconstruit est quasi insensible à l'ordre des
  pulses — le trait que l'Arc A doit mesurer (fermabilité d'une HISTOIRE) est quasi absent du substrat
  reconstruit.
- **(b) survie albedo : PASS — mais différemment.** f_albedo = 0.81–0.85 sur TOUTE la plage S_HALF vs
  f_relief = 0.0020 (seuils tenus, hiérarchie robuste). L'amplitude (83 %) est pourtant ≫ la référence
  (3.5–10 %) : (b) compare séquentiel vs simultané (dynamiques très différentes), pas l'ordre — il peut
  passer fort pendant que (a) meurt.
- **(c) closure f(k) : FAIL (géométrie séparée).** Overlap : 0.338 → 0.214 → 0.187 → 0 (décroissante,
  PASS). Séparée : 0.327 → **0.363** (rebond à k=4) → 0.112 → 0 : non-monotone, FAIL.

**Lecture (hypothèse nommée, NON mesurée)** : le dépôt reconstruit ressemble au « défaut latent »
que le contrat du 2026-06-30 avait nommé — un commit largement terrain-déterminé / quasi-commutatif
(issue 1 déguisée). Suspects : les paramètres a priori de la loi (θ_c = 0.5, k_e = k_d/5 — choisis
par le contrôleur, les valeurs d'origine sont perdues avec le code) et la structure d'épisode
(assèchement complet entre pulses). Diagnostic cheap possible : part d'érosion active (fraction de
cellules·pas où θ > θ_c) — si ≈ 0, la loi est dépôt-seul donc quasi-commutative par construction.

**Conséquence gravée appliquée** : STOP — Tasks 3–6 non lancées, aucun paramètre retouché, aucun
seuil déplacé. Le FAIL est un résultat : la reconstruction « d'après specs de journal » ne suffit
PAS à reproduire le substrat (les verdicts gravés 2026-06-30/07-01 restent valides pour le substrat
d'ORIGINE ; ils ne se transfèrent pas au reconstruit).

**Statut épistémique du gate** : gate d'INSTRUMENT (type G0) — itérer l'instrument jusqu'à validité
est légitime (précédent : chirurgie IB/LBM de T1), à CRITÈRES INCHANGÉS et par changements
physiquement motivés et pré-annoncés ; ce que la règle interdit est le re-réglage silencieux par
l'exécutant. Le fork est remonté à Romain : (1) diagnostic cheap puis UNE itération de conception
nommée, pré-enregistrée, re-validée aux mêmes seuils ; (2) re-pré-enregistrer Arc A sur le substrat
reconstruit tel quel — DÉCONSEILLÉ : corr = 0.993 → quasi pas d'histoire à fermer → k\*(L) saturerait
trivialement = PASS fabriqué ; (3) suspendre Arc A côté état-complet.

**Note d'instrument (complétude)** : dérive de masse d'eau 3–13 %/épisode au mur mouillé/sec —
propriété du solveur figé (padding antisymétrique du moment normal → pente de vitesse asymétrique →
flux HLL non nul au mur), documentée et bornée en test ; aucun lien établi avec le FAIL (a).

**Commits pocPhysicator (branche `arc-a-etat-complet`)** : e36dd23..be88024 (Task 1 : substrat +
instruments, 102 tests verts, KD_CALIBRE = 1.911e-3 gelée, max(s)/relief = 0.206) ; 554c2a9 (gate,
données brutes `outputs/arcA/revalidation.json`).

### 2026-07-04 — Diagnostic du FAIL (a) : l'hypothèse annoncée est MORTE ; la signature manquée est resfrac (0.17 mesuré vs 0.60–0.66 gravé) → itération de conception nommée, pré-enregistrée AVANT implémentation

**Arbitrage Romain** : option (1) — diagnostic cheap, puis UNE itération de conception nommée,
re-validée aux MÊMES seuils. Diagnostic exécuté (pocPhysicator `0619c17`, `outputs/arcA/diag_pathdep.json`).

**Résultats — l'hypothèse « dépôt-seul dominé par la phase calme » est contredite sur 3 mesures /4** :
(1) érosion ACTIVE : 75.3 % des (snapshot, cellule mouillée) en θ > θ_c — pas de régime dépôt-seul ;
(2) 99.3 % du dépôt tombe en snapshots ACTIFS (0.7 % en phase calme) — pas de domination stagnante ;
(4) contrefactuel k_e = 0 : corr(direct, inversé) passe de 0.9929 à 0.9973 — couper l'érosion AUGMENTE
la commutativité de 0.004 seulement : le ratio k_e/k_d n'est pas la cause du FAIL. Seule (3) est
partiellement à charge : corr(s_final, résidence d'eau cumulée) = 0.83 — terrain-déterminisme
substantiel mais pas total. On grave la mort de l'hypothèse annoncée, comme d'habitude.

**Ce que le diagnostic révèle en creux (lecture, base de l'itération)** : la carte d'identité gravée
du substrat d'origine (§A2 : « path-dependent ~61 %, **resfrac 0.60–0.66** ») a DEUX signatures. La
reconstruction n'en a calibré qu'UNE (taux : max(s)/relief ≈ 0.2, tenu à 0.206) ; la seconde —
resfrac, ici définie masse totale érodée / masse totale déposée sur l'histoire — sort à **0.174**,
soit ~4× sous la plage gravée. Le substrat d'origine était en régime de RETRAVAIL fort (deux tiers du
déposé re-repris) ; le reconstruit retravaille à peine — et un dépôt peu retravaillé pèse chaque pulse
presque indépendamment → quasi-commutatif. k_e = k_d/5 était un a priori du contrôleur, pas une
donnée d'origine.

**ITÉRATION NOMMÉE (pré-enregistrée ici, AVANT implémentation)** :
- **Changement unique** : remplacer l'a priori k_e = k_d/5 par une **co-calibration aux DEUX
  signatures gravées** — bissections alternées : k_d → max(s)/relief ∈ [0.18, 0.22] à k_e fixé ;
  k_e → resfrac ∈ [0.60, 0.66] à k_d fixé ; ≤ 4 cycles, seed 12345, 10 épisodes (même protocole de
  calibration que l'origine, une exécution, constantes re-gelées AVANT re-validation). θ_c = 0.5 et
  toute la structure d'épisode INCHANGÉS.
- **Prédiction directionnelle (falsifiable)** : le retravail fort re-pondère le dépôt vers les pulses
  récents par région → corr(direct, inversé) doit BAISSER nettement. On grave la mesure, pas le
  mécanisme.
- **Re-validation aux MÊMES seuils (a)/(b)/(c), même script, aucun seuil déplacé.** Issues gravées :
  3/3 PASS → substrat re-validé, manche 1 reprend (Tasks 3–6). Un critère manqué → **STOP définitif
  de la voie « reconstruction »** : l'itération autorisée a été consommée, on remonte (le fork suivant
  appartient à Romain). Co-calibration infaisable (cibles jointes inatteignables) → BLOCKED, remonter
  sans forcer.

### 2026-07-04 — Gate v2 : FAIL 2/3 — les DEUX signatures gravées sont tenues et la path-dependence d'origine ne revient PAS (corr = 0.9446 vs ≤ 0.7 ; origine 0.39). STOP définitif de la voie « reconstruction » : le substrat d'origine est IRREPRODUCTIBLE depuis son dossier de journal

**Co-calibration (une exécution, seed 12345, convergence au 1er cycle, constantes gelées avant
re-validation)** : KD_CALIBRE_V2 = 1.911e-3 (bit-à-bit = V1), KE_CALIBRE_V2 = 5.233e-3 (k_e/k_d
passe de 0.2 à 2.74). Signatures vérifiées aux constantes gelées : max(s)/relief = **0.18876**
∈ [0.18, 0.22] ET resfrac = **0.64254** ∈ [0.60, 0.66]. 105 tests verts.

**Verdict v2 (mêmes seuils, application mécanique)** :
- **(a) FAIL** : corr(direct, inversé) = **0.9446**. La prédiction directionnelle pré-enregistrée est
  confirmée en SENS (0.9929 → 0.9446) mais pas en amplitude : **resfrac dans la plage gravée est
  nécessaire mais PAS suffisant** pour la path-dependence d'origine. On grave la mesure, pas le
  mécanisme.
- **(b) PASS** : f_albedo_op = 0.8511, f_relief = 0.0020, dominance sur toute la plage S_HALF.
- **(c) FAIL** (séparée) : 0.602 → 0.304 → **0.352** (rebond à k=7 ; v1 : rebond à k=4) → 0 ;
  overlap PASS (0.441 → 0.312 → 0.281 → 0).

**Portée EXACTE (ne pas surclamer, dans les deux sens)** :
- Les verdicts d'origine (2026-06-30 → 07-01 : hors-clé albedo, pas de mur f(k)) restent valides
  POUR LE SUBSTRAT D'ORIGINE. Rien ici ne les infirme.
- Ce qui est établi de neuf : **le dossier de journal du substrat (loi nommée + point d'op + les deux
  signatures) ne suffit PAS à le reproduire** — deux reconstructions conformes aux specs gravées
  donnent corr 0.99 puis 0.94 là où l'origine donnait 0.39. L'information qui portait les ~61 %
  d'histoire vivait dans des choix d'implémentation non consignés (forme exacte de la loi, structure
  de forçage/BC, transport éventuel) et est perdue avec le code. Leçon d'instrument consignée :
  les artefacts load-bearing se committent (déjà appliqué : tout ce travail est commité).
- Arc A manche 1 : **suspendue en l'état** — pas de mesure k\*(L) sur un substrat quasi sans histoire
  (corr 0.94 → PASS de fermeture trivial garanti = fabriqué).

**Fork remonté à Romain (la voie « reconstruction » est close ; options nouvelles)** :
(1) **re-fonder** : bâtir un substrat path-dependent PAR CONSTRUCTION (transport en suspension :
érosion → champ suspendu advecté → re-dépôt aval ; physiquement le porteur d'ordre le plus plausible,
et la brique du readout « turbidité » déjà nommé au §A1) — nouveau build pré-enregistré avec les
critères (a)/(b)/(c) requalifiés de gate d'identité en gate de CONCEPTION (mêmes seuils, itérable
sous discipline d'instrument normale, type G0) ; Arc A garde ses claims §A1–§A5 inchangés ;
(2) mesurer d'abord, pour ~2 min, la borne d'histoire-readout du v2 (f_albedo entre les deux ordres)
avant de décider — si même le readout ne voit rien, (1) ou (3) ; (3) suspendre l'état-complet et
réallouer.

**Commits pocPhysicator** : 0619c17 (diagnostic), 486d637 (co-calibration V2), ebdf32c (gate v2 +
outputs), 6512f14 (ledger).

---

## Addendum Arc A — Re-fondation du substrat (§A6–§A11)

> **Statut : gravé le 2026-07-04, AVANT toute mesure sur l'objet neuf.** Suite du fork du
> 2026-07-04 (gate v2 FAIL 2/3, voie « reconstruction » close). Arbitrage Romain : option (2)
> puis option (1), sous cinq conditions de durcissement + BC périodiques (motivé physiquement)
> + note de but « obligations étagées ». **Les claims §A1–§A5 sont INCHANGÉS.**

### §A6 — Note de niveau BUT : obligations étagées (gravée, pour lecture des verdicts)

L'exigence du jeu (re-dérivée en session depuis le game design, convergente avec les trois
routes analytiques déjà actées) est le **registre-commis à obligations étagées** :
1. **Invariants durs, observés ou non** : conservation (la masse sédimentaire lâchée est
   quelque part en aval, jamais évaporée ni remontée) ; les événements commis restent commis.
   Tout ré-échantillonnage vit dans la classe d'équivalence compatible avec le registre causal.
2. **Émissions** : ce qui a atteint un écran reste re-dérivable **sous JND** (captures,
   revisites conjointes, comparaisons entre joueurs).
3. **Le reste : libre** — mais libre = **fonction déterministe d'un registre avec perte**
   (seed = hash(registre)), JAMAIS bruit de ré-échantillonnage. La liberté vit dans ce que le
   registre oublie, pas dans l'aléa au retour.

**Pré-étiquetage de lecture (bloquant)** : un FAIL de la manche 1 n'est pas une crise (cellule
2 = le filet ; le différenciateur survit). Un PASS de la manche 1 ne clôt pas la question
existentielle : l'implication cellule-1 vaut sur CE substrat uniquement (§A5) ; le claim
registre-commis à l'échelle du but (autres écritures, densité d'observation
croissante/adversariale — garde anti-vacuité de la manche 2) reste ouvert.

### §A7 — Re-dérivation de l'ordre des manches (la prémisse d'origine est morte)

La justification gravée en §A0 (« instrumentation existante, le moins cher ») est morte avec
le substrat. Re-dérivation explicite, l'ordre est **MAINTENU** :
1. Le build manche 1 (compresseur + régénérateur sur champ persistant) reste matériellement
   plus petit que la machinerie manche 2 (définition d'événements, ledger, reconstruction
   conditionnée, balayage de densité d'observation).
2. Le court-circuit logique tient : PASS manche 1 ⇒ manche 2 close par implication sur ce
   substrat → l'espérance de coût favorise toujours manche 1 d'abord.
3. Le substrat re-fondé (suspension/turbidité) sert les **deux** manches — un basculement en
   manche 2 ne perd rien de la construction.

### §A8 — M-0 : mesure préliminaire sur v2 (la « 2 minutes », gravée avant exécution)

**Objet** : borne d'histoire-readout du substrat v2 gelé (KD_CALIBRE, KE_CALIBRE_V2) :
`f_ordre(v2)` = fraction du domaine où |ΔA(direct, inversé)|/⟨A⟩ > JND, JND balayé sur
[2 %, 5 %], albedo A au point d'op (S_HALF = 0.05, sensibilité [0.005, 0.20] rapportée).

**Double usage gravé** : (i) confirmer/infirmer que v2 est mort jusqu'au readout inclus ;
(ii) premier point de calibration de la fonction de transfert état→readout
(corr = 0.9446 ↔ f_ordre) et **baseline du bras nul §A11**.

**Lectures pré-écrites** :
- `f_ordre(v2)` ≤ 1 % du domaine à JND = 5 % → v2 confirmé mort au readout → bras nul propre,
  build §A10 lancé.
- `f_ordre(v2)` > 1 % à JND = 5 % → le readout AMPLIFIE une histoire que corr sous-estime →
  **STOP, remonter** : la hiérarchie corr↔readout est inversée, le gate §A9 doit être repensé
  avant tout build.

### §A9 — Gate de CONCEPTION du substrat re-fondé (critères figés avant build)

**Requalification actée** : gate d'identité → gate de conception (type G0). Itérable, à
critères **INCHANGÉS**, par changements **physiquement motivés, nommés, pré-annoncés** ;
**chaque résultat de gate remonte à Romain** — aucune boucle silencieuse jusqu'au PASS.

- **(a') CRITÈRE PORTEUR — readout-first (remplace (a) comme primaire)** :
  ordre-mémoire au readout albedo entre deux histoires même-multiset d'ordre inversé :
  `f_ordre ≥ 5 %` du domaine supra-JND, **robuste sur toute la plage** JND ∈ [2, 5] % et
  S_HALF ∈ [0.005, 0.20]. (Référence d'origine : ~9–14 % supra-JND ; le seuil 5 % est un
  placeholder nommé, en-deçà de l'origine avec marge, pas à défendre.)
  `corr(s_A, s_B)` est **rétrogradé en diagnostic** (rapporté, non bloquant ; origine 0.39
  à titre indicatif). Motif gravé : corr est une métrique d'espace-état, la classe que la
  discipline du projet a répudiée trois fois (18.6 %→0 %, 61 %→~0 %) ; la propriété que
  l'Arc A ferme est celle qui atteint le pixel.
- **(b') survie du canal** : `f_ordre(albedo) ≥ 3 × f_ordre(relief)` sur toute la plage
  S_HALF (référence d'origine : ≈3–8×). Le (b) d'origine (séquentiel-vs-simultané) est
  rapporté en diagnostic.
- **(c') closure sanity** : `f(k)` **moyenné sur ≥ 3 seeds**, monotone décroissante aux deux
  géométries (overlap et séparée). Le moyennage inter-seeds est une réduction de variance
  (les rebonds v1/v2 étaient des lectures single-run), PAS un déplacement de seuil : le
  critère de monotonie est inchangé.
- **resfrac** (masse érodée/déposée) : rapporté en signature diagnostique. Il n'est PLUS une
  cible de calibration (leçon v2 : nécessaire-pas-suffisant, gravée).

### §A10 — Design du substrat re-fondé (figé avant build)

- **Mécanisme — transport en suspension (porteur d'ordre PAR CONSTRUCTION)** :
  érosion (θ > θ_c) alimente un champ **suspendu** `c(x)` ; `c` est **advecté** par le
  courant (+ diffusion faible) ; dépôt `ds/dt = w_s·c` sur cellules où θ < θ_c ;
  reprise depuis `s` comme avant ; rétroaction `b_eff = b0 + s` inchangée à chaque épisode.
  L'ordre des pulses s'écrit dans OÙ le suspendu voyage avant de retomber.
- **BC PÉRIODIQUES (gravé, motivé physiquement)** : la topologie cible du but est un monde
  fermé convexe bouclé (le niveau 0 de Harten = la planète ; conservation exacte sans termes
  de bord). Conséquence immédiate : la fuite au mur documentée (dérive d'eau 3–13 %/épisode)
  disparaît par construction. L'assèchement inter-épisodes reste un opérateur d'épisode
  (h remis à sec, `s` et `c`… — voir contrainte : `c` est DÉPOSÉ intégralement à
  l'assèchement, jamais détruit, sinon la conservation ment).
- **Knob neuf : `w_s`** (vitesse de chute ; longueur d'advection L_adv ≈ |u|·h/w_s) — c'est
  LE porteur d'ordre, donc le knob qui peut fabriquer le verdict. **Balayage Goldilocks
  PRÉ-ENREGISTRÉ (bloquant)** : ≥ 5 valeurs log-espacées ; à chaque w_s : `f_ordre(w_s)`
  (readout) + diagnostics (corr, resfrac, part érodée redéposée à distance > 1 cellule).
  Extrêmes pré-nommés : w_s→∞ = dépôt local instantané → retombe sur v1/v2, `f_ordre`→0
  attendu ; w_s→0 = rien ne tient → `f_ordre`→0. **Point d'op = choisi sur un PLATEAU de la
  courbe** (région où (a') tient avec robustesse), jamais un point isolé qui passe. Aucun
  w_s ne tient sur la plage → **BLOCKED, remonter sans forcer**.
- **Conservation de la masse sédimentaire = ASSERT BLOQUANT** :
  masse(s) + masse(c) − (érodé − déposé) comptée à chaque pas ; tolérance 1e-10 relatif par
  épisode (le périodique la rend exacte, sans flux de bord). Un assert qui tire = bug, STOP.
  Motif gravé : un champ suspendu qui fuit peut fabriquer OU masquer de l'ordre-mémoire ;
  les écritures persistantes exigent la conservation (principe §A6-1).
- **Calibration** : k_d/θ_c — UNE calibration pré-annoncée vers le point d'op gravé
  (s_max ≈ 20 % du relief à L₀ = 10 pulses), constantes gelées AVANT le gate. **w_s n'est
  PAS calibré au gate : il est balayé (Goldilocks) et son point d'op gravé depuis la courbe.**
- **Readout turbidité** (intégrale de colonne de `c`) : désormais disponible par construction
  — construit et rapporté en **diagnostic**, NON porteur du gate (l'albedo reste le canal
  gravé). Son éventuelle entrée dans M-A1/M-A2 (le §A1 nomme « turbidité/albédo ») est une
  décision à graver au point d'arrêt post-gate, pas un choix d'exécutant.

### §A11 — Bras nul v2 (contrôle négatif intégré, bloquant pour l'instrument)

- v2 (constantes gelées) est **conservé et commité comme substrat-contrôle** :
  quasi-commutatif, deux signatures tenues — un négatif calibré gratuit.
- **Usage gravé** : le pipeline manche 1 complet (compresseur, régénérateur, M-A1/M-A2/M-A3,
  k\*(L)) tourne AUSSI sur v2. Attendu pré-écrit : **saturation triviale** de k\*(L)
  (fermeture facile d'une histoire absente). Si le pipeline ne distingue PAS v2 du substrat
  re-fondé (courbes k\*(L) indiscernables), **c'est l'instrument qui ment → STOP**.
- M-0 (§A8) fournit la baseline readout du bras.
