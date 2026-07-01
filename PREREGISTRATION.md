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
