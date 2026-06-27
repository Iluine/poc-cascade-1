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

**Ne pas surclamer dans aucun sens** : ni « le routage est mort » (faux), ni « il suffit de monter
le Re » (non mesuré), ni « T1 a testé l'architecture » (faux : 2 sorties sur 3, descend non interrogé).
