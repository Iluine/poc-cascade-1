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

### 2026-07-04 — M-0 (§A8) : LECTURE 2 — f_ordre(v2) = 0.7595 à la cellule porteuse (seuil ≤ 0.01) — le readout voit sur ~76 % du domaine une « histoire » que corr = 0.9446 déclarait quasi absente. STOP pré-enregistré : le gate §A9 est à repenser AVANT tout build

**Exécution** : Task 0 (smoke bras nul : `run_history(12345, 10)` aux constantes gelées redonne
max(s)/relief = 0.18876, hash consigné au ledger) puis Task 1 M-0, protocole STRICTEMENT celui du
critère (a) du gate v2 (seed 7, 10 centres, même multiset, ordre direct vs inversé,
`SedimentParams()` V2 gelées) ; formule §A8 (dénominateur symétrique ⟨(A_dir+A_inv)/2⟩ — écart de
convention avec `f_fraction` documenté, le contrat fait foi). Revue indépendante : conforme,
vérification croisée protocole + sanity-check numérique (fractions multiples exactes de 1/4096).

**Mesure (balayage complet en `outputs/arcA/m0_v2_readout.json`)** :
- corr(s_direct, s_inversé) = **0.94455** (= gate v2, confirme les constantes).
- f_ordre(albedo) ∈ [0.74, 0.89] sur TOUT le balayage JND ∈ {2..5} % × S_HALF ∈ {0.005..0.20}·relief ;
  cellule porteuse (JND = 5 %, S_HALF = 0.05·relief) : **f_ordre = 0.7595**.
- Plancher de canal (relief ombré, même protocole) : f_ordre ≤ 0.0015 — l'amplification est
  spécifique au readout albedo, pas un artefact de la méthode.

**Verdict mécanique (§A8, pré-écrit)** : 0.7595 > 0.01 → **lecture_2** : « le readout AMPLIFIE une
histoire que corr sous-estime → STOP, remonter : la hiérarchie corr↔readout est inversée, le gate
§A9 doit être repensé avant tout build. » Tasks 2–4 (build suspension, Goldilocks, gate) **NON
lancées**. Aucun seuil retouché.

**Portée EXACTE et implications (sans surclamer, dans les deux sens)** :
1. **Le critère porteur (a') tel qu'écrit est VACANT** : le bras nul lui-même donne f_ordre = 0.76
   ≫ 5 %. Un substrat re-fondé qui « passerait » (a') n'aurait rien démontré — le seuil est
   franchi par un substrat quasi-commutatif en état. C'est exactement ce que M-0 (le moins cher
   qui peut échouer) devait attraper avant le build.
2. **Ce que la mesure ne tranche PAS** : si ces 76 % sont de l'ordre-mémoire STRUCTURÉ ou de la
   sensibilité amplifiée sans structure (« texture » : corr état 0.94 ⇒ ~11 % de variance non
   partagée, que la normalisation JND-relative rend supra-seuil presque partout). f_ordre compte
   les cellules franchissant le seuil, il ne voit pas si l'écart est organisé.
3. **La suspension de la manche 1 (2026-07-04, gate v2) était motivée par corr = 0.94 — une
   métrique d'espace-état**, la classe que la discipline du projet répudie. Au readout — l'espace
   où vit le verdict M-A1/M-A2 — v2 n'est PAS « quasi sans histoire » : l'attendu « saturation
   triviale de k*(L) » du bras nul (§A11) n'est plus garanti, et la prémisse « mesurer k*(L) sur
   v2 fabriquerait un PASS trivial » est affaiblie. On grave le fait, pas une réhabilitation.
4. La référence d'origine (~9–14 % supra-JND) est très en-dessous du 76 % de v2 : soit la
   convention f d'origine différait (code perdu, invérifiable), soit le point d'op d'origine était
   moins amplifiant. Comparaison indicative seulement — ne pas s'y appuyer.

**Fork remonté à Romain (STOP §A8, options)** :
(1) **Repenser le critère porteur pour qu'il isole la STRUCTURE d'ordre, pas le franchissement de
seuil** — p. ex. critère CONTRASTIF au bras nul (l'objet §A9 devient « le re-fondé porte
significativement plus d'ordre-mémoire readout que v2 », mesuré par une statistique organisée :
corrélation spatiale de ΔA avec le déplacement des dépôts, ou f_ordre à JND élevé où v2 retombe),
puis build §A10 sous ce gate repensé ;
(2) **Ré-examiner la suspension de la manche 1 sur v2** : si le readout porte l'histoire (76 %),
la fermeture sub-JND d'un résumé grossier (M-A1/M-A2) sur v2 n'est plus triviale par construction
— la manche 1 redevient peut-être testable SANS nouveau substrat (le moins cher qui peut échouer) ;
(3) suspendre l'Arc A. Les options (1) et (2) ne s'excluent pas : (2) d'abord est l'ordre du
moindre coût si la testabilité sur v2 se confirme.

**Commits pocPhysicator** : 09fca34 (M-0 + JSON), 0568e55 (ledger). Contrat : 598fabd (§A6–§A11),
b778924 (plan).

### 2026-07-04 — M-0bis PRÉ-ENREGISTRÉE (avant exécution) : sonde de testabilité de la manche 1 sur v2 — arbitrage option (2) du fork M-0

**Arbitrage Romain** : « 2, mesure la testabilité sur v2 d'abord ». Le risque à exclure avant de
relancer la manche 1 sur v2 est le **PASS fabriqué** (fermeture triviale d'un résumé grossier,
l'attendu §A11 du bras nul). La sonde est le moins cher qui peut falsifier la testabilité.

**Protocole (figé avant build/run)** :
- Build de `src/summary.py` **strictement selon la spec gravée du plan manche 1, Task 4**
  (`docs/superpowers/plans/arc-a-manche1.md` : block-mean dyadique ℓ ∈ {1,2,3} + masse totale +
  masses 4×4 ; régénération bilinéaire, clip ≥ 0, rescale multiplicatif par sous-domaine, division
  protégée ; tailles 1041/273/81 floats vs 4096) + les tests du plan (M-A3 ≤ 1e-12 et identité
  cross-process, anti-fuite par `inspect.signature` + décoy, tailles exactes, invariants 1e-10,
  round-trip constant exact). Aucun écart de spec sans STOP.
- Champs sondés : `s_direct` et `s_inversé` de M-0 (seed 7, L₀ = 10, constantes V2 gelées).
- **M-A1 instantané** : Δχ(albedo(regenerate(summarize(s, ℓ))), albedo(s)) au point d'op S_HALF,
  critère R1 (max sur bandes porteuses ≥ 10 % AC), pour ℓ ∈ {1, 2, 3}.
- **M-A2-mini** : rollout de 1 épisode — centre = 11ᵉ tirage de la même rng (seed 7), IDENTIQUE
  pour les deux histoires — depuis s_regen(ℓ) ET depuis s_vrai ; Δχ des albedos à chaque snapshot
  d'intégration Exner le long de l'épisode ; statistique = **max le long du rollout** (statistique
  M-A2 gravée, restreinte à 1 épisode). L'accès aux s intermédiaires se fait par fonction ADDITIVE
  (aucun chemin de code v2 modifié).
- Niveaux sous le cap anti-trivialité (§A3 : ≤ 409.6 floats) : **81 et 273** ; ℓ=1 (1041) rapporté
  hors-cap à titre indicatif. Diagnostic rapporté : f_close (convention M-0) par niveau.

**Lectures pré-écrites (mécaniques)** :
- **T-triviale** : pour TOUS les niveaux sous cap (81 ET 273) et LES DEUX histoires,
  max(M-A1, M-A2-mini) < 2 % (JND le plus sévère) → la fermeture est triviale sur toute la plage
  JND → manche 1 sur v2 VACANTE en l'état → retour au fork (option 1 : gate contrastif + build
  suspension redevient la voie).
- **T-testable** : sinon (≥ 1 cellule niveau-sous-cap × histoire × JND ∈ [2,5] % qui ne ferme pas)
  → l'instrument a de la plage dynamique → remonter avec recommandation de relancer la manche 1
  sur v2. NOTE : T-testable falsifierait AUSSI l'attendu pré-écrit §A11 (« saturation triviale du
  bras nul ») — à requalifier au point d'arrêt, pas silencieusement.

**Portée** : la sonde ne TRANCHE que la testabilité. Elle n'est PAS le verdict manche 1 (k\*(L)
exige le balayage L ∈ {10,20,40,80} × 5 seeds du plan gravé) et ne s'y substituera pas.

### 2026-07-04 — M-0bis : T-TESTABLE — max(M-A1, M-A2-mini) sous cap ∈ [0.13, 0.63] vs seuil 0.02 : la fermeture d'un résumé grossier sur v2 n'est PAS triviale ; l'attendu §A11 (« saturation triviale du bras nul ») est FALSIFIÉ

**Exécution (protocole gravé `ae1c926`, appliqué sans écart)** : `src/summary.py` construit en TDD
strictement selon la spec gravée du plan manche 1 Task 4 (block-mean dyadique + invariants ;
régénération bilinéaire + rescale de masses 4×4 sur grille fine, division protégée ; 21 tests dont
M-A3 double-régénération ≤ 1e-12 + identité cross-process, anti-fuite signature+décoy, tailles
exactes 1041/273/81) ; trajectoire d'épisode par fonction ADDITIVE (`run_episode_trajectoire`,
0 ligne existante modifiée, +4 tests) ; suite complète **130 verts** ; sonde déterministe
(double run bit-identique). Revue indépendante : conforme, 0 finding bloquant — additivité,
continuité de la rng (10 premiers centres bit-à-bit = M-0) et rescale 4×4 vérifiés sur pièces.

**Mesure (S_HALF op, critère R1 ; détail en `outputs/arcA/testability_v2.json`)** :

| Histoire | ℓ (floats) | M-A1 | M-A2-mini (max rollout) | sous cap ? |
|---|---:|---:|---:|:---:|
| direct  | 1 (1041) | 0.0317 | 0.0339 | non (indicatif) |
| direct  | 2 (273)  | 0.1456 | 0.1665 | oui |
| direct  | 3 (81)   | 0.5060 | 0.5476 | oui |
| inverse | 1 (1041) | 0.0256 | 0.0256 | non (indicatif) |
| inverse | 2 (273)  | 0.1310 | 0.1582 | oui |
| inverse | 3 (81)   | 0.5509 | 0.6280 | oui |

Bande porteuse dominante : 4–7 cycles/domaine (taille des bosses du terrain). Diagnostic f_close
cohérent (39–94 % du domaine supra-JND selon ℓ).

**Verdict mécanique (pré-écrit)** : les 4 cellules sous cap × 2 histoires sont ≥ 0.13 ≫ 0.02 →
**T_testable**. Aucun seuil retouché.

**Portée EXACTE** :
1. **La manche 1 sur v2 n'est PAS vacante** : la fermeture sub-JND d'un résumé grossier y est une
   vraie question — il existe une plage dynamique réelle (ℓ=1 : ~0.03 ; ℓ=2 : ~0.15 ; ℓ=3 : ~0.6,
   à cheval sur la plage JND [0.02, 0.05]). La prémisse de la suspension (« PASS trivial garanti »)
   est falsifiée pour v2.
2. **L'attendu pré-écrit §A11 est FALSIFIÉ** : v2 ne produit PAS de saturation triviale. Si la
   manche 1 tourne SUR v2, il n'y a plus de bras nul distinct — le contrôle « instrument menteur »
   de §A11 doit être requalifié au fork (pas silencieusement).
3. **Indication non verdictale** (à L₀=10 seulement, 1 seed, sans préjuger du k\*(L) de la manche) :
   rien sous cap ne ferme même à JND=5 % ; ℓ=1 (hors cap) ferme à 5 % mais pas à 2 %. Le verdict
   manche 1 (§A3 : saturation de k\*(L) sous 409.6 floats) a une vraie question à trancher dans les
   deux sens.
4. La sonde ne se substitue PAS au verdict manche 1 (balayage L × seeds requis).

**Fork remonté à Romain** :
(1) **Relancer la manche 1 SUR v2** — Tasks 3–6 du plan gravé `arc-a-manche1.md` (histoires
L ∈ {10,20,40,80} × 5 seeds, M-A1/M-A2/M-A3, k\*(L), verdict §A3 mécanique), avec DEUX amendements
à graver avant lancement : (i) substrat = v2 gelé (co-calibré, gate d'identité FAIL — assumé : les
claims §A1–§A5 portent sur « ce substrat », pas sur l'original perdu) ; (ii) §A11 requalifié — le
contrôle-instrument devient un bras nul SYNTHÉTIQUE à histoire détruite (p. ex. mélange spatial du
champ s préservant l'histogramme, ou champ s d'une histoire à 1 seul pulse de masse équivalente),
attendu : k\*(L) plat/trivial sur le bras détruit, contrasté sur v2 ;
(2) build suspension d'abord (§A10, gate §A9 à repenser — l'objection M-0 « (a') vacant » reste
entière pour CE chemin) puis manche 1 dessus — plus cher, ne teste pas plus de claim ;
(3) suspendre. **Recommandation : (1)** — la manche 1 redevient testable au moindre coût, et le
build suspension reste disponible si le verdict manche 1 l'exige.

**Commits pocPhysicator** : 32e9004 (summary.py TDD), 48d771f (trajectoire additive), cc61056
(sonde + JSON), bf62b7e (ledger).

### 2026-07-04 — Arbitrage fork M-0bis : option (1) — RELANCE DE LA MANCHE 1 SUR v2. Deux amendements gravés AVANT lancement

**Arbitrage Romain** : « 1. je suis ta recommandation ». La manche 1 (plan gravé
`arc-a-manche1.md`, Tasks 3/5/6 — la Task 4 `summary.py` est déjà construite et revue sous
M-0bis) se relance sur le substrat v2, avec les deux amendements suivants. Les seuils, mesures et
verdicts de §A2–§A3 sont par ailleurs INCHANGÉS (L ∈ {10,20,40,80} × seeds {101..105},
M-A1/M-A2/M-A3, cap 409.6 floats, k\*(L) médiane + clause 2×JND, pente IC 95 % bootstrap,
sensibilité JND {2..5} % bloquante).

**Amendement (i) — substrat** : le substrat de la manche 1 est **v2 gelé**
(`SedimentParams()` par défaut = KD_CALIBRE_V2/KE_CALIBRE_V2, terrain `default_terrain`,
protocole d'épisode inchangé). Assumé : v2 n'est PAS le substrat d'origine (gate d'identité FAIL
2/3, voie reconstruction close) ; les claims §A1–§A5 se lisent sur CE substrat, conformément à
§A5. La testabilité est établie par M-0bis (T_testable), pas présumée.

**Amendement (ii) — §A11 requalifié : bras de contrôle instrument SYNTHÉTIQUES** (l'attendu
d'origine « saturation triviale de v2 » est falsifié par M-0bis ; v2 devient le bras de mesure).
Le pipeline complet (summarize/regenerate, M-A1, M-A2, k\*) tourne AUSSI sur deux champs de
contrôle à réponse connue par construction, aux cellules L ∈ {10, 80} × seeds {101..105} :
- **Contrôle-fermable** : `s_ferm = regenerate(summarize(s_L, ℓ=3))` (le champ vrai remplacé par
  sa propre version 81-floats régénérée). Attendu pré-écrit : k\* = 81 floats (sous-JND dès ℓ=3,
  M-A1 ET M-A2) à tous les JND de {2..5} %.
- **Contrôle-infermable** : `s_shuf` = permutation aléatoire des 4096 cellules de s_L
  (histogramme exactement préservé, structure spatiale détruite), permutation tirée de
  `default_rng(9001 + 1000·L + seed)`. Attendu pré-écrit : k\* = ∞ (aucun niveau ne ferme, à
  aucun JND de la plage).
- **Lecture (gate d'instrument, type G0)** : toute cellule de contrôle qui viole son attendu →
  **STOP, remonter** — l'instrument est suspect (itération d'instrument légitime sous discipline
  normale : nommée, pré-annoncée, jamais silencieuse). Les contrôles ne participent PAS au
  verdict §A3 ; ils conditionnent le droit de le lire.

**Coût annoncé** : histoires 5 × 80 épisodes (~40 min CPU) + M-A2 mesure 4×5×3×2 épisodes
(~11 min) + contrôles 2×2×5×3×2 épisodes (~23 min) — ~1h15 CPU total, en arrière-plan.

### 2026-07-04 — Manche 1 sur v2, pipeline complet exécuté : GATE D'INSTRUMENT EN VIOLATION (contrôle-fermable 40/40) — verdict §A3 SCELLÉ, non prononçable. Le contrôle synthétique a attrapé un plancher instrumental supra-JND : le régénérateur bilinéaire ne peut certifier k\*=81 pour AUCUN champ

**Exécution (Tasks 3/5/6 du plan gravé + amendements `8e02dd6`, revues indépendantes à chaque
tâche, un correctif de traçabilité au gate — même seuils, verdict.json inchangé au champ près)** :
- Task 3 : 5 histoires × 80 épisodes (seeds 101–105), checkpoints L ∈ {10,20,40,80}, npz commités,
  sanity PASS (path-dependence inter-seeds au readout f = 0.71 ; replay bit-à-bit).
- Task 5 : mesures M-A1/M-A2 des trois bras (v2 grille complète ; contrôles ferm/shuf à L ∈ {10,80}),
  `measures.npz`. Déterminisme vérifié.
- Task 6 : k\*(L) + clause 2×JND, gate des contrôles PAR CELLULE avant toute lecture, pentes
  bootstrap (10 000, seedé), verdict §A3 mécanique, figures + GIF pire-cas. 139 tests verts.

**GATE DES CONTRÔLES (amendement ii, lu en premier)** :
- **shuf (attendu k\* = ∞) : CONFORME 40/40** — max(M-A1,M-A2) ∈ [1.36, 2.28], jamais fermé.
- **ferm (attendu k\* = 81) : VIOLATION 40/40** — jamais k\* = 81 (souvent 1041, parfois ∞).
- → **verdict §A3 du bras v2 SCELLÉ « NON LISIBLE »** (audit, non-verdict : INDÉTERMINÉ ×4 JND,
  k\*(L) = ∞ partout — rien ne ferme sous le cap à aucun L). Cellule §A0 : NON DÉTERMINÉE.

**Diagnostic (mesuré, pas spéculé — chiffres du contrôle-fermable = champ DÉJÀ lisse à l'échelle
8 cellules)** : le round-trip du compresseur, R∘S, n'est pas une projection — l'upsampling
bilinéaire n'est pas un inverse à droite de la moyenne par blocs. Son PLANCHER propre, mesuré sur
son propre point image : Δχ ≈ 0.014–0.032 (ℓ=1), 0.04–0.08 (ℓ=2), **0.09–0.19 (ℓ=3) — au-dessus
de TOUTE la plage JND [0.02, 0.05]**. Conséquence structurelle : avec CE régénérateur, aucun champ
— même parfaitement résumable — ne peut fermer à 81 floats, et 273 est marginal. La question §A3
(k\*(80) ≤ 409.6) était inrépondable positivement par construction. Le bras nul synthétique a fait
exactement son travail (leçon M-0/M-0bis appliquée : c'est le 3ᵉ instrument que les contrôles
attrapent avant qu'il fabrique un verdict).

**Ce que les données disent SOUS le scellé (audit, à ne pas surclamer)** : le signal v2 est 2–5×
au-dessus du plancher instrumental à chaque niveau (ℓ=1 : 0.02–0.12 vs plancher ~0.02 ; ℓ=2 :
0.10–0.39 vs ~0.06 ; ℓ=3 : 0.27–0.85 vs ~0.14) — la non-fermeture de v2 n'est pas QUE du
plancher. Mais la part exacte est indémêlable tant que le plancher est supra-JND : verdict
illisible, pas de FAIL prononcé.

**Fork remonté à Romain (STOP obligatoire du plan, options)** :
(1) **Itération d'instrument nommée : régénérateur → upsampling constant-par-blocs (ordre 0,
Harten canonique)**. S∘R devient l'identité EXACTE (R∘S = projection) : plancher = 0 bit-à-bit,
le contrôle-fermable passe par construction (et reste un vrai test de plomberie), toute
non-fermeture mesurée sur v2 devient du signal pur. Prix assumé : les artefacts de bloc du
régénéré comptent comme différence perceptuelle réelle — c'est honnête, le claim §A2 porte sur le
couple résumé+régénérateur. Implémentation additive (`summary.py`), mêmes seuils, re-run
Tasks 5–6 (~12 min). Amendement de la spec Task 4 du plan (bilinéaire → constant) à graver.
(2) Garder le bilinéaire et redéfinir le contrôle-fermable comme point fixe de R∘S — répare
l'équité du contrôle mais PAS le plancher : le verdict v2 resterait confondu. Déconseillé.
(3) Suspendre.
**Recommandation : (1).**

**Commits pocPhysicator** : 04ac96f (histoires), a64185b+4a572c5 (mesures), 96eaf70+cf6c131
(verdict+figures), 12f0446 (traçabilité gate), 846abce (ledger).

### 2026-07-04 — Arbitrage fork instrument : option (1), régénérateur → CONSTANT-PAR-BLOCS, avec argument renforcé + TROIS CLAUSES gravées AVANT le re-run + règle de forme

**Arbitrage Romain (option 1), endossé sur un argument plus fort que celui de l'exécutant** :
l'alternative « intelligente » évidente — garder le bilinéaire et le corriger par constante
additive par bloc pour rendre les moyennes exactes (prolongation conservative classique) —
échoue précisément sur CE substrat : elle exige de pouvoir descendre sous zéro aux bords des
dépôts, où le bilinéaire undershoot ; le clip ≥ 0 casse alors l'exactitude des moyennes exactement
là où s vit (champ sparse, majoritairement nul). Sur un champ positif et sparse avec clip, le
constant-par-blocs n'est pas seulement plus simple : **c'est la seule projection exacte de la
famille**. Et il est propre en cascade : moyennes de blocs exactes ⇒ masses 4×4 exactes ⇒
rescale ×1 ⇒ clip inactif ⇒ S∘R = id bit-à-bit. L'option (2) (point fixe) est écartée : répare
l'équité du contrôle, laisse le verdict confondu.

**Note d'implémentation (fidèle à l'argument cascade, gravée)** : le rescale devient une
VÉRIFICATION (masses 4×4 à 1e-12 relatif) sans multiplication — les moyennes étant exactes par
construction, multiplier par un facteur 1±ulp (ordre de sommation) détruirait l'identité
bit-à-bit sans gagner d'exactitude. Le clip reste (no-op sur champ ≥ 0). Le bilinéaire est
conservé sous un nom d'audit (reproductibilité de M-0bis et du run v1) mais n'est plus le
régénérateur du claim.

**Clause 1 — la cellule de verdict manquante (gravée avant le re-run, pendant que c'est
incertain)**. La grille §A3 présuppose des k\* finis : PASS lit une pente nulle, FAIL-mur lit une
pente positive. L'audit scellé montre k\*(L) = ∞ partout, et le constant-par-blocs va durcir les
Δχ — **prédiction directionnelle gravée : M-A1/M-A2 de v2 montent à chaque ℓ** (on grave la
mesure, pas le mécanisme). L'issue « k\* = ∞ uniforme, rien ne ferme sous le cap à aucun L, y
compris L₀ » est pré-écrite comme **INDÉTERMINÉ-CAPACITÉ** : la famille {block-mean ℓ, invariants}
manque de capacité à ce JND. Et surtout ce qu'elle n'est PAS : **ce n'est pas le mur** — le mur
est un énoncé de croissance avec l'histoire ; un instrument qui ne ferme même pas une histoire de
10 pulses ne dit rien sur l'accumulation. Conséquence pré-écrite : ne sélectionne PAS la cellule
2/3 de §A0 ; remonte le **fork famille-vs-manche-2** (une itération de famille serait une décision
neuve, nommée, pré-enregistrée — pas un réflexe). Note honnête : sous le cap de 409.6 floats, k\*
ne peut valoir que {81, 273, ∞} — la machinerie de pente sur trois valeurs quantifiées est fragile
par construction ; connu, assumé, pas retouché maintenant.

**Clause 2 — le prix des blocs, chiffré au lieu de subi**. Avec le constant-par-blocs, le
contrôle-fermable devient exact au round-trip : son M-A2 ne sonde plus rien. Diagnostic
NON-BLOQUANT ajouté, une cellule suffit — gravée : **(L=10, seed=101, ℓ=3)** : rollout depuis
s_ferm (bloqué) vs depuis s_L (lisse), trajectoire complète Δχ(t) sérialisée ; lecture rapportée =
Δχ(0) (part compression statique) vs max_t Δχ(t) (avec dynamique) et leur rapport = la
contribution de la dynamique-sur-marches seule. Si le verdict tombe en famille-insuffisante, ce
chiffre démêle la part escalier de la part compression — sans lui on rejouerait le débat du
plancher un cran plus loin.

**Clause 3 — requalification explicite du contrôle-fermable**. Il passe désormais par
construction : il DESCEND au rang de contrôle de plomberie (déterminisme, anti-fuite, idempotence
bit-à-bit). Le contrôle discriminant restant est le shuf. Que personne ne relise « ferm 40/40
conforme » comme une validation forte.

**Leçon méta (consignée, sans en faire un arc)** : deuxième fois que la grille de verdicts
pré-enregistrée ne couvre pas l'espace réel des issues (la grille §A0 à trois cellules supposait
un substrat qui existe ; la grille §A3 supposait des k\* finis). **Règle de forme pour les
prochaines pré-enregistrations : toute grille inclut par défaut une cellule « l'instrument/la
famille ne peut pas répondre ».** Le scellé d'aujourd'hui a fonctionné, mais parce qu'un
amendement de dernière minute l'a fourni, pas parce que la grille le prévoyait.

**Exécution ordonnée** : amender la spec Task 4 du plan (bilinéaire → constant-par-blocs),
apposer les clauses, re-run Tasks 5–6 + diagnostic clause 2, remonter le verdict — qui sera,
pour la première fois de la manche, lisible quel qu'il soit. Les sorties v1 (bilinéaire) restent
dans l'historique git (commits a64185b, 4a572c5, cf6c131).

### 2026-07-04 — Manche 1 sur v2, instrument ordre 0 : gate CONFORME, PREMIER VERDICT LISIBLE — global INDÉTERMINÉ, cellule INDÉTERMINÉ-CAPACITÉ à JND=2 % (clause 1 tirée) ; prédiction directionnelle FALSIFIÉE (les Δχ baissent 120/120) ; première structure en L de la manche : k\*=1041 aux petits L, ∞ à L=80 (JND 4–5 %), hors cap

**Exécution (arbitrage `9bcb09a` + clauses, 4 volets, revue indépendante avec vérification sur
données)** : `regenerate` = constant-par-blocs (clip conservé, vérification masses 1e-12 SANS
multiplication) ; bilinéaire conservé en audit. Incident consigné : la promesse « S∘R = id
bit-à-bit » échouait d'1 ULP à ℓ=3 (réduction numpy non-binaire sur 64 élts/bloc) — l'implémenteur
a STOPpé correctement ; résolution du contrôleur : **`summarize` ré-implémentée en cascade 2×2
itérée** (définition RÉCURSIVE du cell-average de Harten, sémantique inchangée, écarts ≤ ulps
documentés) — l'argument cascade gravé exigeait l'implémentation récursive pour être vrai en
flottant. S∘R strict et idempotence stricte vérifiés aux 3 niveaux. 164 tests verts.
Reproduction bit-exacte de M-0bis/run v1 : par checkout historique (consigné).

**Gate des contrôles : CONFORME 0/80** — ferm 0/40 (Δχ = 0.0 exact, passe PAR CONSTRUCTION :
contrôle de plomberie, clause 3, à ne pas relire comme validation forte) ; shuf 0/40 (k\* = ∞
partout, Δχ ∈ [0.86, 2.18] — le contrôle discriminant). **Le verdict est LISIBLE.**

**Prédiction directionnelle gravée (« M-A1/M-A2 v2 montent à chaque ℓ ») : FALSIFIÉE.** Baisse
dans 120/120 cellules (0 hausse, 0 égalité ; recoupé indépendamment contre les valeurs v1).
Factuel : le bilinéaire + rescale produisait des Δχ spectraux PLUS GRANDS que l'escalier
constant-par-blocs. On grave la mesure ; le mécanisme n'est pas tranché.

**k\*(L ; JND) bras v2 (médiane 5 seeds + clause 2×JND ; v1 : ∞ partout — premières cellules
finies de la manche)** :

| L | JND=2 % | JND=3 % | JND=4 % | JND=5 % |
|---|---|---|---|---|
| 10 | ∞ | 1041 | 1041 | 1041 |
| 20 | ∞ | ∞ | 1041 | 1041 |
| 40 | ∞ | ∞ | 1041 | 1041 |
| 80 | ∞ | ∞ | ∞ | ∞ |

Aucun k\* fini < 1041 : ℓ=2 (273) et ℓ=3 (81) ne ferment JAMAIS, à aucun JND, aucun L.

**Verdict §A3 (mécanique, lisible)** : JND=2 % → **INDÉTERMINÉ-CAPACITÉ** (clause 1, message
gravé : famille insuffisante à ce JND ; N'EST PAS le mur ; pas de cellule §A0 2/3 ; fork
famille-vs-manche-2). JND=3/4/5 % → INDÉTERMINÉ (k\*(80) = ∞ → pente indéfinie ; 1041 > cap
partout ailleurs). **Global : INDÉTERMINÉ** (non stable sur la plage).

**Diagnostic escalier (clause 2, non-bloquant, cellule gravée L=10/seed=101/ℓ=3)** :
Δχ(0) = 0.3265 (part compression statique), max_t Δχ(t) = 0.3875 (t=87), rapport **1.187** —
la compression statique domine, la dynamique-sur-marches ajoute ~19 %. Le débat du plancher est
démêlé d'avance : en cas de famille-insuffisante, ~84 % du Δχ vient de la compression elle-même.

**Portée EXACTE (hors verdict, à ne pas surclamer)** : à 1041 floats (25 % du champ, HORS cap),
la fermeture tient aux petits L et casse à L=80 (JND 4–5 %) — **première observation en forme de
mur de la manche** (k\* croît avec l'histoire), mais au-dessus du cap et illisible par la grille
§A3 : ce n'est PAS un verdict de mur. Sous le cap, la famille {block-mean ℓ, invariants} n'a pas
la capacité, à aucun JND — c'est le sens précis d'INDÉTERMINÉ-CAPACITÉ.

**Fork remonté à Romain (pré-écrit par la clause 1 : famille-vs-manche-2, décision neuve, nommée)** :
(1) **Itération de famille nommée** : passer du coarse uniforme à un résumé ADAPTATIF (garder les
plus grands coefficients de détail de la MRA de Harten sous un budget de floats — la machinerie
conceptuelle existe côté pocCascade2phys ; portage numpy minimal ici). Capacité par float
strictement supérieure ; l'observation hors-cap (fermeture à 1041 aux petits L) donne une vraie
chance de faire rentrer k\*(petits L) sous le cap et de rendre la croissance en L LISIBLE dans la
grille. Pré-enregistrement requis : famille nommée, grille §A3 amendée avec cellule
« ne-peut-pas-répondre » (règle de forme), attendus des contrôles reconduits.
(2) **Basculer manche 2** (le filet §A6 : le différenciateur survit) — la machinerie
événements/ledger/reconstruction, plus chère, mais le substrat et l'instrument perceptuel
sont maintenant en place.
(3) Suspendre.
**Recommandation : (1)** — une itération de famille est exactement ce que l'observation hors-cap
appelle, et elle recycle tout le pipeline validé (histoires, mesures, gate, verdict).

**Commits pocPhysicator** : c35e4dd (ordre 0 + cascade), ab7de1e (clause 1), 04b8163 (re-run),
c518b11 (diag escalier), 7f12dbd (ledger).

### 2026-07-04 — Arbitrage fork famille-vs-manche-2 : option (1) AMENDÉE — famille 2 = QUADTREE de moyennes (raffinement adaptatif par blocs), cinq clauses gravées avant le run, trois issues pré-écrites. Gardes de relecture posées sur le verdict précédent

**Gardes de relecture (les deux sens, gravées)** : la structure hors-cap (1041 tient à
L ∈ {10,20,40}, casse à 80, à JND 4–5 %) est **un saut, sur une échelle quantifiée à trois
valeurs, à deux JND sur quatre, au-dessus du cap : un INDICE qui motive l'itération, pas « la
forme du mur »** — que personne ne le raconte ainsi dans dix sessions. Symétriquement : que
personne ne l'enterre — c'est la première dépendance en L de toute la manche, et elle est
exactement ce que la famille suivante doit rendre lisible dans la grille.

**Arbitrage Romain : option (1) endossée, famille AMENDÉE — le top-k de coefficients de détail a
un piège, nommé avant qu'il tire** : la reconstruction MRA tronquée d'un champ positif peut
undershooter sous zéro (un détail fin gardé appliqué sur une moyenne intermédiaire jetée) → clip
≥ 0 → R∘S n'est plus une projection → plancher non nul — le piège d'il y a douze heures, remonté
d'un niveau. Le gate ferm l'attraperait, mais ce serait un cycle d'instrument prévisible et payé
pour rien. **Famille 2 = raffinement adaptatif par blocs : le quadtree de moyennes de cellules,
l'adaptativité de Harten au sens canonique.** Blocs larges où s≈0, fins sur les dépôts. Trois
propriétés d'un coup : (i) moyennes d'un champ positif = positives → pas de clip → **S∘R = id
exact, plancher = 0 conservé par construction** — la propriété qui vient de rendre le verdict
lisible ; (ii) coût d'indexation trivial et comptable (bits d'arbre) ; (iii) **cohérence
d'architecture** : c'est littéralement le mécanisme de stockage que les invariants-dans-z
utiliseraient dans le moteur (niveaux de Harten adaptatifs, pas une soupe de coefficients
globaux). Si cette famille ferme, le verdict est directement architecture-pertinent ; si elle ne
ferme pas, il l'est aussi. Le top-k reste nommable en (1b) avec un gate de plancher quantifié
(mesuré ≤ 0,5 % à chaque budget) en remplacement du zéro exact — non retenu : rien qu'il achète
que le quadtree n'a pas.

**Les cinq clauses (gravées avant le run)** :
1. **Comptabilité du résumé** : la taille inclut la structure (bits d'arbre, convention gravée :
   32 bits = 1 float-équivalent), gravée avant mesure, cap INCHANGÉ à 409,6. Sans ça, les k\*
   inter-familles ne sont pas comparables et le cap anti-trivialité est truqué en silence.
2. **Grille de budgets gravée** : {32, 64, 128, 256, 400} sous cap + {1024, 2048} hors-cap en
   diagnostic. Réparation de la fragilité notée : k\*(L) devient une courbe sur 5+2 points au
   lieu de trois valeurs quantifiées — la machinerie de pente §A3 a enfin de quoi mordre.
3. **Règle de dernière famille** : c'est la DEUXIÈME ET DERNIÈRE famille sous cette
   pré-enregistration. Un second INDÉTERMINÉ-CAPACITÉ → la manche 1 se clôt en **NON-DÉMONTRÉ** :
   sans sélectionner la cellule 2 ni claimer le mur, la planification d'architecture procède sur
   l'hypothèse ledger, et la dépense suivante est la manche 2. Toute famille ultérieure exige un
   fork remonté avec une décision nommée qui en dépend. Sinon « une famille de plus » est le
   treadmill avec un déguisement neuf.
4. **La surface k\*(L, JND), rapportée telle quelle** — additif, règles §A3 inchangées. Lecture
   pré-écrite : des verdicts par-JND divergents mais propres = global INDÉTERMINÉ et la surface
   k\*(L, JND) portée à l'**Arc C** comme l'objet que son pin résout. Le cadrage codec paie :
   cette surface est la courbe débit-distorsion du champ persistant, et le JND réel choisira la
   courbe opérante. La manche 1 et l'Arc C se rejoignent là où les gates l'avaient prévu.
5. **Contrôles reconduits** : shuf inchangé (discriminant) ; ferm en plomberie avec attendu
   zéro-exact restauré par le quadtree.

**Les trois issues, pré-écrites — chacune change une décision, aucune n'est un gâchis** :
- **Fermeture plate sous cap** → cellule 1, court-circuit de la manche 2 sur ce substrat — la
  raison pour laquelle ces ~12 minutes valent d'être jouées avant de payer la machinerie ledger.
- **Croissance lisible sous cap** → FAIL-mur enfin prononçable, cellule 2, le ledger passe
  d'hypothèse à obligation.
- **Second INDÉTERMINÉ-CAPACITÉ** → NON-DÉMONTRÉ, manche 2.

**Spec opérationnelle du quadtree (figée avant build — l'exécutant n'improvise pas la
géométrie)** :
- Partition dyadique adaptative du domaine 64×64 ; chaque feuille stocke la MOYENNE de son bloc.
- Construction gloutonne déterministe : gain d'un split = SSE expliquée = Σ_enfants
  n_c·(moyenne_c − moyenne_parent)² ; on splitte le gain max d'abord ; tie-break lexicographique
  (gain, puis y, puis x) ; **un nœud à gain nul n'est JAMAIS splitté** (condition de la
  projection) ; arrêt quand le budget est atteint ou plus aucun gain > 0.
- Comptabilité : floats = n_feuilles + ceil(n_nœuds/32) (1 bit de topologie par nœud, préordre) ;
  un split coûte +3 feuilles, +4 nœuds. Budget respecté APRÈS chaque split.
- Régénération : chaque feuille peinte à sa moyenne (constant-par-blocs adaptatif), 1 paramètre
  (anti-fuite), pas de clip nécessaire (moyennes ≥ 0), vérification de cohérence interne.
  Propriété testée : S∘R = id bit-à-bit y compris l'ARBRE (les gains sur le champ repeint sont
  identiques par linéarité des moyennes ; les splits sous-feuilles ont gain nul).
- Invariants (masse totale, masses 4×4) : DÉRIVABLES exactement des feuilles (les feuilles
  dyadiques ne chevauchent pas les sous-domaines 16×16 ou les contiennent entièrement) — non
  stockés, pas comptés au budget, vérifiés en test.
- Contrôle ferm : s_ferm = R(S(s_L, budget=32)) — attendu : fermeture EXACTE (Δχ = 0) à TOUS les
  budgets de la grille, k\*(ferm) = 32 partout. Contrôle shuf : inchangé, attendu k\* = ∞.
- Verdict : §A3 inchangé sur la nouvelle grille de budgets + cellule INDÉTERMINÉ-CAPACITÉ +
  mapping mécanique des trois issues (PASS global → cellule 1 ; FAIL global → cellule 2
  FAIL-mur ; global INDÉTERMINÉ-CAPACITÉ → NON-DÉMONTRÉ manche 2 ; sinon → global INDÉTERMINÉ +
  surface portée à l'Arc C).
- Scripts de mesure/verdict : NOUVEAUX (`*_qt.py`), les scripts de la famille 1 restent
  intouchés (reproductibilité par historique).

### 2026-07-04 — Famille 2 (quadtree), run complet : GATE EN VIOLATION — mais cette fois c'est SHUF, 7/40, UNIQUEMENT au budget plafond 2048 ; sous le cap le discriminant est intact (0 violation) et ferm est parfait (Δχ = 0.0 exact, 70 cellules). Verdict scellé. Troisième erreur de PORTÉE d'attendu — fork sur la portée de l'attendu shuf

**Exécution (module quadtree + harnais _qt + run, trois revues indépendantes, 295 tests)** :
- Module `src/summary_quadtree.py` : S∘R = id bit-à-bit ARBRE INCLUS aux 7 budgets × 3 champs
  (vérifié en revue par exécution indépendante) ; comptabilité bits d'arbre exacte (budget 2048
  rempli à 2048 exactement ; refus au bord 402 > 400) ; glouton (gain, y, x) déterministe.
  Arbitrage consigné (validé) : les masses 16×16 ne sont PAS restituables vs l'original sous
  troncature (feuille englobante = redistribution uniforme, contre-exemple en test exécutable) —
  conservation TOTALE 1e-12 garantie, fidélité spatiale jugée par le readout (design).
- Harnais `run_arcA_measure_qt.py`/`run_arcA_verdict_qt.py` (famille 1 intouchée), run complet
  déterministe, violation vérifiée EN DONNÉES BRUTES par la revue.

**GATE DES CONTRÔLES : VIOLATION — lecture fine (les faits, tous vérifiés)** :
- **ferm (plomberie) : CONFORME 0/40** — Δχ = 0.0 EXACT aux 7 budgets × 2L × 5 seeds : la
  projection quadtree tient en conditions réelles, le plancher instrument est bit-à-bit nul.
- **shuf : VIOLATION 7/40 — TOUTES au budget plafond 2048, JND 3–5 % seulement** (2 seeds/5 à
  L=10, 1/5 à L=80). Aux budgets ≤ 400 : max(M-A1, M-A2) ∈ [0.52, 0.91] sur les MÊMES cellules —
  jamais fermé. **Sous le cap, le discriminant est intact.**
- Mécanique respectée : verdict §A3 v2 SCELLÉ, aucune des trois issues sélectionnée.

**Diagnostic (structurel, pas un bug)** : à 2048 float-éq pour 4096 cellules (~50 % de la
résolution), même un champ PERMUTÉ devient partiellement approximable — la « fermeture » à ce
budget ne discrimine plus structure et bruit. L'attendu gravé « shuf : k\* = ∞ PARTOUT » couvrait
les budgets diagnostics hors-cap ; la propriété qu'il protège (l'anti-trivialité) vit SOUS le cap.
**Troisième erreur de portée d'attendu** (ferm k\*=81 famille 1 ; grilles §A0/§A3 ; maintenant
shuf aux budgets diagnostics) — la règle de forme se précise : **un attendu de contrôle se grave
avec la PORTÉE de la propriété qu'il garde, pas sur toute la grille par défaut.**

**Corollaire gravé (vaut quel que soit l'arbitrage)** : les fermetures HORS-CAP (1024, 2048) ne
discriminent pas structure/bruit — toute cellule k\* ∈ {1024, 2048} de la surface v2 se lit avec
ce caveat. Sous le cap, la discrimination tient (shuf n'y ferme jamais).

**Sous scellé (audit, non-verdict, à ne pas surclamer)** : la surface v2 contient les PREMIÈRES
fermetures SOUS CAP de toute la manche — k\* médian = 400 (et 256 à L=10) à JND 4–5 %, à TOUS
les L y compris 80. La lecture formelle n'existera qu'après arbitrage du gate.

**Fork remonté à Romain (portée de l'attendu shuf)** :
(1) **Requalifier l'attendu à sa portée fonctionnelle** : « shuf : k\* = ∞ SOUS LE CAP
(budgets ≤ 400) » ; les budgets hors-cap restent des diagnostics (clause 2) EXCLUS du gate, et
portent le caveat gravé ci-dessus. Re-LECTURE du gate sur les mesures existantes (aucun re-run,
aucun seuil §A3 touché) → si conforme, verdict lisible. C'est un amendement d'attendu APRÈS
données — jamais silencieux, d'où ce fork ; sa base principielle est identique aux
requalifications ferm : erreur de portée, pas de complaisance (le discriminant sous-cap n'a
jamais failli).
(2) Retirer {1024, 2048} de la grille — plus dur, perd la queue débit-distorsion que la clause 4
destine à l'Arc C.
(3) Suspendre.
**Recommandation : (1).**

**Commits pocPhysicator** : 18e52b9 (module), 2ee1120 (measure_qt), 0a81832 (verdict_qt),
a86b929 (run). Contrat : a8ed659 (famille 2 + clauses).

### 2026-07-04 — Arbitrage portée-shuf : option (1) ENDOSSÉE, avec le test de défendabilité gravé AVANT la re-lecture, deux clauses de forme, et TROIS GARDES DE LECTURE posées avant de relire la table

**Test de défendabilité (gravé — pourquoi cet amendement post-données n'est pas de la
complaisance)** : un amendement post-données est de la complaisance si la PROPRIÉTÉ gardée
change ; c'est une correction de PORTÉE si la propriété est intacte et que seule l'étendue de la
grille était écrite trop large. Ici la propriété que shuf garde est l'anti-trivialité du
verdict : « aucun k\* sous le cap ne peut être atteint par un champ sans structure ». Cette
propriété vit sous le cap PAR CONSTRUCTION — c'est le cap qui définit ce que le verdict a le
droit de lire. Les budgets 1024/2048 sont des diagnostics de la clause 4, exclus du verdict
depuis leur gravure. Qu'un champ permuté ferme à ~50 % de la résolution n'est pas une fuite de
l'instrument : **c'est un théorème d'échantillonnage** — à ce budget, tout histogramme se
rapproche, et c'est précisément pour ça que le cap existe. Point décisif pour l'intégrité : la
requalification NE PEUT PAS fabriquer le verdict, parce qu'elle ne touche aucune cellule que le
verdict lit — les 40/40 sous cap sont conformes (0.52–0.91 sur les cellules mêmes qui posent
problème au plafond). L'option 2 serait la sur-correction classique : amputer la queue
débit-distorsion que l'Arc C attend, pour préserver un attendu mal écrit. **On garde les données,
on corrige la phrase.**

**Amendement gravé** : attendu shuf requalifié — « k\* = ∞ SOUS LE CAP » ; opérationnellement :
cellule shuf CONFORME ssi k\* > 409.6 (aucune fermeture à budget ≤ 400) ; VIOLATION ssi
k\* ≤ 400. Ferm inchangé (k\* = 32 exact, 7 budgets). Re-LECTURE sur les mesures existantes —
zéro re-run, zéro seuil §A3 touché.

**Clause de forme 1 (l'énoncé complet, les deux moitiés gravées ensemble)** : toute grille
inclut une cellule « l'instrument ne peut pas répondre » (leçon des scellés) ET tout attendu de
contrôle se grave avec la PORTÉE de la propriété qu'il garde (leçon d'aujourd'hui). Deux moitiés
d'une même discipline — pré-écrire l'espace des issues Y COMPRIS les issues de l'instrument
lui-même. Trois erreurs de portée en un arc : un pattern de rédaction, plus un accident.

**Clause de forme 2 (l'étoile Arc C)** : le caveat non-discriminant sur toute cellule
k\* ∈ {1024, 2048} SUIT LA SURFACE dans le livrable Arc C — étoile permanente « budget
non-discriminant vs bruit » portée par la sérialisation et la figure, pas seulement par le
journal. Sinon dans trois semaines quelqu'un lit la surface sans le journal.

**TROIS GARDES DE LECTURE (posées AVANT la re-lecture de la table — dernière fenêtre où elles
peuvent être écrites sans être suspectes)** :
1. **La clause de sensibilité §A3 est bloquante** et JND = 2 % est vraisemblablement encore
   infermable sous cap : le global attendu est « cellule 1 par-JND partiel + surface portée à
   l'Arc C », PAS cellule 1 tout court. C'est le cas pré-écrit de la clause 4 — verdicts par-JND
   divergents mais propres → global INDÉTERMINÉ, surface à l'Arc C comme l'objet que son pin
   résout. La manche ne se clôt pas aujourd'hui ; elle se CONDITIONNE proprement à l'Arc C —
   état final légitime et prévu.
2. **La structure en L se lit dans les deux sens avant toute pente** : 256 à L=10 contre 400
   ailleurs est AUSSI une croissance de k\* avec l'histoire, du même ordre que celle vue hors-cap
   au run précédent (1041 qui casse à L=80). Si la pente §A3 la déclare compatible-zéro, c'est
   PASS-par-la-règle et on le prononce ; mais la cohérence des deux indices de croissance se
   consigne comme INDICE CONVERGENT au journal, non comme verdict. Ni enterré, ni surclamé.
3. **Rappel de portée invariant** : cellule 1 sur ce substrat court-circuite la manche 2 SUR CE
   SUBSTRAT UNIQUEMENT (§A0/§A5, réaffirmé §A6). Le claim existentiel à l'échelle du but
   (obligations étagées, densité d'observation) reste ouvert quoi qu'il arrive dans cette table.

### 2026-07-04 — MANCHE 1, PREMIER VERDICT PRONONÇABLE : PASS à JND 4 % et 5 % (k\* = 400 float-éq ≤ cap, plat en L), INDÉTERMINÉ à 2–3 % — GLOBAL INDÉTERMINÉ (clause de sensibilité) → la surface k\*(L, JND) est portée à l'Arc C. Les trois gardes de lecture appliquées

**Re-lecture (amendement fa54d20 appliqué, re-revue indépendante)** : gate re-lu **CONFORME
0/80** (ferm 0/40 inchangé ; shuf 0/40 sous la portée requalifiée — les 7 ex-violations, toutes
k\* = 2048, hors du domaine que le gate protège). Vérification d'intégrité décisive : les
k\*/pentes/IC du verdict re-généré sont **bit-identiques** à l'audit pré-amendement — le
déverrouillage vient de la seule règle de lecture du gate, pas d'un recalcul. 301 tests verts.

**Surface k\*(L, JND) — bras v2, médianes 5 seeds (★ = cellule non-discriminante vs bruit,
étoile portée par le JSON et la figure — clause de forme 2)** :

| L | JND=2 % | JND=3 % | JND=4 % | JND=5 % |
|---|---|---|---|---|
| 10 | 1024★ | 400 | 400 | **256** |
| 20 | 1024★ | 400 | 400 | 400 |
| 40 | 1024★ | 1024★ | 400 | 400 |
| 80 | 2048★ | 1024★ | **400** | **400** |

**Verdict §A3 (mécanique)** : pentes L∈{40,80} + IC 95 % bootstrap : 2 % : 25.6 [-25.6, 25.6] ;
3 % : 0 [-25.6, 41.2] ; 4 % : 0 [0, 15.6] ; 5 % : 0 [0, 3.6].
- **JND 4 % : PASS** (k\*(80) = 400 ≤ 409.6 ET IC ∋ 0). **JND 5 % : PASS** (idem, k\*(10) = 256).
- JND 2 % : INDÉTERMINÉ (k\*(80) = 2048 > cap ; pente non concluante). JND 3 % : INDÉTERMINÉ
  (k\*(80) = 1024 > cap).
- **GLOBAL : INDÉTERMINÉ** (clause de sensibilité §A3, bloquante — non stable sur la plage).
- **Issue mappée (clause 4, verbatim)** : « global INDÉTERMINÉ — surface k\*(L, JND) portée à
  l'Arc C ».

**Les trois gardes appliquées (gravées AVANT cette lecture, fa54d20)** :
1. **« Cellule 1 par-JND partiel + surface à l'Arc C », PAS cellule 1 tout court.** Le global
   attendu par la garde est advenu à l'identique. Sens du PASS partiel, portée exacte : sur v2,
   à JND ∈ {4, 5} %, un résumé quadtree de ≤ 400 float-éq (9.8 % du champ) tient le readout
   albedo sous JND, instantanément (M-A1) ET sous dynamique (M-A2), à TOUS les L ∈ {10..80},
   avec k\*(L) plat — le claim §A2 tient à ces JND sur ce substrat. À 2–3 %, la question reste
   ouverte sous le cap (et les cellules hors-cap y sont non-discriminantes ★). **La manche ne se
   clôt pas : elle se CONDITIONNE à l'Arc C** — le pin JND réel choisira la courbe opérante sur
   la surface étoilée. État final légitime et prévu.
2. **PASS-par-la-règle prononcé ; l'indice de croissance consigné comme indice, ni enterré ni
   surclamé** : 256 (L=10) → 400 (L≥20) à JND 5 % est une croissance de k\* avec l'histoire, du
   même ordre que l'indice hors-cap de la famille 1 (1041 qui casse à L=80). Noter aussi que les
   IC des pentes PASS sont [0, 15.6] et [0, 3.6] — bornés à zéro PAR LE BAS (k\* quantifié ne
   décroît pas) : la compatibilité-zéro y est la forme la plus faible. **Indice convergent d'une
   croissance douce de k\* en L, sous le seuil de résolution de la grille actuelle** — consigné
   comme indice au journal, PAS comme verdict. L'Arc C et toute manche 2 doivent le connaître.
3. **Portée invariante** : le PASS partiel court-circuite la manche 2 SUR CE SUBSTRAT UNIQUEMENT
   et AUX JND où il tient (§A0/§A5/§A6). Le claim existentiel à l'échelle du but (obligations
   étagées, densité d'observation croissante/adversariale) reste ouvert.

**État de l'Arc A à la clôture de cette entrée** : manche 1 = verdict prononcé, conditionnée à
l'Arc C via la surface étoilée (courbe débit-distorsion du champ persistant). La règle de
dernière famille n'a pas eu à s'appliquer (pas de second INDÉTERMINÉ-CAPACITÉ). Le bras nul v2,
les contrôles (plomberie + discriminant borné au cap), le pipeline complet (histoires, mesures,
k\*, gate, verdict) et les deux familles sont commités et reproductibles.

**Commits pocPhysicator** : 49189d1 (gate requalifié + étoile), 11091a0 (verdict re-généré).
Contrat : fa54d20 (amendement + gardes). Historique complet de la manche : 04ac96f → 11091a0.

---

## Addendum Arc C — Pin du référent perceptuel (JND) : §C0–§C5 (gravé le 2026-07-05, AVANT toute mesure et avant toute lecture bibliographique)

> Arbitrage Romain : Arc-C-d'abord (manche 2 différée derrière le pin — ses seuils M-A
> hériteraient sinon du placeholder, la faute originelle de l'arc réinjectée dans le claim
> existentiel). Le pin est le goulot : quatre consommateurs gatés attendent.

### §C0 — Décision de niveau BUT : mapping obligations étagées → régimes de mesure
> **PROPOSÉ — point d'arrêt Task 0, validation Romain requise avant toute mesure.**

Les obligations étagées (§A6) impliquent des JND **étagés**. Mapping proposé :
- **Étage 2 (émissions : captures, revisites conjointes, comparaisons entre joueurs)** →
  régime **SÉVÈRE** : ABX simultané, stimuli côte-à-côte, inspection libre sans limite de
  temps — le pire cas réel (une capture comparée pixel à pixel au monde re-dérivé).
- **Étage 3 (revisite libre, mémoire du joueur)** → régime **LAXISTE** : ABX séquentiel,
  délai de rétention D = 5 s + masque bruité entre présentations, exposition limitée (2 s).
  **Borne sévère assumée de l'étage 3 réel** (la vraie revisite se compte en heures/jours,
  sans référence) : tout PASS à JND_lax vaut a fortiori — le conservatisme est du bon côté.
- Le pin livre **deux scalaires par axe** (JND_sev, JND_lax), chacun avec IC.
- Lecture double de la surface k\* : la ligne sévère dit ce que les ÉMISSIONS coûtent ;
  la ligne laxiste ce que la REVISITE LIBRE coûte. **Le claim §A2 (fermeture du couple
  résumé+régénérateur) se lit à JND_sev** — c'est l'obligation d'émission qu'il garantit.
  L'écart sev/lax, s'il est grand, pré-annonce un stockage à deux étages (régions jamais
  émises = résumés plus maigres) — noté pour la spec, aucune décision ici.

### §C1 — Portée et statut du pin (gravé)

- Le pin est un **référent d'ingénierie** : n = 1 (Romain), sujet informé, écran et
  conditions fixés. Ce n'est PAS un invariant psychophysique universel. Biais nommés et
  gardés (§C5), pas éliminés.
- **Deux axes** : (α) **spatial** — Δχ-albedo, l'instrument R1 existant (bande porteuse
  dominante mesurée : 4–7 cycles/domaine) ; (β) **temporel** — déficit de largeur spectrale
  σ_ω des grandes structures (l'axe de l'étage 1 Boussinesq ; balayage pré-conçu [−10, −60] %).
- **Conditions d'affichage gravées en Task 0 avant mesure** : distance d'observation, taille
  affichée du domaine (→ taille angulaire), colormap = le rendu réel (`render.py`),
  luminance/gamma de l'écran, durée d'exposition par régime. Tout changement de condition =
  nouvelle mesure consignée, jamais une retouche.
- Le pin **remplace le placeholder [2, 5] %** dans tout usage futur (seuils M-A de la
  manche 2 inclus). Les verdicts passés ne sont jamais relus rétroactivement ; les verdicts
  CONDITIONNELS (surface k\*, cellule kx=1@1 %) se lisent au pin — c'est leur définition.

### §C2 — C-1 : encadrement bibliographique (jamais un pin seul)

**Exécutant : la session critique (chat), PAS Claude Code.** Procédure de conversion
pré-écrite, gravée AVANT d'ouvrir le moindre papier (le lecteur connaît la surface k\* ;
chaque degré de liberté de conversion est un knob) :
1. Fréquence : bande porteuse (4–7 cycles/domaine) × taille angulaire gravée → cycles/degré.
2. Adaptation : luminance moyenne du rendu → niveau photopique (choix nommé n°1 : CSF de
   référence, une seule, citée).
3. Seuil CSF (contraste Michelson) → Δχ-RMS : ratio RMS/Michelson **mesuré sur NOS stimuli**
   (jamais supposé) (choix nommé n°2 : taille de patch / sommation spatiale).
**Budget de choix arbitraires ≤ 3, nommés.** Au-delà → C-1 rétrogradé à ENCADREMENT d'ordre
de grandeur, C-2 tranche. Lectures pré-écrites : intervalle [lo, hi] entièrement ≥ 4 % →
présomption côté PASS, C-2 confirme ; entièrement ≤ 2 % → présomption sévère ; chevauchant →
aucun verdict. **C-1 ne pin jamais seul** (gratings ≠ heatmap texturée).
Axe temporel : transfert littérature (flicker/discrimination temporelle) attendu MAUVAIS —
C-1-temporel = encadrement au mieux, gravé d'avance.

### §C3 — C-2 : harnais ABX sur stimuli réels (le pin)

- **Stimuli spatiaux** : paires (vrai, dégradé) issues des npz commités de la manche 1 —
  aucun stimulus synthétique, aucun re-run de simulation. Axe de stimulation continu :
  mélange linéaire `stim(t) = (1−t)·vrai + t·régénéré(budget)`, t ∈ [0, 1] ; **Δχ(t) est
  MESURÉ par l'instrument R1 sur chaque stimulus présenté** (pas supposé linéaire).
- **Stimuli temporels** : films projetés Boussinesq Ra = 10⁷ à σ_ω(kx=1) réduit
  paramétriquement de −10 à −60 % (le balayage pré-conçu). Archives perdues (reboot) →
  régénération nécessaire ; coût plafonné (§Partie 2). L'axe spatial passe EN PREMIER
  (stimuli déjà commités, zéro régénération).
- **Procédure** : ABX (réponse au percept, pas de oui/non → pas de biais de critère),
  staircase adaptatif 2-down-1-up (convergence ~70.7 % correct), seuil = moyenne des 6
  derniers renversements, exprimé en Δχ mesuré (spatial) ou en % de réduction σ_ω (temporel).
  **≥ 3 staircases par condition** (axe × régime), ordre des essais non révélé au sujet.
- **Sorties** : JND_sev^spat, JND_lax^spat, JND_sev^temp, JND_lax^temp — chacun valeur + IC
  (dispersion inter-staircases).

### §C4 — Lectures pré-écrites des quatre consommateurs (gravées avant le premier essai)

**1. Surface k\*(L, JND) — manche 1.** Lecture à JND_sev^spat, sur l'IC ENTIER :
- IC entier ≥ 4 % → **cellule-1-candidate**. GARDE GRAVÉE (avant le pin, dernière fenêtre
  non suspecte) : le prononcé définitif exige l'**extension 16L₀** (L = 160, l'option (a)
  de §A3, une seule fois) — les IC de pente PASS sont bornés à zéro par le bas (k\* quantifié
  ne décroît pas), la compatibilité-zéro y est sa forme la plus faible ; L = 160 donne à la
  pente une vraie chance de casser. Tient → cellule 1, portée §A5 (ce substrat, ce JND).
- IC entier ≤ 3 % → la ligne opérante est infermable/non-discriminante → manche 1 =
  **NON-DÉMONTRÉ au pin** (la règle de dernière famille s'applique : pas de 3ᵉ famille sans
  décision nommée) → hypothèse ledger, la dépense suivante est la manche 2.
- IC à cheval sur la frontière 3/4 % → INDÉTERMINÉ maintenu ; UNE session C-2
  supplémentaire autorisée pour resserrer l'IC (une seule, gravée) ; sinon porté tel quel.
**2. σ_ω / W1 — lecture en ÉTAU, l'économie pré-écrite.** La psychométrique temporelle se
lit aux DEUX bornes de l'incertitude W0 : −10 % (W-loi) et −19 % (W-plancher, le chiffre
Arc V). Trois issues :
- les deux sub-JND_t → déficit imperceptible quel que soit loi/plancher → le descend est au
  chômage à Ra = 10⁷ sur cet axe → **W1 MORT** (l'achat 19.4 h s'annule), frontière fovéa-z
  simplifiée d'autant.
- les deux supra-JND_t → déficit perceptible même à l'estimation optimiste → job du descend
  confirmé perceptuellement → **W1 MORT AUSSI** (le 3ᵉ point ne changerait pas le verdict).
- seuil DANS [−19, −10] → W1 redevient décisionnel → re-arbitrage de l'achat (options W0),
  avec une justification enfin réelle.
Deux issues sur trois tuent une dépense de 19.4 h : c'est l'économie que l'ordre
Arc-C-d'abord achète.
**3. Cellule grise kx=1 @ JND = 1 %.** JND_sev^spat ≥ 2 % → moot (gravé d'avance) ; < 2 % →
relecture des données existantes au pin (gratuit). Portée : le pin spatial est mesuré sur
l'albedo ; le transfert aux films de luminance Boussinesq est une hypothèse NOMMÉE — si les
sessions temporelles ont lieu de toute façon, une staircase spatiale sur stimuli Boussinesq
(coût marginal) remplace l'hypothèse par une mesure.
**4. Gate fovéa-z.** S'ouvre sur : manche 1 lisible-au-pin (consommateur 1) + pin (cet arc)
+ W1 résolu-ou-mort (consommateur 2). Pré-écrit : dans 2 des 3 issues du consommateur 2,
le gate n'attend plus que le consommateur 1.

### §C5 — Gardes anti-fabrication (bloquantes)

1. **Le sujet connaît la surface k\*** — biais nommé. Parades structurelles : ABX (se
   tromper volontairement est la seule triche possible, et elle est détectable), staircase
   adaptatif à niveau non révélé, axe t continu (le sujet ignore le Δχ de l'essai courant).
2. **Catch trials** à Δχ fort insérés aléatoirement (~10 % des essais) : réussite ≥ 90 %
   sinon session INVALIDE. Deux sessions invalides consécutives → STOP, remonter (fatigue
   ou protocole), jamais de moyenne complaisante.
3. **Cohérence inter-staircases** : dispersion ≤ 30 % de la moyenne, sinon la condition est
   INDÉTERMINÉE (pas de cherry-pick de staircase).
4. **Lecture des consommateurs sur l'IC entier du pin**, jamais au point central (leçon
   T1.5-e : le verdict qui bascule dans l'IC est INDÉTERMINÉ).
5. Toute grille de ce document inclut sa cellule « l'instrument ne peut pas répondre »
   (sessions invalides, staircases incohérentes) ; tout attendu de contrôle est gravé avec
   la **portée de la propriété qu'il garde** (règle de forme, deux moitiés, désormais
   standard).
6. Ordre gravé : validation C-0 → conditions d'affichage → C-1 (chat) → C-2 spatial →
   C-2 temporel → lectures §C4. Aucune lecture de consommateur avant la fin des mesures de
   l'axe concerné.

### §C6 — Task 0 : validation C-0 + amendement géométrie + conditions + fork temporel (gravé le 2026-07-05, AVANT tout stimulus)

**Concession de niveau contrat (Romain).** Le cap « régénération plafonnée à 2 h CPU » du
prompt de mission (Partie 2) **présupposait un solveur Boussinesq présent sur la machine**. Il
ne l'est pas (grep exhaustif src/scripts/docs/historique : le code convection n'a jamais été
porté ici — journal seul, comme le sédiment et les archives Arc V). C'est la troisième fois
qu'un artefact load-bearing manque à l'inventaire ; cette fois l'hypothèse silencieuse était
**dans le document de mission**, et la reconnaissance de l'exécutant (Task 0) l'a attrapée
avant qu'elle coûte. Gravé comme fait.

**C-0 — VALIDÉ, avec amendement de géométrie (bloquant, il déplace le pin).**

Le reste de §C0 est co-signé tel quel : étage 2 → **sévère** (ABX simultané, inspection libre) ;
étage 3 → **laxiste** (ABX séquentiel, D = 5 s, masque bruité, 2 s d'exposition, borne sévère
assumée de l'étage 3 réel) ; **le claim §A2 se lit à JND_sev**. L'amendement ne porte que sur
la géométrie d'affichage — qui n'est PAS un détail de confort mais un **knob qui déplace le
pin** :

- **Le knob, nommé.** La géométrie « naturelle » proposée en Task 0 (domaine 64×64 à ~10 cm de
  côté, vu à ~60 cm ≈ 9.5° d'angle) place la bande porteuse (4–7 cycles/domaine) à **0.4–0.7
  cycles/degré** — très en dessous du pic de sensibilité de l'œil (~2–5 c/deg). Conséquence
  DIRECTIONNELLE : à cette géométrie on mesure un JND **plus laxiste** que ce qu'un observateur
  au pic percevrait — c.-à-d. le réglage qui rend le **PASS de la surface k\* le plus facile**.
  Personne ne l'a choisi pour ça, mais c'est ce qu'il fait, et quiconque connaît la surface
  sait que 4 % = PASS. Le knob se neutralise, il ne se laisse pas jouer.
- **Décision gravée (adversariale-mais-bornée).** Une **seule géométrie pour les deux régimes**,
  calculée pour placer la bande porteuse **au pic CSF (~3 c/deg)** — soit un domaine affiché à
  **~1.8–2° d'angle visuel** (l'exécutant calcule la taille en pixels depuis le matériel réel,
  cf. conditions ci-dessous).
- **Bonus non accidentel** : à cette taille, une cellule ≈ **1.7 minute d'arc**, sous l'acuité —
  le stimulus est une **texture, pas une grille pixel-peepable** → confound de comptage de
  pixels écarté.
- **Direction du conservatisme, pré-écrite** : pin au pic = **borne sévère** ; **tout PASS à ce
  pin vaut a fortiori à toute géométrie plus lâche.**
- **Zoom libre EXCLU de l'opérationnalisation de l'étage 2 (décision de portée, gravée — pas
  glissée).** Le zoom illimité (le pire cas littéral d'une capture comparée pixel à pixel)
  effondrerait le pin vers le **plancher de quantification de la colormap (~0.4 %)** et rendrait
  à peu près tout infermable. L'exclure est défendable — **la comparaison naturelle de captures
  se fait à taille naturelle** — mais c'est un choix de portée qui appartient au contrat, donc
  il est écrit ici et pas supposé. Résiduel nommé.
- **Condition diagnostique optionnelle.** La géométrie 9.5° peut rester comme **condition
  diagnostique NON-VERDICTALE** (si le budget d'essais le permet) — **jamais** comme ligne de
  lecture d'un consommateur.

**Conditions d'affichage.** Validées en l'état : **colormap = viridis, vmin = 0, vmax = 1,
origin = lower** (`render.py`/`io_utils.py`) = le rendu réel, **zéro choix neuf**. Gamma **sRGB
supposé** + **luminosité de l'écran fixée et notée** = **choix nommé n°3** du budget C-1. Le
**budget C-1 est donc CONSOMMÉ** (CSF de référence, taille de patch, gamma) — plus aucun choix
libre disponible, exactement là où on voulait être. **En attente de Romain (ne peut venir que
de lui, entre dans la conversion c/deg)** : modèle/taille/résolution de l'écran des sessions et
**distance mesurée** (mesurée, pas estimée). Ces deux chiffres closent la taille-pixels de la
géométrie ~1.8–2° ; ils ne bloquent QUE Task 3 (sessions), pas Tasks 1–2.

**Axe temporel — (c) backup en parallèle de (b) spatial-d'abord ; fork nommé, tranché avant
Task 4 seulement.** Raison de fond (statut, pas confort) : les bornes de l'étau §C4-2
**[−19, −10] %** sont des **propriétés de l'instrument d'origine** (solveur, κ=4, projection) ;
un JND mesuré sur des films **reconstruits** porte une **hypothèse de transfert** vers ces
chiffres. Aujourd'hui ne se décide QUE : spatial démarre, backup se cherche. Si (c) échoue, le
fork se nomme entre :
- **(a)** reconstruction Boussinesq, **hypothèse de transfert gravée** ;
- **(a′)** films synthétiques à σ_ω paramétrique — ~10× moins cher, **hypothèse de transfert
  pire** (la structure de panache compte probablement dans la perception de largeur de bande) ;
- **(d)** porter l'étau **non résolu** → W1 **mort-par-défaut**, gate fovéa-z suspendu au **seul
  consommateur 1**.
Aucune de ces branches ne se décide ce jour.

### §C7 — Refonte géométrie « cible-jeu » + levée du point d'arrêt Task 0 (VALIDÉ Romain, 2026-07-05)

**Deux corrections de Romain au point d'arrêt Task 0**, qui corrigent l'**appareil de mesure** et
la **portée** — aucun seuil touché : **(1)** la cible est un **jeu** → ne pas baser le pin sur un
écran fixé, il doit valoir *quel que soit l'écran* ; **(2)** on ne peut **pas présumer** ce que
feront les joueurs. Elles **amendent §C6 sur deux résiduels** — la « taille en pixels depuis le
matériel » (dépendance à un écran) et l'**exclusion du zoom** (présomption de comportement).
§C6 reste au journal, non réécrit (append-only) ; §C7 le supersède sur ces deux points.

**1. Auto-calibration — pin ancré en cycles/degré, garantie inter-écrans par borne.**
Le pin n'est jamais « l'écran de Romain » : c'est un scalaire **ancré en c/deg**. Garantie
inter-écrans par **borne de sensibilité**, pas par moyenne d'écrans :
- Le pin est mesuré à la **géométrie la plus sensible du régime texture** (porteuse au pic CSF
  ~3 c/deg). Tout joueur, tout écran, toute distance « comme prévu » voit à une sensibilité
  **≤** celle du pin → **un PASS au pin tient a fortiori pour tous**.
- Le **harnais s'auto-calibre à l'écran où il tourne** : au démarrage de session, longueur de
  référence mesurée à l'écran + distance d'observation mesurée → pixels-par-degré → le domaine
  est rendu à la taille angulaire cible sur *n'importe quel* écran. Ces deux chiffres sont un
  **input de calibration par session** (consignés comme **conditions de validité** de la
  session, avec repère physique de distance maintenu pendant les essais) — **plus** un paramètre
  d'écran gravé au contrat.

**2. Deux régimes bornés — plancher pixel-peep MESURÉ (pas argué).** On remplace l'exclusion du
zoom (§C6) par un **bornage aux deux extrêmes**, aucune présomption entre :
- **Régime texture** — pin primaire au pic CSF (ce que mesure C-2 spatial).
- **Régime pixel-peep** — quand le joueur zoome jusqu'à voir les cellules, l'artefact devient la
  **blockiness structurelle** du quadtree ; le référent s'effondre vers le **plancher de
  quantification colormap (~0.4 %)**. Ce plancher est **MESURÉ** (une staircase à l'échelle
  pixel, coût marginal une fois le harnais construit) — **pas** argué depuis la colormap (seule
  règle que l'arc n'a jamais enfreinte : mesuré ≠ supposé).
- **Deux lectures pré-écrites, quel que soit le côté où la mesure tombe** : plancher mesuré
  franchement sous le pin texture → les deux régimes sont nettement séparés, le fork zoom
  (pièce 4) tranche lequel borne ; plancher mesuré proche du pin texture → les régimes se
  recouvrent, le pin texture borne quasi partout et le fork zoom perd son mordant.

**3. Colmatage du trou de monotonie — plafond quantitatif du régime texture.** Entre pic-CSF et
pixel-peep, la sensibilité n'est pas garantie monotone en taille angulaire (à ~1.8–2°, une
cellule est déjà près de l'acuité — la couture de la pièce 2). On la ferme par un **plafond
calculé, pas affirmé** : le régime texture est borné par la taille angulaire où **une cellule =
le plafond d'acuité (borné en minutes d'arc)**, **calculé par le harnais depuis la calibration**.
Règle pré-écrite : **si le zoom max du jeu approche ce plafond, une staircase à la géométrie
max-autorisée remplace l'argument** (on mesure au bord au lieu de supposer que le pic-CSF borne
tout le régime).

**4. Fork zoom OUVERT — avec son étiquette de prix (gravée pour qu'il ne se referme pas par
oubli).** Le zoom max du jeu **sélectionne la borne liante** ; c'est une **décision de design du
jeu (à Romain), non due aujourd'hui**. Le prix de chaque côté est écrit :
- zoom plafonné à l'échelle texture → **pin texture borne** ;
- **zoom pixel illimité sur champs persistants** → référent au **plancher** → **obligations de
  re-dérivation quasi exactes** sur l'émis-en-zoom → **stockage grade-ledger** pour ces régions
  (rejoint la note §C0 : régions jamais émises = résumés plus maigres ; ici l'inverse — régions
  zoomées-émises = résumés quasi exacts).
- **Conséquence load-bearing, non lissée** : la surface k\* de la manche 1 se lit à JND_sev
  **dans le régime texture uniquement**. Si le jeu autorise le zoom pixel, le référent liant de
  la lecture k\* devient le plancher, et à ce JND la manche 1 part très probablement
  **NON-DÉMONTRÉ**. **La politique de zoom du jeu est donc en amont du verdict de la manche 1.**

**Levée du point d'arrêt Task 0 (les deux items restants, VALIDÉS Romain).**
- **C-0 — mapping VALIDÉ** : étage 2 (émissions) → **sévère** (ABX simultané, inspection libre) ;
  étage 3 (revisite) → **laxiste** (ABX séquentiel, D = 5 s, masque bruité, 2 s d'exposition) ;
  **claim §A2 lu à JND_sev**. (§C0 n'est plus PROPOSÉ : validé.)
- **Conditions d'affichage** : colormap = viridis/vmin=0/vmax=1/origin=lower = rendu réel, **zéro
  choix neuf** ; gamma **sRGB supposé** + **luminosité notée par session** = choix nommé n°3 ;
  **budget C-1 consommé** (CSF réf, taille de patch, gamma). Les paramètres d'écran gravés sont
  remplacés par la **calibration par session** (pièce 1).
- **Axe temporel — (b) + (c) CONFIRMÉ** : spatial d'abord (stimuli commités, zéro régénération)
  pendant que Romain cherche le code Arc V sur backup ; fork **(a/a′/d) nommé (§C6), dû avant
  Task 4 seulement si (c) échoue**.
- **Point d'arrêt Task 0 entièrement levé.** La mesure procède. Question terminale inchangée :
  **JND_sev^spat avec son IC, et de quel côté de la frontière 3/4 % il tombe entier** (lecture
  du consommateur 1, §C4).

### §C8 — Arbitrage Task 2 → Task 3 (VALIDÉ Romain, 2026-07-05) : décision géométrie (i) + contingence géométrie-plafond ; axe de mesure Δχ + catch séparé ; ancre-famille = UN point ouvert

**Ce que la machine a fait.** La tension que §C7 pièce 3 anticipait au bout « zoom du jeu », le
harnais l'a localisée **dans la géométrie du pin lui-même** : sur un contenu 64×64 à porteuse
4–7 cyc/domaine, **pic-CSF (3 c/deg) et cellules sous-acuité sont incompatibles par arithmétique**
(cellule au pic-CSF = 1.73 arcmin = 1.73× le plafond d'acuité, quasi-invariant écran). Posée au
bon endroit AVANT la première session. Crédit revue : le **test adversarial règle-inversée** (ne
converge jamais / erreur 132 %) prouve que la tolérance 20 % du test-clé discrimine.

**Décision 1 — (i) ENDOSSÉ, raison plus forte que celle de l'exécutant.** Le cue de **bloc n'est
PAS un confound** pour le consommateur 1 — **c'est du signal**. Les stimuli C-2 sont les **sorties
réelles du couple résumé+régénérateur** sur les champs réels de la manche 1 ; il est gravé que
« les artefacts de bloc comptent comme différence perceptuelle réelle : le claim §A2 porte sur le
couple ». Le **pin opérationnel** — « à quel Δχ mesuré un humain distingue vrai de régénéré, quel
que soit le cue qui porte sa réponse » — est **exactement le référent que la surface k\* doit
lire, par construction, sur la même famille de stimuli**. Ré-étiquetage gravé : **« JND à la
géométrie la plus sensible comme-prévu, cues combinés »**. **(iii) REJETÉ fermement** : interpoler
l'affichage lisse **précisément la classe d'artefact que le régénérateur produit** — ce serait
maquiller le signal pour purifier l'étiquette. Étiquette honnête maintenue : le 1 arcmin est du
Snellen haute-fréquence ; la résolvabilité de bloc dans cette texture est probablement plus
grossière → la tension est **peut-être petite** (mesurable, cf. contingence).

**Contingence géométrie-plafond (gravée AVANT toute donnée — dernière fenêtre non suspecte).**
(i) est une **borne sévère, donc asymétrique** :
- **PASS** au pin combiné → vaut **a fortiori partout** — rien à ajouter.
- Mais un pin dont l'**IC ENTIER ≤ 3 %** basculerait la manche 1 en **NON-DÉMONTRÉ** et
  déclencherait la dépense manche 2 **sur un référent plus sévère que ce que le jeu exigera
  peut-être** — la géométrie du pin est **1.73× au-delà du plafond texture**, c.-à-d. dans la
  bande de zoom qu'une **politique texture-cappée interdirait**. **Donc, gravé** : si l'IC du pin
  sévère tombe **entier ≤ 3 %**, **UNE staircase à la géométrie-plafond (1.07°, cellules à
  l'acuité, porteuse 5.1 c/deg) est DUE avant tout prononcé NON-DÉMONTRÉ** — c'est le **référent
  liant sous zoom texture-cappé**, exigé par le **fork §C7-4**, PAS de la complaisance (la lecture
  qui tue une manche se fait au référent que le but impose). La **« staircase-au-bord » de §C7
  pièce 3 EST devenue cette contingence**. Si elle tire, l'**écart entre les deux staircases
  mesure la tension** au lieu qu'on en débatte.

**Décision 3 — PRINCIPE (VALIDÉ).** L'**axe de mesure = Δχ mesuré par stimulus, PAS le budget**.
Les budgets **256/400 sont les objets JUGÉS, pas l'axe de mesure** — les employer comme ancre
plafonnerait la plage à ~2.78 % (pire source) et **gonflerait le seuil par saturation** (finding
de la revue). Règle gravable : l'**ancre de dégradation = la régénération la plus crue** ; son
**plafond Δχ doit être ≥ 2× le haut de la plage JND plausible par source, sinon la source est
EXCLUE et NOMMÉE**. **BUDGET_CATCH sur une réserve SÉPARÉE** (revue M2) — un catch qui puise dans
l'axe de staircase perd sa marge « sans ambiguïté ».

**Ancre-FAMILLE — UN point OUVERT (contradiction interne relevée par le contrôleur, à trancher
avant Task 3).** Romain a nommé l'ancre **« 81 floats » (×2)**. Or **81 floats = famille 1**
(block-mean ℓ=3), **pas la famille quadtree** — et la surface k\* que le pin lit (verdict manche 1,
`6a82175`) est la **famille 2 QUADTREE** (budgets, cap 409.6). La **décision 1 rend le « même
famille de stimuli » load-bearing** ; une ancre famille-1 ferait que les blends près du JND
injectent un artefact **block-mean uniforme**, pas quadtree adaptatif → **mismatch avec les objets
jugés**. Deux résolutions cohérentes :
- **(α)** ancre **famille-1, 81 floats** (Δχ 0.27–0.85, marge max ; mais **artefact cross-famille**
  près du seuil, et requiert de câbler `regenerate(summarize(s,3))` — non présent dans le harnais
  Task 2, qui n'utilise que le quadtree) ;
- **(β)** **crudest QUADTREE = budget 32** (Δχ 0.155–0.644 par §3.3 du rapport Task 2, **MÊME
  famille** que les objets jugés ; la règle ≥2× **nomme** toute source dont l'ancre < 2× le
  haut-JND — au pire ~1 source des 20 si haut-JND ~8 % ; déjà supporté par le harnais).
**Recommandation contrôleur : (β)** — il honore la **décision 1** (même famille) ET la
**décision 3** (crude, axe = Δχ mesuré, règle ≥2× avec exclusions nommées), sans changement de
code. **En attente d'un mot de Romain** : c'est le **seul endroit où les décisions 1 et 3 se
tirent l'une contre l'autre**. (Décision 3 gravée en principe ; la famille de l'ancre attend ce
mot.)

**Décision 2 — non décidée ici.** Les deux chiffres d'écran (longueur de référence + distance
mesurée) sont des **inputs de calibration au début de chaque session**, consignés comme
**conditions de validité** — c'est le rôle de §C7 pièce 1.

**État.** Tasks 0–2 closes (harnais construit, revu, 382/382). **Task 3 gatée** sur : ancre-famille
(ci-dessus) résolue + Romain **sujet devant l'écran**. Forks d'aval inchangés : **temporel
(a/a′/d)** dû avant Task 4 si le backup Arc V ne rend rien ; **politique de zoom §C7-4** avec son
étiquette de prix. **Le prochain chiffre de l'arc sera le premier que la machine ne peut pas
produire seule.**

**CLÔTURE du point ouvert (VALIDÉ Romain, 2026-07-05) : (β) — ancre = quadtree budget 32.** Le
« 81 floats » était un lapsus inter-famille (Δχ de la famille block-mean puisés en mémoire parce
que les plus gros, sans vérifier que la famille du chiffre = la famille du verdict) — exactement
la classe d'erreur que la décision 1 venait de rendre interdite. **Décision 1 bat le chiffre
confortable** : ancre = la régénération la plus crue **de la famille jugée** (quadtree budget 32,
Δχ 0.155–0.644, zéro code neuf). Ce que (β) préserve : la **chaîne de légitimité du stimulus reste
d'un seul tenant** — champs réels → compresseur réel → régénérateur réel → famille du verdict — du
staircase jusqu'à la lecture de la surface k\*. (α) aurait acheté de la marge Δχ au prix d'une
**rupture de cette chaîne au seul endroit qui compte, près du seuil**.

**Trois conséquences opérationnelles — pliées dans le brief Task 3, NE ROUVRENT PAS le contrat** :
1. **L'exclusion nommée cesse d'être théorique** (ancre au pire 0.155 → la règle ≥ 2× mord dès que
   le haut-JND plausible dépasse ~7.75 %, ~1 source sur 20 à haut-JND ~8 %). Contrôle d'ancrage
   **par source, avant sa première présentation** ; source exclue = **nommée au log avec son Δχ
   d'ancre**, jamais retirée silencieusement. 20 − une poignée = échantillonnage sain ; **≥ 10
   exclues = résultat à REMONTER, pas à absorber**.
2. **Asymétrie de l'ancre, sens inconfortable** : budget-32 rend les catch **légèrement plus durs**
   sur les sources à Δχ d'ancre bas. Si le taux de catch < 90 % à cause de ça → **session INVALIDE
   quand même** (on ne recalibre PAS le critère de validité sur la difficulté de l'ancre après
   coup ; le catch atteste l'attention, pas la réussissabilité-par-construction). Trop dur
   structurellement → **STOP-remonter** ; option propre alors NOMMÉE : tirer les catch de la
   réserve séparée sur les **sources à ancre la plus forte** (décision à ce moment-là, pas
   maintenant).
3. **La contingence géométrie-plafond hérite du principe** : si elle tire (IC ≤ 3 %), sa staircase
   utilise la **même ancre budget-32, même famille, mêmes sources** — sinon l'écart entre les deux
   staircases, qui doit mesurer la **tension géométrique**, serait contaminé par un écart d'ancre.

### §C9 — Conditions de validité d'EXÉCUTION des sessions Task 3 (VALIDÉ Romain, 2026-07-05 — dernière fenêtre avant données irréversibles)

Trois conditions dues au contrat (§C1/§C6/§C7) mais absentes du protocole d'exécution —
verrouillées ici AVANT toute donnée :

1. **Durées d'exposition réelles LOGGÉES par essai (condition de validité du régime laxiste, PAS
   un confort).** Le laxiste empile des timers matplotlib (exposition 2 s, masque bruité, rétention
   5 s, défilement séquentiel) — notoirement approximatifs selon la machine et la charge. Un
   « 2 s » réalisé à 3,5 s **fausse le paramètre qui DÉFINIT le régime**, et on le découvrirait sur
   des staircases déjà consommées. Donc : chaque essai laxiste **consigne ses durées réalisées**
   (timestamps par phase X/masque/A/B suffisent). Le **pré-vol B-bis** (--regime laxiste, quelques
   essais, Ctrl-C) VÉRIFIE la séquence visible + les durées consignées **AVANT** la campagne ;
   mismatch grossier → STOP, corriger le mécanisme de timing avant toute staircase.
2. **Luminosité de l'écran FIXÉE à une valeur, NOTÉE au manifeste, jamais changée entre sessions**
   — c'est le **choix nommé n°3 du budget C-1** (§C6). Doit exister par écrit, pas par habitude.
3. **Repère physique de distance maintenu pendant les essais + éclairage ambiant stable.** En
   laxiste surtout : se pencher vers l'écran pendant la rétention = **zoom involontaire qui déplace
   le référent en pleine staircase** (le biais nommé §C7). Pas une session en plein jour et la
   contingence de nuit. Consignés au manifeste.

**Note d'hygiène (amorçage)** : le rodage synthétique à **θ = 0.025** (au lieu de 0.04) évite
d'afficher un pin pile à la frontière de décision juste avant la vraie session — structurellement
inoffensif en ABX (on ne pilote pas son seuil sans se tromper volontairement), mais zéro-reproche.

**Posture de session (la seule qui compte)** : répondre au **PERCEPT, pas à la stratégie**. Les
essais où l'on ne voit rien et où l'on **devine au hasard SONT le protocole** — deviner n'est pas
un échec, c'est ce que le staircase attend.

### §C10 — Contrat de sortie `pins_spatial.json` (schéma gravé le 2026-07-05, AVANT toute donnée humaine)

Le fichier `schemas/arcC-pins-spatial-v1.schema.json` (ce dépôt) est le **contrat de sortie** de
`pins_spatial.json`, gravé avant toute donnée. `run_arcC_pins.py` **DOIT valider sa sortie contre
ce schéma avant d'écrire** — un fichier non conforme n'est jamais écrit (fail loud, jamais un
référent malformé). Le fichier rapporte des **MESURES ; AUCUN verdict §C4** (l'`additionalProperties:
false` l'interdit structurellement — la lecture du consommateur 1 se fait au point d'arrêt, jamais
dans l'artefact). Unités : tout Δχ/JND en **fraction** (0.04 = 4 %).

Éléments load-bearing (élaborations gravées de §C4/§C5/§C9, pas de nouvelles décisions de fond) :
- **`controle_directionnel`** : attendu **JND_lax ≥ JND_sev** (le laxiste est moins sensible → seuil
  plus grand). Portée : violation = **IC disjoints dans le mauvais ordre** → `ic_disjoints_mauvais_ordre
  = true` → **STOP-remonter** (instrument/protocole suspect), pas une note. Des IC qui se
  chevauchent, même mal ordonnés, ne violent PAS (bruit, pas défaut) — règle de forme.
- **Branche à-cheval (IC combiné)** : IC primaire = **min/max des seuils** (§C4-1 ; à n=3, le
  bootstrap serait de la fausse précision). Si la cellule à-cheval s'active (§C4-1 : UNE session C-2
  supplémentaire, 6 seuils au total), `branche_combinee_active = true` et `ic_combine` = **mean±2SEM
  sur les 6 seuils** — présent UNIQUEMENT dans ce cas.
- **`timing_laxiste`** roll-up (§C9 pièce 1) : `exposition_moyenne_s` + `derive_max_pct` (> 25 % =
  régime suspect, session à remonter).
- **Statut ∈ {RESOLU, INDETERMINE, INVALIDE}**, `motif_statut` **obligatoire si ≠ RESOLU** (jamais
  un statut nu) : mapping du harnais — RESOLU ; dispersion inter-staircases > 0.30 → **INDETERMINE**
  (§C5-3) ; taux de catch < 0.90 ou 2 sessions invalides → **INVALIDE** (§C5-2, `jnd`/`ic` = null).
- **Provenance** : `commit_harnais` (SHA pocPhysicator), `date_session`, `calibration` embarquée
  (px_par_degre, taille_domaine_deg, cellule_arcmin, luminosité, conditions — §C7/§C9), et les
  **seuils BRUTS par staircase** (la branche à-cheval recalcule dessus).
- **`sujet: synthetique`** = rodage, **jamais commité dans `outputs/arcC/`** (garde anti-fabrication).

**Invariant gravé (VALIDÉ Romain, 2026-07-05 — l'implicite rendu explicite le jour où on le
découvre, pas le jour où il mord).** **`pins` = campagne complétée par construction (`minItems:3`
VOULU) ; l'arrêt protocolaire est un fait de MANIFESTE, jamais de RÉFÉRENT.** Portée de la
frontière : un régime qui a tourné ses **3** staircases et échoué est une campagne complétée dont
le résultat est l'invalidité → se rapporte dans le référent (statut `INVALIDE`, motif, `jnd/ic`
null). Un régime **coupé à 2** par le garde §C5 (2 sessions invalides consécutives → STOP,
remonter) est une campagne **inachevée** → **aucun pins** (fail-loud), le STOP vivant au manifeste
(registre des faits de session). Conséquences gravées sur le fail-loud :
- Le message d'erreur est un **aiguillage** : il porte le fait (« campagne arrêtée à N staircases
  par le garde 2-invalides §C5 »), le **régime** concerné, le **chemin du manifeste** où le STOP
  est consigné, et la **conduite** (« aucun pins ne peut être émis ; la reprise passe par le point
  d'arrêt — remonter »). Il **tire AVANT** la validation jsonschema (détection explicite du statut
  STOP au manifeste) — le « is too short » cryptique n'est plus atteignable sur ce chemin.
- **Le STOP reste un STOP** : le message ne suggère **aucune voie de complétion automatique** (pas
  de « relancer une 3ᵉ pour débloquer pins »). La seule sortie est l'**arbitrage humain** (fatigue ?
  protocole ? conditions ?). C'est POUR ÇA que pins doit refuser d'exister : un fichier émis « quand
  même » serait la pression silencieuse vers la 3ᵉ session de complaisance.
- **Espace des sorties fermé et testé** : les trois issues de campagne — complétée-résolue,
  complétée-invalide, arrêtée — ont chacune leur comportement verrouillé par un test e2e.

---

## §C11 — PIN SPATIAL MESURÉ (première donnée humaine) + lecture §C4 mécanique (2026-07-05)

**Le premier chiffre que la machine n'a pas fabriqué.** Session ABX, sujet **Romain (n=1, référent
d'ingénierie §C1)**, 2026-07-05T18:43, `base_seed=20260705`, harnais `commit 3d2a6ed`. Campagne
**complète** : 3 staircases × {sévère, laxiste}, toutes complètes ; **1/20 source exclue**
(`s103_L10`, Δχ d'ancre 0.155 < 0.16 — l'exclusion nommée prédite §C8). `pins_spatial.json`
**validé contre le contrat gravé §C10** (schéma draft-07).

**Validité §C5** : catch **12/12 = 100 %** (les deux régimes), dispersions inter-staircases **0.18**
(sévère) et **0.196** (laxiste), toutes deux ≤ 0.30 → **les deux régimes RESOLU**. `timing_laxiste`
dérive max **1.5 %** (≪ 25 %). Mesure propre.

**Pins (fraction ; 0.04 = 4 %)** :
- **JND_sev = 0.0733**, IC **[0.0603, 0.0867]** (émissions, étage 2 — la ligne du claim §A2).
- **JND_lax = 0.1154**, IC **[0.0937, 0.1387]** (revisite libre, étage 3).
- **Contrôle directionnel §C10** : `ic_disjoints_mauvais_ordre = false`. JND_lax > JND_sev et IC
  **disjoints dans le BON ordre** (laxiste entièrement au-dessus du sévère) — séparation nette et
  bien ordonnée, l'instrument mesure du réel.

### Lecture §C4 — mécanique, pré-écrite, appliquée telle quelle (aucune interprétation au-delà)

**Consommateur 1 — surface k\*(L, JND), manche 1.** Lu à **JND_sev**, sur l'**IC ENTIER** = [6.0 %,
8.7 %]. Règle §C4-1 : **IC entier ≥ 4 % → cellule-1-candidate.** Ici l'IC entier est même ≥ 5 % —
le point de grille le plus laxiste de la manche 1, déjà **PASS** (k\*(80) = 400 ≤ cap) ; k\* étant
non-croissant en JND, la surface lit **PASS a fortiori sur tout l'IC**. → **CELLULE-1-CANDIDATE.**
**GARDE §C4-1 (gravée avant le pin)** : le prononcé DÉFINITIF de la cellule 1 exige l'**extension
16L₀ (L = 160, option (a) §A3, une seule fois)** — les IC de pente PASS étaient bornés à zéro par
le bas ; L = 160 donne à la pente une vraie chance de casser. **Mission SÉPARÉE, gatée derrière ce
résultat, NON lancée ici.**

**Contingence géométrie-plafond (§C8).** Armée seulement si l'IC de JND_sev tombe **entier ≤ 3 %**.
IC = [6.0 %, 8.7 %] ≥ 4 % → **NE S'ARME PAS.** Le pin sévère tient tel quel ; pas de staircase à la
géométrie-plafond. (La « staircase-au-bord » de §C7 pièce 3 / §C8 D-4 reste au repos.)

**Consommateur 3 — cellule kx=1 @ JND = 1 %.** JND_sev ≥ 2 % → **MOOT** (gravé d'avance §C4-3).
7.3 % ≥ 2 %.

**Consommateurs 2 (W1/σ_ω) et 4 (gate fovéa-z).** Dépendent du **pin TEMPOREL**, **non mesuré**
(axe spatial seul ; l'axe temporel est le fork **(a/a′/d)** différé, dû avant Task 4 si le backup
Arc V ne rend rien). **En attente du pin temporel.** Le gate fovéa-z a son **consommateur 1 servi**
(cellule-1-candidate) mais reste suspendu au pin temporel + à W1.

### Portée (§A5) et blemish de record

- **Portée** : ce pin, ce substrat, cet observateur (n=1, §C1). La cellule-1-candidate est une
  lecture au pin sévère **sur ce substrat** ; le prononcé définitif est gaté sur 16L₀. Le **claim
  existentiel à l'échelle du but reste ouvert** (obligations étagées, densité d'observation
  croissante/adversariale).
- **Blemish de record NOMMÉ (§C9 pièce 3)** : le champ `conditions` du manifeste est resté le
  **placeholder « CONDITIONS »** (le repère de distance + éclairage réels n'ont pas été saisis). La
  mesure **n'est pas invalidée** (validité §C5 nette, timing conforme, géométrie 23.9 mm vérifiée à
  la règle), mais le **registre des conditions a un trou** — à amender au texte réel si le record
  doit être complet, sinon consigné comme tel. Un blemish de traçabilité, pas de mesure.

**Forks d'aval (décisions Romain, non tranchées ici)** : (1) lancer l'**extension 16L₀** pour
prononcer la cellule 1 définitivement ? (2) l'**axe temporel** (backup Arc V, sinon fork a/a′/d)
pour servir les consommateurs 2 & 4 ; (3) la **politique de zoom §C7-4** (avec son étiquette de
prix), qui reste en amont de la portée du claim.

---

## §C12 — C-1a : micro-mesures pour l'encadrement bibliographique (VALIDÉ Romain, 2026-07-05)

Deux mesures d'**analyse pure** (0 simulation, 0 RNG, 0 nouvelle donnée humaine ; entrées toutes
vérifiées présentes AVANT tout code — falsifiabilité levée) alimentent C-1 (encadrement biblio,
hors-code) et re-lisent la surface k\* au pin réel. Harnais pocPhysicator (branche `arc-c-pin-jnd`) :
Task 1 ratio = commits `061885a`+`98d30f0` ; Task 2 re-seuillage k\* = `27fdde7`+`6304426`. Chaque
task : implémenteur frais → revue spec+qualité → non-régression verrouillée. Suite 499/499.

### Le fait dur (garde d'honnêteté §C0, elle a fait son travail)

Le **contraste de Michelson de la porteuse (bande 4-7) est > 1 sur 19/19 sources**
(c_michelson ∈ [1.90, 4.82], médiane 2.29). **Michelson > 1 n'est PAS un bug de mesure** : c'est un
fait physique sur nos stimuli — l'albedo moyen est minuscule (~0.04) et la porteuse y déborde
(χ_rms > 1 pour 14/19 ; bande 4-7 porteuse ≥ 10 % AC pour les 19). Traduit : nos champs albedo **ne
sont pas des gratings sinusoïdaux de faible contraste** — ce sont des **textures à très fort
contraste local, quasi-binaires par endroits**. La CSF de la littérature est mesurée **au seuil de
détection, en petit contraste** — un régime où nos stimuli ne vivent pas. Le pont biblio que C-1
devait construire enjambe une rivière plus large que prévu.

### Décision ratio — TRANCHÉE (Romain) : **les deux**

- **Natif gardé intact** : le ratio χ_rms/C_michelson mesuré (`ratio_rms_michelson.json`, vrai
  médiane 0.477 IQR 0.093, régénéré-32 médiane 0.386 IQR 0.074) est la **mesure pré-enregistrée** —
  on ne réécrit pas une mesure gravée.
- **r0 ajouté en compagnon** : `r0 = 2·RMS/(peak-to-peak)` de la composante de bande = le ratio
  **scale-invariant**, celui du **régime petit-contraste**, donc le seul qui puisse légitimement
  convertir des seuils CSF. **Mandat propre** : passe **additive**, même harnais, **même sanity**
  (grating pur → r0 = 1/√2), **dispersion inter-sources rapportée** comme pour le natif. (Null-first
  respecté : l'exécutant a eu raison de ne PAS l'ajouter sans mandat ; le mandat est ici.)

### C-1 RÉTROGRADÉ — verdict pré-écrit AVANT toute lecture biblio (Romain)

Même avec r0, la conversion CSF→Δχ reste un pont **détection → discrimination-en-texture-fort-
contraste**. Le facteur de masquage supra-seuil pré-annoncé (~5–15×) **n'est pas un nombre que la
littérature grating donne proprement pour ce régime** (il vaut pour des masques modérés, pas pour
Michelson > 1). Le **régime fort-contraste consomme à lui seul plus d'incertitude que trois choix
nommés** → par la règle §C2 gravée (« budget de choix ≤ 3 ; au-delà → **C-1 rétrogradé à ordre de
grandeur, C-2 tranche** »), **C-1 est un encadrement d'ordre de grandeur, PAS un pont serré**.

- Ce n'est **pas un échec** : c'est que **C-2 est le pin**, et **C-1 confirme seulement qu'on est
  dans le bon ordre de grandeur** — la hiérarchie a toujours été gravée (« ne jamais pin depuis C-1
  seul » → ici **« ne jamais pin depuis C-1 tout court »**).
- **C-1 sera livré** sous cette forme : une **fourchette large** + une **vérification de cohérence
  de signe et d'échelle**, PAS une deuxième mesure indépendante du 7.3 %. La **jambe population
  existe, plus courte qu'annoncée**. Dit **maintenant** (non-suspect) plutôt qu'au retour.

### Mesure 2 — re-seuillage k\* au pin réel (`surface_kstar_aux_pins.json`) : un cadeau, chiffré

Re-lecture MÉCANIQUE (aucun verdict lu, aucune pente — la lecture est le fork 16L₀). Non-régressions
vertes : `sous_jnd` recalculé == stocké bit-à-bit (4 JND grille) ; k\*(80)@5 % = 400.

- **Au pin sévère (7.33 %), k\* est PLAT à 256 float-éq sur tout L ∈ {10..80} — PAS 400.** Soit : sur
  ce substrat, à ce pin, le diff committé d'un champ persistant **se compresse à 6,25 % de l'état
  fin (256/4096), plat sur 8× l'historique**. C'est le **chiffre plancher de la sauvegarde
  seed+diff**, descendu **d'un tiers sous ce que la manche 1 supposait** (400). Pour le game
  designer : le **coût de persistance par région observée est plus bas que le budget provisionné**.
- **Stockage-deux-étages §C0, désormais chiffré** : écart sévère/laxiste au pin = **−128 floats à
  L=10** (le laxiste ferme à 128 < 256), **0 au-delà** (L ∈ {20,40,80} convergent à 256). La
  distinction émissions/revisite **ne paie en stockage qu'aux courtes histoires** ; aux longues, les
  deux régimes convergent. **Architecturalement réel, quantitativement marginal sur ce substrat** —
  une **note pour la spec, pas un pilier**.

### Deux gardes sur la table k\* AVANT qu'elle serve d'entrée à 16L₀ (pré-écrites — armer l'œil, NE PAS trancher)

1. **La platitude vit sur une échelle à trois valeurs {128, 256, 400}.** « Plat à 256 » = « n'a
   bougé ni vers 128 ni vers 400 sur la plage testée » — plateau **réel mais grossièrement
   quantifié**, exactement la fragilité qui a motivé 16L₀. À L=160 la question est **binaire et
   nette** : reste-t-il à 256, ou **saute-t-il à 400 (la marche vers le mur)** ? La grille fine
   {32,64,128,256,400} gravée est ce qui rendra ce saut **lisible** s'il existe.
2. **Le pin d'entrée de 16L₀ est le pin sévère, IC ENTIER** (lecture à 6.0 / 7.3 / 8.7 %). La table
   le permet déjà : à **ic_bas (6.0 %), k\* = 256/256/400/400 — DÉJÀ non-plat** (il monte à L=40).
   C'est le **premier soupçon de croissance, à la borne basse de l'IC**. Pré-écrit (sans le lire
   comme verdict) : si 16L₀ confirme que **la borne basse monte pendant que le pin central reste
   plat**, la lecture cellule-1 devient **JND-dépendante à l'intérieur de l'IC** — et c'est
   précisément la **surface k\*(L, JND)** que §C4 destine à l'Arc C comme l'objet que le pin résout.
   **Ne pas trancher maintenant ; armer l'œil pour L=160.**

### Statut

**Arc C spatial se clôt** après (a) la gravure de r0 (mesure compagne, implémentation SDD immédiate)
et (b) la livraison de C-1 sous sa forme honnête (Romain, hors-code). Restent **à Romain, non dus ce
soir** : l'**extension 16L₀** (prononcé cellule 1) et l'**axe temporel** (W1 / gate fovéa-z). Le pin
temporel reste non mesuré. « Le ratio r0, tu le lances quand tu veux ; C-1 t'attend derrière lui. »

---

## §A12 — Extension 16L₀ : le prononcé de la cellule 1 (pré-enregistré 2026-07-05, AVANT run L=160)

> **Statut : gravé le 2026-07-05, AVANT le run L=160.** Déclenché par la cellule-1-candidate
> (manche 1, verdict INDÉTERMINÉ→PASS-partiel gravé `6a82175`) lue au pin Arc C sévère
> (JND 7.3 %, IC [6.0, 8.7], §C11/§C12). Exécute l'option (a) de §A3, gravée d'avance,
> UNE SEULE FOIS. Formalise les gardes déjà posées en §C12. Harnais pocPhysicator, branche
> `arc-c-pin-jnd` (descendante de `arc-a-etat-complet` — la seule qui réunit les histoires/mesures
> Arc A ET le pin Arc C). `run_history` gelé (blob `88e16e2d`, md5 `b564723a`), appelé avec
> `n_episodes=160`, non modifié.

### §A12 — Extension 16L₀ : le prononcé de la cellule 1 (figé avant run)

**Déclencheur (rappel) :** §A3 option (a) — « étendre à 16L₀ **une seule fois** » — était
pré-gravée pour le cas où la pente k*(L) PASS est trop faible pour trancher. Elle l'est :
au pin sévère, k*(L) est plat à 256 float-éq sur L ∈ {10,20,40,80}, mais sur une échelle à
trois valeurs {128, 256, 400}, et les IC de pente PASS sont bornés à zéro par le bas (k*
quantifié ne décroît pas). L'extension donne à la pente une vraie chance de casser.

**Substrat & instrument (INCHANGÉS, gelés) :** v2 (`SedimentParams()` défaut = KD/KE_V2,
`default_terrain`), compresseur quadtree famille-2, régénérateur constant-par-blocs (plancher
0 bit-à-bit), R1/Δχ-albedo, cap 409.6 float-éq. Aucun re-réglage, aucune re-calibration.

**Le changement, unique :** ajouter le checkpoint **L = 160** (16·L₀) à la génération
d'histoires, sur les **mêmes seeds {101..105}**, même protocole d'épisode. Rien d'autre.

**Grille & lecture :**
- k*(160) mesuré exactement comme k*(10..80) : médiane inter-seeds, clause 2×JND (aucune
  seed > 2×JND), max(M-A1, M-A2) < JND.
- **JND de lecture = pin sévère, IC ENTIER** : 6.0 %, 7.3 %, 8.7 % (les trois colonnes).
  Continuité : les JND de grille {2,3,4,5} % sont aussi re-tabulés à L=160 pour la surface,
  mais le VERDICT se lit à la ligne du pin.
- **Contrôles §A11 reconduits** à L ∈ {10, 160} (bornes) : ferm (attendu Δχ=0 exact,
  plomberie) + shuf (attendu k*=∞ SOUS CAP — portée §C-requalifiée : les budgets hors-cap
  1024/2048 restent non-discriminants, exclus du gate ; règle de forme appliquée). Toute
  violation → STOP-instrument, verdict non lisible.

### §A12-verdicts (figés, appliqués mécaniquement)

Lecture de la **pente k*(L) sur la moitié haute** L ∈ {40, 80, 160} au pin sévère, IC 95 %
bootstrap inter-seeds (10 000, seedé), comme §A3 :

- **CELLULE 1 PRONONCÉE** (fermeture) : k*(160) reste plat (= k*(80), soit 256 au pin
  central) ET pente compatible zéro (IC contient 0) ET k*(160) ≤ cap (409.6) — aux TROIS
  JND de l'IC pin. Portée gravée (§A5, réaffirmée) : **ce substrat (v2), ce pin (n=1,
  Romain, sévère), cette famille de compresseurs**. Court-circuite la manche 2 SUR CE
  SUBSTRAT uniquement. Le claim existentiel à l'échelle du but (obligations étagées, densité
  d'observation) reste ouvert.
- **FAIL-MUR PRONONÇABLE** (le mur enfin lisible) : k*(160) SAUTE au-dessus de k*(80)
  (typiquement 256→400, ou 400→∞/hors-cap) au pin central, pente positive > bruit
  inter-seeds. → cellule 2 : la forme du mur est mesurée, le ledger passe d'hypothèse à
  obligation. La manche 2 (registre-commis) devient la dépense suivante.
- **INDÉTERMINÉ-JND** (le résultat le plus probable au vu de l'entrée) : verdict qui
  DIFFÈRE selon la colonne de l'IC pin — p. ex. plat à 7.3 % et 8.7 % mais montant à 6.0 %
  (ic_bas montait déjà à L=40 sur la surface re-seuillée §C12). → le verdict est
  JND-dépendant DANS l'IC → c'est exactement la **surface k*(L, JND) que §C4 destine à
  l'Arc C comme l'objet que le pin résout**. Porté tel quel : la cellule 1 est prononcée
  À CONDITION que le pin vrai soit ≥ le JND-seuil de bascule, et cette condition est
  chiffrée, pas devinée. AUCUNE 3ᵉ famille, AUCUN run supplémentaire (règle de dernière
  famille §manche-1 : consommée).

### §A12-gardes (les deux de §C12, gravées ici)

1. **Échelle 3-valeurs** : « plat » = « n'a bougé ni vers 128 ni vers 400 ». Le plateau est
   réel mais grossièrement quantifié — la grille de budgets fine {32,64,128,256,400} rend le
   saut lisible s'il existe. Ne pas sur-lire la platitude comme une fermeture forte : c'est
   la MOITIÉ HAUTE {40,80,160} qui porte le verdict, pas les petits L.
2. **ic_bas armé** : le soupçon de croissance est réel et pré-nommé (6 % → 256/256/400/400
   sur la surface §C12). Si L=160 confirme que ic_bas monte pendant que le pin central reste
   plat → INDÉTERMINÉ-JND, pas cellule 1 nette. L'œil est armé, le verdict reste mécanique.

### §A12-résultat — VERDICT PRONONCÉ (2026-07-06) : **INDETERMINE_JND**

Extension exécutée sur couche ADDITIVE (h160 : 5 seeds × 160 épisodes ; `measures_160.npz` ;
agrafage au verdict), instrument gelé préservé **bit-à-bit end-to-end** : préfixes {10,20,40,80}
identiques aux originaux (5/5), contrôles L=10 (ferm ET shuf) identiques à `measures_qt`, surface
k\*(L≤80) identique à l'artefact C-1a `surface_kstar_aux_pins.json` (6 pins × 4 L, garde
non-tautologique). Suite 536/0skip. Revue par-tâche (spec+qualité) + revue opus dédiée du verdict +
revue whole-branch (« PRÊT À GRAVER ») + re-dérivation à la main du contrôleur. `run_history` /
`L_LIST` / `measures_qt.npz` non mutés (blob `88e16e2d` intact).

**Gate §A11 reconduit (L∈{10,160}) : CONFORME** — 0 violation / 80 cellules (ferm k\*=32 exact ;
shuf k\*=∞ sous cap). Verdict **LISIBLE**.

**Surface k\*(L) — pin sévère, IC ENTIER (médiane inter-seeds, budgets float-éq) :**

| colonne | JND | L=10 | L=20 | L=40 | L=80 | **L=160** | lecture par colonne |
|---|---|---|---|---|---|---|---|
| ic_bas | 6.03 % | 256 | 256 | 400 | 400 | **400** | fermée à 400 (pente IC ∋ 0) |
| pin | 7.33 % | 256 | 256 | 256 | 256 | **400** | **saut 256→400 à 16×** ; pente OLS{40,80,160}=1.29, IC=[−0.26, 1.29] ∋ 0 |
| ic_haut | 8.67 % | 256 | 256 | 256 | 256 | **256** | fermée à 256 (plate ×5) |

**Prononcé mécanique (combinateur §A12, AUCUNE lecture à l'œil) :** colonnes fermées =
{ic_bas ✓, pin ✗, ic_haut ✓} → **divergent** → **INDETERMINE_JND**. Ni CELLULE 1 PRONONCÉE (les 3
colonnes ne sont pas toutes fermées : le pin central a sauté), ni FAIL-MUR (le saut central n'est
**pas significatif** — pente IC ∋ 0). C'est le résultat que §A12 nommait « le plus probable ».

**Ce que dit le verdict (portée §A5, non surclamée) :** sur ce substrat (v2), à ce pin (n=1, Romain,
sévère), cette famille (qt) — la cellule 1 **n'est PAS prononcée définitivement**. Le verdict est
**JND-dépendant DANS l'IC** : bord lâche (8.67 %) → fermeture nette à 256 ; pin central (7.33 %) →
le budget saute à 400 à 16× (marche vers le mur, non significative) ; bord serré (6.03 %) →
fermeture à 400. C'est **exactement la surface k\*(L, JND) que §C4 destine à l'Arc C comme l'objet
que le pin résout**. Le seuil de bascule 256↔400 est **chiffré** (lisible dans la surface complète,
entre 7.33 % et 8.67 %), pas deviné. Le **claim existentiel à l'échelle du but reste ouvert**.

**Le plateau 256 (« cadeau » C-1a §C12) est FRAGILE :** il tenait plat à 8× (L≤80) au pin central ;
à 16× le pin central lui-même passe à 400. Le coût de persistance plancher (6,25 % de l'état fin,
256/4096) ne tient qu'aux histoires ≤ 8·L₀ **au pin réel** ; à 16× la marche vers le mur apparaît au
**centre** de l'IC. Note pour la spec game-design : provisionner le budget de persistance en tenant
compte de cette montée à horizon long, pas seulement du plancher court-histoire.

**Gardes §A12 appliquées (traçabilité, pas de sur-lecture) :**
1. **Échelle 3-valeurs {128,256,400}** — la grille fine a rendu le saut 256→400 LISIBLE : il a
   effectivement tiré, au pin central, à L=160. La platitude n'a pas été sur-lue comme fermeture
   forte ; c'est la moitié haute {40,80,160} qui a porté le verdict.
2. **ic_bas armé — soupçon NON confirmé tel que pré-nommé.** Le soupçon gravé était « ic_bas monte
   pendant que le pin central reste plat ». La réalité est AUTRE : ic_bas était déjà à 400 (dès
   L=40), le **pin central** a sauté, ic_haut est resté plat. `soupcon_confirme = false`. L'œil était
   armé pour un motif précis ; le combinateur mécanique (règle générale « divergence entre colonnes »,
   PAS l'histoire anticipée) a tranché correctement quand même. Dit honnêtement : la divergence est
   réelle, sa **forme diffère** de l'anticipation — c'est le design anti-fabrication qui fonctionne.

**Règle de dernière famille (§manche-1) : CONSOMMÉE.** Aucune 3ᵉ famille, aucun run supplémentaire —
§A12 le grave. La mission 16L₀ **s'arrête ici**.

**Point d'arrêt : verdict remonté.** Pas d'enchaînement sur la manche 2 ni sur une 3ᵉ famille.
Artefacts (pocPhysicator, branche `arc-c-pin-jnd`) : `outputs/arcA/verdict_16L0.json`,
`arcA_kstar_16L0.png`, `arcA_kstar_16L0_pirecas.png` ; commits Task 1 `9a2aeac`→ verdict `c645558`.
Durcissement considéré-et-décliné (revue whole-branch, non exigé par le contrat) : assert runtime
d'appariement des seeds au point d'agrafage — invariant déjà verrouillé par constante partagée +
test de non-régression L=10.

---

## §A13 — Manche 2 : registre-commis (gravé le 2026-07-06, AVANT toute mesure)

> **Déclencheur** : la règle §A0 gravée le premier jour — « manche 2 construite seulement si
> manche 1 FAIL ou INDÉTERMINÉ-porté ». Manche 1 close sur INDÉTERMINÉ-JND porté (§A12). La
> dette qui différait la manche 2 est soldée : ses seuils sont les pins MESURÉS de l'Arc C
> (sev 7.33 % IC [6.03, 8.67] ; lax 11.54 % IC [9.37, 13.87]), plus aucun placeholder.
> Texte apposé verbatim depuis la pré-enregistration de Romain.

### §A13-0 — Le claim existentiel, et ce que la grille §A0 devient après §A12

**Claim (§A6, opérationnalisé)** : il existe un **registre** (ledger d'événements commis +
commits d'émission) et un **foncteur de reconstruction déterministe** tels que :
- **(É1, invariants durs)** : les invariants commis sont préservés EXACTEMENT par la
  reconstruction (masses du résumé — propriété de projection du quadtree) ; aucun événement
  commis ne se dé-commet (pulses rejoués depuis leurs seeds, bit-vérifiés).
- **(É2, émissions)** : tout readout émis est re-dérivable sous **JND_sev** ; et à chaque
  émission ultérieure, le monde-reconstruction est **indistinguable (sous JND_sev) du monde
  vrai** — aucun observateur ne peut prendre la substitution en défaut. C'est la
  non-contradiction du témoignage, mesurable.
- **(É3, le reste : libre-mais-déterministe)** : reconstruction = f(registre, seeds
  d'événements) UNIQUEMENT ; double reconstruction bit-identique ; jamais de bruit de
  ré-échantillonnage.
- **Axe de croissance (la question existentielle)** : le coût PAR ÉMISSION (taille de
  commit k) ne croît ni avec l'écart inter-observations Δt ni avec la longueur de chaîne —
  le registre est borné par le budget d'observation, pas par l'historique.

**Grille §A0 relue après §A12 (gravée)** : manche 1 = INDÉTERMINÉ-JND porté, donc :
- **commis PASS** → architecture **ledger VIABLE** (le filet §A6 confirmé) : le jeu procède
  sur seed+registre QUEL QUE SOIT le côté où le pin vrai tranchera l'état-complet.
- **commis FAIL (cette famille)** → PAS cellule 3 automatique (l'état-complet reste
  conditionnellement vivant à ic_haut). Deux conséquences pré-écrites : (i) **le resserrage
  du pin spatial devient DÉCISIONNEL** (il départage la cellule-1-conditionnelle — le
  consommateur que le resserrage attendait) ; (ii) une politique de registre plus riche est
  un fork NOMMÉ, décision neuve, jamais un réflexe.
- **INDÉTERMINÉ-JND** → surface portée, même objet que §A12, lue conjointement.
- **INSTRUMENT-MUET** → scellé, remonté (la cellule obligatoire de toute grille).

### §A13-1 — Opérationnalisation (figée)

- **Substrat** : v2 gelé (`SedimentParams()`, `run_history`/`run_episode_trajectoire`,
  terrain défaut) — INCHANGÉ. Événements = pulses procéduraux seedés (le registre causal =
  la séquence de seeds). Portée nommée : les événements-joueur seraient des entrées de
  registre non-seedées, structurellement identiques — NON testés ici.
- **Modèle d'observateur** : émissions **plein-domaine** aux épisodes programmés (le pire
  cas pour le registre, le plus propre pour la mesure). Le fenêtrage spatial (axe volume)
  est HORS SCOPE, nommé en §A13-5.
- **Politique de registre (famille FIGÉE)** : à chaque émission t_i, commit =
  `summarize(état, budget k)` (quadtree famille-2, comptabilité §clause-1) ; le moteur
  **se ré-ancre** : état-moteur ← `regenerate(commit)` ; entre émissions, le monde-moteur =
  simulation avant depuis l'état ré-ancré, MÊMES seeds de forçage que le vrai.
- **Protocole de chaîne** : vérité et moteur partent du même t_0 ; à chaque émission t_i :
  mesurer Δχ(readout moteur, readout vrai) [R1, bandes porteuses], committer, ré-ancrer,
  continuer. La série Δχ_i le long de la chaîne EST l'objet du verdict : bornée/contractante
  = le registre ferme ; croissante à travers le JND = composition d'erreur = mur-registre.
- **Balayages** : Δt ∈ {1, 4, 16} épisodes (densité d'observation — la garde anti-vacuité
  gravée : la non-contradiction doit survivre à l'observateur DENSE comme au CLAIRSEMÉ,
  les deux bouts peuvent tuer) ; n_émissions = 6 par chaîne ; budgets k ∈ {128, 256, 400}
  (là où k* vit au pin, §C12) ; seeds {101..105}.
- **k\*_chaîne(Δt)** = plus petit k tel que max_i Δχ_i < JND pour la médiane des seeds,
  aucune seed > 2×JND (clause §A2 reconduite).

### §A13-2 — Verdicts (figés, combinateur mécanique de §A12 reconduit)

Lecture au **pin sévère IC ENTIER** (6.03 / 7.33 / 8.67 %), par colonne :
une colonne « ferme » ssi k\*_chaîne(Δt) ≤ cap (409.6) pour TOUS les Δt testés ET la série
Δχ_i ne croît pas à travers le JND (pas de composition).
- **REGISTRE-FERME** : les trois colonnes ferment → commis PASS (portée §A13-5).
- **MUR-REGISTRE** : les trois colonnes ouvertes (composition à travers le JND à tout
  budget ≤ cap) → commis FAIL de CETTE famille → conséquences §A13-0.
- **INDÉTERMINÉ-JND** : colonnes divergentes → surface k\*_chaîne(Δt, JND) portée.
- **INSTRUMENT-MUET** : contrôles §A13-4 en violation → verdict scellé, remonter.
**Diagnostic non-verdictal** : la même table à **JND_lax** (IC [9.37, 13.87]) — le delta
sev↔lax au niveau chaîne chiffre le stockage-deux-étages (§C0) sur l'axe temporel-registre.

### §A13-3 — Séquencement cheapest-first + gate de faisabilité (leçon W0)

- **SONDE d'abord (Task 1, ~minutes)** : 1 seed, Δt = 4, k = 256, 6 émissions. Lectures
  pré-écrites : (S1) Δχ_i traverse JND_sev dès les premières émissions → la composition est
  massive, le verdict d'existence est quasi rendu, la grille complète se re-dimensionne en
  conséquence (remonter avant d'acheter) ; (S2) Δχ_i borné/contractant → la grille s'achète.
  La sonde ne PRONONCE rien : elle dimensionne.
- **Gate d'achat (Task 2)** : benchmark de coût réel AVANT la grille (règle W0 : ≤ une nuit
  ~10 h). Estimation a priori ~4-5 h CPU. Si > gate : grille réduite PRÉ-ÉCRITE (n_émissions
  6→4, puis budget 128 retiré) — jamais improvisée.

### §A13-4 — Contrôles (attendus gravés AVEC LEUR PORTÉE — règle de forme)

1. **ferm-chaîne (plomberie)** : commit = champ complet (sans perte) → chaîne moteur ≡
   chaîne vraie **bit-identique**, Δχ_i = 0 exact partout. Portée : tout Δt, tout L.
2. **shuf-commit (discriminant)** : commit permuté (histogramme préservé) → première
   re-dérivation supra-JND. Portée : budgets **SOUS CAP** uniquement.
3. **corruption (détecteur de contradiction)** : UNE masse de bloc commise altérée → la
   re-dérivation de CETTE émission montre un supra-JND localisé. Portée : détection
   d'instrument, ne participe pas au verdict.
Toute violation → INSTRUMENT-MUET, scellé, remonter. Anti-fuite : la reconstruction ne
reçoit QUE (registre, seeds) — signature + décoy, comme manche 1.

### §A13-5 — Ce que la manche 2 ne prouve PAS (§13 local)

- **Un PASS** ne dit rien : du fenêtrage spatial (axe volume — émissions plein-domaine
  ici) ; des événements-joueur (tous procéduraux ici) ; du multi-observateur concurrent ;
  de la 3D et du couplage raide ; de l'axe temporel (σ_ω, non pinné) ; du tier-3 au-delà du
  déterminisme (aucun axe perceptuel neuf ne juge la « plausibilité du libre »). Substrat
  v2, pin n=1, cette famille de politique.
- **Un FAIL** est celui de CETTE politique (commit-résumé + ré-ancrage + replay), pas de
  tout registre possible — le fork « politique plus riche » est une décision neuve (§A13-0).
- Le pin garde son IC : tout verdict est lu sur l'IC entier, jamais au point central.

### §A13-résultat — VERDICT PRONONCÉ (2026-07-18) : **REGISTRE_FERME**

Grille exécutée sur iluin-tworings3, terminal natif (la VM Cowork locale a été MESURÉE
bit-identique au gel — `test_replay_prefixe_bit_identique` PASS, 30.6 s — mais son infra tue
tout processus à 45 s → repli chemin 1 pré-écrit ; le PASS VM est porté comme fait
d'instrument). Transfert des commits Tasks 2/3 par patch (`git am` sur base b0e4723),
bit-exactitude PROUVÉE par hashes de blobs (0fa5874/9055098/e871af5/9348056/c86b8c1).
25 tests PASS natifs avant exécution ; gate d'achat MESURÉ PASS (2.569 s/ép × 2640 épisodes
= 1.88 h ≤ 10 h, aucune grille réduite invoquée). Grille lancée le 15, interrompue, REPRISE
par parts le 18 (design résumable exercé — SKIPs corrects, zéro re-mesure). Anomalie chrono
seed 103 (2× plus lente, vérité et cellules) : temps horloge uniquement, sans effet mesure.

**Contrôles §A13-4 : 0 violation** — ferm Δχ = 0.0 EXACT partout (3 Δt) ; shuf-commit
supra-JND aux 3 budgets (Δχ₂ = 1.028 / 0.752 / 0.693 ≫ 0.0733) ; corruption détectée
(Δχ ém. 4 = 0.3905 supra-JND, préfixe 1..3 bit-identique, localisation 100 % ≥ 50 %) ;
É2 re-dérivé ≡ émis et É3 double-run bit-exacts. `instrument_muet = false`. Verdict **LISIBLE**.

**k\*_chaîne(Δt) — pin sévère, IC ENTIER (médiane inter-seeds, clause 2·JND §A13-1) :**

| colonne | JND | Δt=1 | Δt=4 | Δt=16 | lecture |
|---|---|---|---|---|---|
| ic_bas | 6.03 % | 256 | 256 | 256 | fermée à 256 |
| pin | 7.33 % | 256 | 256 | **128** | fermée |
| ic_haut | 8.67 % | 128 | 256 | 128 | fermée |

**Prononcé mécanique (combinateur §A13-2, AUCUNE lecture à l'œil) :** les trois colonnes
ferment → **REGISTRE_FERME** → commis **PASS**, portée §A13-5.

**Chiffres inconfortables gravés (règle W1 : tout se reporte, surtout ce qui gêne) :**
(Δt=16, k=400, colonne ic_bas) ne ferme PAS — clause 2·JND : seed 102, émission 4,
Δχ = 0.1369 > 0.1207 ; pic isolé qui relaxe (série sinon ≤ 0.012) et NON-MONOTONE en k
(même cellule à k=256 : 0.0855 — plus de budget, pire pic). Sans effet sur le verdict
(k\* = 256 ferme avant) mais consigné. Cellule la plus dure : (Δt=4, k=128) médiane
0.0963 > pin — l'axe Δt=4 est l'axe dur, forme de la sonde reproduite. Marges à k\* :
med_max 0.0554–0.0562 vs 0.0733 (~24 %) pour Δt ∈ {1, 4} ; 0.0265 à Δt=16.

**Diagnostic laxiste (non-verdictal, §C0) :** k\* passe de 256 (sévère) à 128 presque
partout (laxiste : {1:128, 4:128, 16:128} au pin_lax et ic_haut_lax ; 4:256 à ic_bas_lax)
— le delta sev↔lax vaut UN CRAN de budget (×2) sur l'axe temporel-registre : chiffrage
du stockage-deux-étages.

**Ce que dit le verdict (portée §A13-5, NON surclamée) :** sur ce substrat (v2), à ce pin
(n=1, Romain, sévère, IC entier), cette famille de politique (commit-résumé qt + ré-ancrage
+ replay) : le registre-commis FERME — k\*_chaîne ≤ 256 floats (6.25 % du champ fin, sous
cap 409.6) pour TOUS les Δt ∈ {1, 4, 16}, aucune composition à travers le JND nulle part.
Rien de prouvé sur : fenêtrage spatial, événements-joueur, multi-observateur, 3D/couplage
raide, axe temporel σ_ω, tier-3 (§A13-5, reconduit tel quel).

**Lecture d'innovation portée (non-verdictale, décision neuve à prendre) :**
ressaut-puis-relaxation GÉNÉRALISÉ — les pics par-seed sont isolés et relaxent (jamais de
croissance à travers le JND) et k\*(Δt=16) < k\*(Δt=4) au pin : plus l'écart entre commits
est long, plus le substrat dissipe le bruit de commit. Levier d'architecture possible :
cadencement des commits sur la relaxation. À explorer comme décision neuve, PAS un
enchaînement.

**Point d'arrêt honoré : verdict remonté à Romain, texte endossé avant gravure (2026-07-18).**

### Décision fork temporel (2026-07-18) : **(d) PORTER NON RÉSOLU** — gravée avec conditions

**Contexte.** Backup Arc V : abandonné (rien trouvé — décision Romain 2026-07-18) → fork
(a/a′/d) DÛ (§C6). Branches recalculées après §A13-résultat, dont UNE NEUVE nommée :
**(a″) hybride registre** — films σ_ω générés par la machinerie É2 validée manche 2
(dynamique réelle, σ_ω piloté par le forçage) ; hypothèse de transfert entre (a) et (a′)
(l'indistinguabilité É2 est mesurée sur v2-sédiment/quadtree, PAS sur panaches — transfert
lui-même hypothèse). Nommée, PAS choisie.

**Décision Romain : (d).** L'étau W1 [−19, −10] % est porté NON RÉSOLU → **W1
mort-par-défaut** (réanimation = décision neuve, jamais un réflexe). **Gate fovéa-z :
OUVERT** (§C4-4, 2-des-3 issues : ne dépendait plus que du consommateur 1, servi).

**Motifs gravés (structurels, pas l'élan du PASS) :** (i) le consommateur de σ_ω — W1,
128-vs-256 — érodé deux fois : par la fovéa (borne L_eff, question de résolution globale
suspecte) et par §A13-résultat (le jeu procède sur seed+registre quel que soit le côté de
l'état-complet) ; (ii) le chemin critique de viabilité passe par le pin r_fovea (mesurable,
harnais Arc C existant), pas par σ_ω ; (iii) mesurer σ_ω aujourd'hui = acheter un instrument
avant que la question existe — si la spec nomme un référent temporel, il sera plus précis
que le σ_ω générique (ex. perceptibilité des transitoires de ré-ancrage au cadencement).

**DETTE NOMMÉE σ_ω — condition de réveil (gravée) :** si le paper-grade de la spec fovéa-z
nomme une décision qui consomme un référent temporel, le fork ROUVRE avec cette question
précise en main. Branches vivantes au réveil : (a″) préférée sur (a′) ; (a) seulement
re-chiffrée (son prix d'origine est mort avec Arc V). AUCUN réveil silencieux.

**Test d'honnêteté consigné :** (d) n'aurait PAS été choisi après un MUR — l'état-complet
serait redevenu load-bearing et l'étau avec lui. Le PASS change le calcul par sa
conséquence structurelle (§A13-0 : ledger viable), pas par son élan.

## §A14 — Sonde fenêtrage-fovéa (pré-enregistrée 2026-07-18, AVANT toute mesure)

> **Statut : gravé le 2026-07-18, avant toute ligne de code de sonde.** Gate fovéa-z ouvert
> (décision (d) supra) ; cette sonde est le falsificateur le moins cher du mariage
> registre×fovéa — l'exclusion §A13-5 « fenêtrage spatial » attaquée AVANT que la spec fige.

**Objet.** Deux mécanismes que la manche 2 (émissions plein-domaine) ne pouvait pas voir :
(1) **contamination** — entre émissions, la dynamique fait entrer dans la fenêtre de
l'information venue du dehors non-contraint ; (2) **couture** — discontinuité au bord de
fenêtre au ré-ancrage. Sonde = lecture remontée, AUCUN verdict, aucun branchement
automatique.

**Cellule.** Seed 101, Δt = 4, 6 émissions (reconduite sonde §A13-3). Substrat v2 gelé,
exécution machine-instrument (terminal natif).

**Fenêtre.** Fixe, **32×32**, alignée quadtree, quadrant (0,0) par défaut. **Garde
anti-vacuité pré-écrite** : la vérité DANS la fenêtre doit montrer une dynamique supra-JND
vs t₀ (fenêtre morte = sonde triviale) ; sinon quadrants essayés dans l'ordre FIXE
(0,0)→(0,1)→(1,0)→(1,1), premier vivant retenu — règle mécanique, zéro choix après lecture.

**Protocole (bras unique, NU — décision Romain 2026-07-18 : un seul étage).** Vérité et
moteur partent du même t₀, mêmes seeds de forçage. À chaque émission t_i : mesurer
Δχ_fen(readout moteur∣fenêtre, readout vrai∣fenêtre) ; commit = summarize(état∣fenêtre,
k_fen) ; ré-ancrage DANS la fenêtre seulement (état-moteur∣fenêtre ← regenerate(commit)) ;
DEHORS : le moteur continue SANS contrainte. La série Δχ_fen,i est l'objet de la lecture.

**Choix d'implémentation pré-enregistrés (endossés avec ce texte) :**
- (i) **k_fen = 64** — aire-proportionnel au k\* = 256 de la grille (32²/64² × 256) ;
- (ii) ré-ancrage par remplacement de zone : la couture est ASSUMÉE (mécanisme mesuré,
  pas un défaut d'instrument) ;
- (iii) nouveau script consommant les primitives du cœur Task 0 — cœur INTOUCHÉ ;
- (iv) lecture au jnd_sev, formes de la sonde §A13-3 (bornée/contractante vs traverse),
  Δχ_fen,1 ≡ 0 structurel exclu (aucun commit avant la première émission).

**Vérification d'instrument DUE en build, AVANT run :** les bandes porteuses de
max_carrier sont-elles définies sur 32×32 ? Si les porteuses basses sont plus grandes que
la fenêtre, la sonde est muette PAR CONSTRUCTION — remonter avant toute mesure.

**Escalade PRÉ-ÉCRITE (jamais improvisée) :** si la série traverse (contamination avérée),
le remède à tester est le bras **DEUX-ÉTAGES** : commit fin dedans (k_fen = 64) + commit
grossier dehors (k_out = 64, soit un niveau LOD d'écart par unité d'aire), ré-ancrage des
deux zones. Paramètres FIGÉS ici, avant toute lecture — ils ne se re-règlent pas après.

**Diagnostic non-verdictal :** Δχ plein-domaine de la chaîne fenêtrée (chiffre l'erreur
totale, dehors non-contraint inclus).

**Ce que la sonde ne dit pas :** fenêtre mobile (bras ultérieur nommé), multi-fenêtres /
multi-observateur, excentricité perceptuelle réelle (le JND hors-fovéa n'est PAS pinné —
tout grossier-dehors est une politique, pas une promesse perceptuelle), 3D.

**Provenance de la décision :** le deux-étages, d'abord retenu, a été rétrogradé en
escalade pré-écrite après objection d'attribution (le remède embarqué dans la sonde
masquerait le mécanisme qu'elle doit exposer) — décision Romain, 2026-07-18.

### §A14-lecture — SONDE EXÉCUTÉE (2026-07-18) : AUTRE, bornée loin sous le pin

Build S1 revu (conforme §A14, vérification indépendante 13 tests PASS dans la VM
bit-identique) ; **choix (v)-(viii) ENDOSSÉS par Romain** (revue remontée, run lancé sur
endossement — consigné ici explicitement). Vérification d'instrument PASS (bandes définies
sur 32×32, porteuses fenêtre présentes sur les gelés s_L10/s_L20, non muette). Run natif
(machine-instrument), garde anti-vacuité : quadrant (0,0) VIVANT au premier essai
(Δχ vs t₀ ∈ [0.59, 1.45]) ; commits saturés 64/64 (budget au travail, mesure non triviale).

**Séries (seed 101, Δt=4, k_fen=64, fenêtre (0,0)) :**
Δχ_fen = [0, 0.0169, 0.0107, **0.0460**, 0.0315, 0.0303] ;
Δχ_plein (diagnostic) = [0, 0.0115, 0.0215, 0.0047, 0.0115, 0.0076].

**Lecture mécanique : AUTRE** — pas de traversée (max 0.0460 = 63 % du pin sévère), pas S2
(ratio 4-6/2-3 = **2.73 > 1.5** — le facteur load-bearing remonte la forme une 2e fois au
lieu de l'absorber). **Escalade deux-étages NON déclenchée** (condition pré-écrite =
traversée ; il n'y en a pas).

**Trois faits remontés :**
1. Le fenêtrage à budget aire-proportionnel N'A PAS dégradé la tenue — amplitudes
   équivalentes à la sonde §A13-3 plein-domaine k=256, PIC AU MÊME ENDROIT (i=4, t=16 :
   même événement du seed × perte de commit, pas un artefact du fenêtrage), relaxation
   derrière. Premier appui mesuré du mariage registre×fovéa.
2. La contamination dehors→dedans est restée sous le plancher de la cellule : Δχ_plein <
   Δχ_fen (i=4 : 0.005 vs 0.046) — le dehors non-contraint (même seed, même t₀) reste
   quasi-vérité, le flux dominant est dedans→dehors et il est minuscule. Le témoignage
   fenêtré tient alors que le dehors n'est JAMAIS commis.
3. La forme « ressaut-puis-relaxation borné sous pin » est maintenant vue DEUX fois, dans
   deux protocoles (plein-domaine §A13-3, fenêtré §A14) — consistance croissante, appui
   pour la question cadencement de la spec.

**Portée (stricte) :** 1 seed, 1 Δt, 1 fenêtre fixe (0,0), k_fen=64, 6 émissions — une
sonde, pas un verdict. Fenêtre mobile, multi-fenêtres, excentricité perceptuelle, 3D :
toujours hors-lecture (§A14). Suite : la spec fovéa-z paper-grade s'appuie sur cette
lecture ; toute grille fenêtrée verdict-grade = décision neuve si la spec la demande.

### F0′-lecture (2026-07-18) — sonde locale de la divergence inter-machines : DEUX NULS ÉLOQUENTS

Opérationnalisation endossée (Romain) : F0′ local (leviers de dispatch sur la
machine-instrument, VM bit-identique) + F0-cloud opportuniste à la prochaine session.
Protocole : `run_history(101,10)` vs `h_s101.npz` (le test de gel), levier forcé, lecture
état + Δχ readout. **Résultats : (1) OPENBLAS_CORETYPE=NEHALEM → bit-identique (0 cellule).
(2) NPY_DISABLE_CPU_FEATURES="AVX2 FMA3 AVX F16C X86_V3" → bit-identique.**

**Lecture.** Les deux leviers descendants sont nuls : le chemin de calcul est stable de
la baseline à AVX2 sur cette machine. Fait structurel : la machine-instrument N'A PAS
d'AVX512 (features énumérées) ; le CPU cloud (serveur) en a presque sûrement ; np.exp est
dans le chemin chaud (dépôt de pulse, readout albédo) et numpy possède des implémentations
SIMD spécifiques AVX512 pour exp. **Attribution raffinée : la divergence cloud vient
vraisemblablement des chemins SIMD AU-DESSUS du plafond de la machine-instrument
(AVX512) — irréproductible localement PAR CONSTRUCTION (on ne peut que descendre, et la
descente est stable).** Cohérent avec : la VM locale (même CPU, userland noble) PASS le
gel ; le cloud (autre CPU) FAIL.

**Conséquences.** (a) L'hypothèse « divergence inter-machines = sous-JND perceptuel »
reste NON MESURÉE — aucune divergence locale productible ; **F0 se complète à la
prochaine session cloud** : run 76 s, SAUVEGARDER l'état divergent (la leçon : l'artefact
du FAIL originel n'avait pas été persisté), Δχ vs gelé. (b) Le gate de sortie de la spec
v1 attend ce point (§8). (c) Leçon d'instrument gravée : les états divergents se
persistent TOUJOURS — un FAIL non persisté coûte une session de plus.

## §A15 — Pré-enregistrement TRANCHE-MOTEUR (F1) (gravé 2026-07-18, AVANT toute ligne de code de tranche)

> **Statut : gravé le 2026-07-18 sur endossement Romain (texte + décisions T1-T7).**
> La LISTE des 4 mesures vient de SPEC-FOVEA-Z §8-F1 (D13) ; les chiffres se figent ICI,
> avant tout run — jamais après. Brouillon de travail : pocPhysicator
> `claude/prereg-tranche-moteur-2026-07-18-BROUILLON.md`.

**Objet.** F1 brûle 4 hypothèses d'un coup : (a) frame-time vs L_eff [le falsificateur
du mot « moteur »] ; (b) écart live↔rederive sous f32 [contrat D5/Option A] ; (c) débit
PCIe du schéma diff [topologie §5] ; (d) coût de load/replay du ledger [§4]. C'est une
**MESURE, pas un prototype** : code autorisé = harnais de tranche (scripts consommant
les primitives existantes + chemin GPU jetable), PAS le moteur. Un critère de mort
déclenché = remontée + décision, jamais un contournement. Verdict SANS MORT requis
pour : la spec fait foi (gate iii §8), campagne r_fovea débloquée (garde §7).

**Ordre (T7, tranché).** Exécution F1 **DÉCOUPLÉE** de F0-cloud : F0 ne gate que
l'hypothèse inter-machines (§5), pas la tranche. F0-cloud reste dû à la prochaine
session cloud (persister l'état divergent).

**Instrument.** iluin-tworings3, terminal natif, RTX 3050 Ti 4 Go. Stack GPU (T6) :
nommé au chiffrage du build, endossé AVANT toute ligne (candidats CuPy/torch/JAX ;
contraintes : f32, 4 Go, coexistence avec numpy 2.4.6 du chemin rederive). Chrono GPU :
synchronisation explicite, warmup ≥ 30 frames EXCLU, série ≥ 300 frames, **médiane ET
p99 reportées toutes deux**. **Vérifications d'instrument DUES avant toute lecture
(sinon tranche muette, remonter) :** (1) résidence VRAM concordante allocateur vs
nvidia-smi ; (2) sensibilité du chrono (n_fov 256→512 doit bouger le frame-time) ;
(3) le rederive CPU f64 reproduit le gel bit-exact sur la machine ; (4) bande passante
PCIe soutenable MESURÉE À VIDE — le dénominateur de M-c, avant de mesurer le schéma diff.

**M-a — frame-time vs L_eff.** Enveloppe §6 épinglée (c=8, f32, n_fov=512, N_niv=10,
γ₂≈3 → ~7.9 M cellules actives, ~0.9 Go). F représentatif (T1, tranché) : **substrat v2
ÉTENDU à c=8 champs** — représentatif en COÛT (stencil, champs), pas en physique-jeu.
Fenêtre fovéa **MOBILE** (translation continue → re-prédiction Harten au déplacement).
Scan : N_niv actifs ∈ {2, 4, 7, 10} × n_fov ∈ {256, 512}.
**MORT-a : médiane frame-time > 16.7 ms à la cellule d'enveloppe (n_fov=512, N_niv=10)**
(T2) ; p99 reportée, non verdictale en v1. Gate §6 reconduit (pas une mort) : résidence
mesurée > 1.5 × 0.9 Go ⇒ re-épinglage du quadruplet — décision, jamais glissement.

**M-b — écart live↔rederive aux émissions.** Chaîne d'émissions fenêtrées (k_fen
aire-proportionnel, cap 10 % reconduit) sur le chemin VIVANT GPU f32 ; rederive =
f(registre, seeds), CPU f64, disciplines du harnais ; Δχ readout aux émissions
(albedo + delta_chi, max_carrier — l'espace instrument, jamais l'état). Cellule :
**3 seeds {101, 102, 103}, Δt=4, 6 émissions** (Δt=4 = l'axe dur mesuré §A13-résultat).
**MORT-b : max de série Δχ live↔rederive > 0.0733 (pin sévère), PAR-SEED** (T3) — un
seul seed qui traverse suffit. ⇒ Option A morte, repli Option B (tout-déterministe) =
**décision neuve** — son coût de frame n'est pas mesuré dans cette tranche sans nouveau
pré-enregistrement.

**M-c — débit PCIe réel du schéma diff.** Octets/frame réels CPU↔GPU : descente
(prédiction Harten des fenêtres actives) + remontée (coefficients de détail). Enveloppe
attendue : échelle n_fov² — brut ~8.4 Mo/direction/frame plein-fovéa (512²×8×4) ; le
diff doit faire MIEUX que le plein. **MORT-c (deux branches, T4) : (1) temps de
transfert > 25 % × 16.7 ms = 4.2 ms** à la cellule d'enveloppe ; **(2) fuite
d'échelle** : le trafic croît avec la taille du monde à fovéa fixe (doubler le monde
doit laisser le trafic ~constant, sinon la topologie §5 est morte telle que dessinée).

**M-d — coût de load/replay du ledger.** Ledger de référence : 1 h simulée à
1 commit/s ≈ 3600 commits ≈ 11.5 Mo, généré par la tranche. load = re-dérivation chemin
bit-exact jusqu'à la dernière émission + reprise du vivant.
**MORT-d : temps de load > 30 s SANS snapshot-racine** (T5). Diagnostic non-verdictal
reporté : même load AVEC un snapshot-racine (format (d), §4) au milieu du ledger —
chiffre ce que la compaction achète, sans qu'elle porte le verdict.

**Portées (non surclamées).** Rien sur : r_fovea/excentricité (campagne gatée APRÈS
verdict sans mort), 3D (64³ = enveloppe calculée, pas mesurée), multi-vue/concurrence
(v1.1), qualité perceptuelle produit (pin n=1), coût du RENDU (physique seule), coût de
l'Option B (nouveau pré-enregistrement si repli).

**Chiffrage du build DÛ avant achat (règle plan).** Estimation Claude Code du harnais
(chemin GPU F + fovéa mobile, portage commits/readout au vivant GPU, compteurs PCIe,
générateur de ledger 1 h), remontée à Romain ; gate d'achat = endossement. **Risque
nommé :** le portage commits/readout (M-b) est le morceau le plus susceptible de faire
glisser la tranche de « mesure » vers « prototype » — si le chiffrage est gros, la
question de SCINDER M-b se pose AVANT achat, pas après.

**Décisions consignées (Romain, 2026-07-18) :** T1 substrat v2 étendu c=8 ; T2 B_frame
= 16.7 ms (p99 non verdictale) ; T3 MORT-b = pin 0.0733, par-seed ; T4 f_PCIe = 25 %
(⇒ 4.2 ms) ; T5 T_load = 30 s (snapshot = diagnostic) ; T6 stack GPU déféré au
chiffrage, endossement avant build ; T7 exécution découplée de F0-cloud.

### §A15-complément — Gate d'achat TRANCHÉ (2026-07-18) : chiffrage endossé, SCISSION, CuPy

Chiffrage remonté (pocPhysicator `claude/chiffrage-tranche-moteur.md`) et ENDOSSÉ
(Romain, 2026-07-18 — E1 à E7). Fait structurel consigné : **(a) et (b) ne sont pas le
même F** — M-b exige un portage numériquement fidèle de `run_episode` (wetdry O2
CFL-adaptatif + Exner + pulses) en f32 GPU, sinon Δχ confond « non-déterminisme f32 »
et « physique différente ». Le risque nommé §A15 est confirmé au chiffrage : (b) ~40 %
de l'effort, seul risque ÉLEVÉ.

**E1** — Chiffrage global endossé (8–11 séances, ~3200–4700 LOC style maison).
**E2/T6 clos** — Stack = **CuPy** (`cupy-cuda12x`, même venv, `numpy==2.4.6` épinglé,
vérif #3 gel bit-exact REJOUÉE après install). Second choix : torch. JAX écarté pour
cette tranche : transferts/placement implicites (tue l'attribution PCIe exacte de M-c),
préallocation VRAM (fausse la vérif #1), recompilation par shape (la fovéa mobile).
**E3** — **SCISSION** : tranche-1 = M-a/M-c/M-d via (a)(c)(d)(e)(f) [5–6 séances] ;
tranche-2 = M-b via (b) [3.5–5 séances], achetée SEULEMENT si tranche-1 SANS MORT.
Décision d'ORDRE D'ACHAT : aucun critère de mort, protocole ni portée modifiés ;
**verdict F1 = les 4 mesures** — « la spec fait foi » (gate iii §8) et la campagne
r_fovea (garde §7) attendent tranche-2. Nommé ici : latence assumée, aucun glissement.
**E4** — Choix du composant (a) : (E4a) 8 champs = (h,hu,hv,s)×2 sous stencil complet
— majorant honnête, l'alternative « 3+5 passifs » minorerait le coût et fabriquerait un
M-a complaisant ; (E4b) shapes fixes préalloués par niveau ; (E4c) trajectoire =
balayage 1 cellule fine/frame (saut de fenêtre = diagnostic non-verdictal) ; (E4d)
remontée = coefficients des seules fenêtres touchées, chaque frame (c'est le schéma
diff que M-c mesure).
**E5** — Le vivant f32 calcule SON PROPRE dt CFL — forcer la séquence dt du f64 dans
le vivant réduirait l'écart et fabriquerait le verdict : interdit, consigné avant build.
**E6** — Génération du ledger M-d : Δt=1 épisode/commit, CPU natif de nuit,
NON-verdictal (MORT-d ne chronomètre que le LOAD, jamais la génération ; la taille du
ledger — 3600 commits, ~11.5 Mo — est préservée, c'est elle que le parse paie).
**E7** — Load = parse+validation INTÉGRALE du ledger + regenerate(dernier commit) +
replay du segment courant + reprise du vivant. Conséquence mécanique du ré-ancrage :
l'état∣fenêtre post-ré-ancrage = f(dernier commit) seul ; le replay intégral serait de
la VÉRIFICATION (l'audit É3), pas de la construction — résultat prouvé identique (H3).

**Trois consignes adversariales attachées à E7 (endossées avec lui) :**
1. **MORT-d re-scopé, DIT EN FACE** : 30 s borne le coût du LEDGER (parse, intégrité,
   regenerate), pas celui de la physique. Le critère devient peu exigeant — consigné
   comme tel, pas déguisé en falsificateur dur.
2. **Le diagnostic snapshot-au-milieu est RETOURNÉ en test de la claim E7** : attendu
   pré-enregistré = écart de load avec/sans snapshot ≈ 0. Tout écart matériel = AUTRE
   remonté (le diagnostic falsifie E7 au lieu d'être vidé par lui).
3. **Note de spec (consignée, §4 NON réécrit)** : la compaction snapshot-racine achète
   du disque et de la rétention, PAS du temps de load — « le replay repart de là »
   (§4) ne doit pas être lu comme une promesse de load.

**Consigne tranche-2 (gravée d'avance)** : son pré-enregistrement inclura une
re-mesure frame-time du F FIDÈLE (non-verdictale) — contrôle de représentativité du
proxy T1, rendu gratuit par la scission.

**Prochain pas : build TRANCHE-1 (Claude Code)** — composants (a)(c)(d)(e)(f),
implantation `src/f1_gpu/` + `scripts/run_f1_*.py` + `tests/test_f1_*.py`, cœur
manche 2 INTOUCHÉ, revue adversariale avant merge, aucune lecture sans vérification
d'instrument PASS. Mesures : iluin-tworings3 natif uniquement. Point d'arrêt aux
lectures — aucun enchaînement automatique.

### §A15-complément-2 — Build tranche-1 endossé (2026-07-18) : B1-B9 + 3 consignes

Build livré, revu (bloquant sha256 dtype/shape corrigé), endossé (Romain) et commité
(pocPhysicator 4e0e719). Choix B1-B9 consignés dans `src/f1_gpu/__init__.py`.
**Trois consignes gravées avec l'endossement :**
1. **EPS_DETAIL = 1e-4 (B4)** = paramètre d'instrument `[NON-ANCRÉ perceptuellement]` —
   il achète le trafic de M-c et rien en tranche-1 ne mesure son coût perceptuel.
   HÉRITÉ par le pré-enregistrement tranche-2 comme SUSPECT NOMMÉ si M-b lit AUTRE ;
   la lecture M-c reporte le dense-équivalent EN ÉVIDENCE (chiffres inconfortables).
2. **Fenêtres d'énergie figées (B3)** : le trafic de déplacement mesuré est un
   MINORANT nommé (seule la fovéale bouge) — porté dans la portée de la lecture M-c.
3. **Faits d'instrument** : CuPy 14.1.1 (clause « plus récente supportant
   numpy 2.4.x » exercée, divergence vs 13.x nommée) ; la VM voit le GPU
   (passthrough 570) — dev/tests uniquement, JAMAIS une mesure ; vérif #3
   (gel bit-exact) DUE en natif avant toute lecture.

**Séquence de runs (natif, Romain exécute, chaque lecture remontée, zéro
enchaînement) :** 1. run_f1_verifs.py → 2. run_f1_ma.py → 3. run_f1_mc.py →
4. run_f1_md_generer.py (nuit, non-verdictal) → 5. run_f1_md.py.

### §A15-complément-3 — Correction de la consigne 2 (2026-07-18) : concession, majorant rétabli

La consigne 2 du complément-2 (« fenêtres d'énergie figées ⇒ trafic de déplacement =
minorant ») reposait sur une lecture erronée de B3 par la session critique
(« offsets y figés » = offsets de POSITION relatifs, pas fenêtres statiques) —
erreur concédée. Le build endossé 4e0e719 faisait translater les 3 fenêtres par
niveau en lockstep avec le balayage E4c ; l'alignement du code sur la consigne
erronée (fluide-reduit 7967883) est REVERTÉ — **décision Romain : la géométrie
retenue est « les 3 fenêtres bougent »** — majorant honnête de la descente,
cohérent avec E4a (pour un critère de mort, on mesure la borne haute).
**Consigne 2 RÉ-SCOPÉE :** le trafic de déplacement mesuré est un **MAJORANT DE
CADENCE** (lockstep avec le regard) ; la géométrie de production (fenêtres
d'énergie pilotées par l'énergie, cadence irrégulière) n'est PAS mesurée —
portée nommée de la lecture M-c. Consignes 1 (EPS_DETAIL) et 3 (faits
d'instrument) inchangées. Aucun run n'avait été lancé — aucune mesure invalidée.

### §A15-lecture-M-a (2026-07-19) — MORT-a DÉCLENCHÉ — décision : micro-mesure fusionnée CAPPÉE

**Instrument armé au préalable (run_f1_verifs.py, natif)** : #1 concordance VRAM exacte
(écart 0 sur témoin 256 Mo) ; #4 bande passante mesurée à vide (plateau pinned ~8.4 Go/s
H2D / ~6.0 Go/s D2H ; à 64 Ko le débit effectif tombe à 2.8–4.6 Go/s — latence
dominante, nommé AVANT M-c) ; #3 gel bit-exact PASS (26.7 s) après install CuPy.

**Lecture M-a (run_f1_ma.py, natif, chiffres inconfortables en évidence) :**
- ENVELOPPE (512, 10) : médiane **703.0 ms**, p99 711.0 — seuil gravé 16.7 ms :
  **MORT-a DÉCLENCHÉ, facteur ×42**.
- AUCUNE cellule du scan ne tient : même (256, N_niv=2) = 21.3 ms > 16.7.
- Scan complet (med ms) : 256/2=21.3 ; 512/2=78.9 ; 256/4=64.0 ; 512/4=237.3 ;
  256/7=126.8 ; 512/7=470.7 ; 256/10=190.2 ; 512/10=703.0. p99 serrés partout
  (mesure propre) ; vérif #2 passée de fait (×3.7 à n_fov double).
- Diagnostic pré-enregistré : coût LINÉAIRE en niveaux (≈×1/×3/×6/×9) et ~100 ms/M
  cellules constant → PAS l'overhead de lancement, le CALCUL. Limite du diagnostic
  NOMMÉE : il distingue lancement vs travail, pas travail-naïf vs travail-minimal
  (les passes mémoire intermédiaires CuPy comptent comme « calcul »).
- Gate §6 NON déclenché : résidence 0.96 Go < 1.35 — la note VRAM (0.9 Go calculé)
  est CORROBORÉE. La moitié mémoire du claim tient ; la moitié temps casse.
- Fait acquis, gravé : **le F naïf CuPy ne tient la frame nulle part dans le scan.**

**Arithmétique d'enveloppe (calcul, pas mesure)** : état 7.08 M cellules × 8 champs
f32 ≈ 226 Mo ; stencil fusionné idéal ~4–12 passes mémoire/frame ≈ 0.9–2.7 Go à
~150 Go/s soutenus ⇒ **~6–18 ms : À CHEVAL sur le seuil**. Le papier ne tranche pas.

**Décision Romain (2026-07-19) : micro-mesure fusionnée, CAPPÉE** — décision neuve
pré-nommée au chiffrage (« mitigations nommées, PAS achetées »). Pré-enregistrement :

**M-a′ — micro-mesure de la borne d'implémentation (pré-enregistrée ICI, avant tout code) :**
- Protocole : RawKernel CUDA fusionné du MÊME motif E4a (2 systèmes (h,hu,hv,s),
  stencil wetdry O2 complet, 2 étages SSP-RK2, réduction CFL payée non consommée,
  dt figé, bathymétrie plate) — 1 niveau, 3 fenêtres n_fov=512, chrono B6
  (warmup 30 exclu, série 300, médiane ET p99), même base que M-a (calcul,
  transferts hors-mesure). Machine-instrument, natif.
- **CRITÈRE PRÉ-ÉCRIT : médiane par-niveau > 16.7/9 ≈ 1.856 ms ⇒ MORT-a CONFIRMÉE
  au niveau ARCHITECTURE** (à cette enveloppe, ce quadruplet §6) — l'extrapolation
  ×9 niveaux est étiquetée TRANSPOSITION (linéarité en niveaux MESURÉE sur le naïf).
- Si < 1.856 ms : MORT-a est RE-SCOPÉE à l'implémentation naïve ; le claim (a)
  survit au niveau borne — et le règlement COMPLET de M-a exige alors une décision
  neuve (re-mesure enveloppe pleine avec F fusionné + fovéa mobile + E4d), qui ne
  s'enchaîne PAS automatiquement.
- **CAP ANTI-TAPIS-ROULANT GRAVÉ : dernière escalade d'implémentation.** Au-delà du
  RawKernel fusionné (CUDA Graphs, exotique), toute nouvelle escalade est INTERDITE —
  si le fusionné dépasse, la mort est architecturale, point.
- Coût : ~1 séance build + run minutes. Chiffrage remonté avant achat si > 1 séance.

**Séquence restante inchangée** : M-c, génération (nuit), M-d restent dus — leurs
lectures (PCIe, ledger) informent LES DEUX branches. Chaque lecture remontée.

### §A15-lecture-M-c (2026-07-19) — PAS DE MORT — le PASS repose sur la remontée seuillée

**Lecture mécanique (run_f1_mc.py, natif)** : transfert total enveloppe (512,10)
médiane **2.784 ms** < 4.2 gravé (p99 3.676 = 87 % du seuil, nommé) ; diff 12.74 Mo
< plein 16.78 Mo ; fuite d'échelle N_niv 10→11 : ratio **0.997** (structurel ~1.11,
seuil instrument 1.5). **MORT-c(1) et (2) NON déclenchés.**

**Chiffres inconfortables en évidence (consigne 1 exercée) :**
- Trafic à 99.4 % en remontée seuillée (D2H 12.64 Mo ; descente 74 Ko) — le PASS
  repose sur EPS_DETAIL=1e-4 : diff = 5.6 % du dense-équivalent 226.5 Mo ; niveau
  fin dominant (882k coefficients ≈ 7 Mo, densité ~14 %). Suspect nommé pour
  tranche-2 : RECONDUIT.
- Sémantique du « plein » : la référence gravée = fovéale seule dense aller-retour ;
  le diff remonte les 27 fenêtres — gain réel vs dense-all : ×18 ; gain vs plein
  gravé : ×1.32 seulement.
- Ratio d'échelle 0.997 < 1.11 structurel : le niveau grossier ajouté ne remonte
  ~rien — cohérent §5 (monde en log, grossier quasi-statique) MAIS en partie
  artefact d'EPS_DETAIL + série 300 frames (origine grossière immobile). Non-verdictal.
- Concession session critique : l'arbitrage B3 (complément-3) porte sur la
  descente = 0.5 % du trafic — sain en principe, immatériel dans ce régime.
- Débit effectif D2H ~4.6 Go/s, cohérent vérif #4 — l'overhead petits-transferts
  nommé avant lecture n'a pas mordu (transferts batchés par niveau).

**Portées (reconduites du driver)** : majorant de cadence (géométrie de production
non mesurée) ; octets dépendants d'EPS_DETAIL non-ancré perceptuellement.

**Reste dû** : génération ledger (nuit), M-d, et M-a′ (build Claude Code). Chaque
lecture remontée.

### §A15-lecture-M-a′ + VERDICT (2026-07-19) — **MORT-a ARCHITECTURALE PRONONCÉE** (Romain)

**Lecture mécanique (run_f1_ma_prime.py, natif)** : médiane **5.056 ms/niveau**
(p99 5.541) contre seuil pré-écrit 1.856 — **×2.7 au-dessus**. Critère rempli.
La fusion a gagné ×15.6 sur le naïf (78.9 → 5.06 ms/niveau) : le build a fait son
travail, et ça ne suffit pas. Extrapolé ×9 : ~45.5 ms/frame (TRANSPOSITION,
linéarité défendue : 524k threads/niveau, GPU saturé — pas de parallélisme
inter-niveaux à récupérer). CAP honoré : dernière escalade, aucune autre.

**Discussion consignée avant prononcé (point d'arrêt honoré) :**
- T2 n'est pas le coupable : 45.5 > 33.3 aussi — même le budget laxiste meurt.
  Aucun B_frame ne sauvait ce quadruplet.
- 5.056 = borne d'une fusion COMPÉTENTE, pas l'optimum théorique — l'incertitude
  entre les deux est celle que le cap a choisi d'accepter (gravé avant run).
- PORTÉE STRICTE : meurt LE QUADRUPLET §6 (c=8 plein à tous niveaux, n_fov=512,
  N_niv=10, budget 16.7 ms, sur 3050 Ti) — pas la fovéa-z, pas Cascade. L'enveloppe
  était un majorant délibéré, leviers nommés NON comptés. Machine-scopée par
  construction.

**VERDICT (Romain, 2026-07-19) : MORT-a ARCHITECTURALE à cette enveloppe.**
Conséquences mécaniques : F1 tranche-1 = verdict AVEC MORT ; la spec v0 ne fait pas
foi (gate iii §8) ; campagne r_fovea toujours gatée (garde §7) ; tranche-2 (M-b)
non achetable (E3). Le re-épinglage du quadruplet est la décision neuve prévue par
le gate §6 — jamais un contournement.

**Direction nommée pour le re-épinglage (intuition Romain, consignée) :** les
fenêtres de Harten sont OBLIGATOIRES, mais le raffinement fin ne doit exister qu'à
un NOMBRE LIMITÉ d'emplacements — un cap dur de fenêtres actives respecté par le
routeur. Chiffrage au modèle mesuré (T ≈ 5.06 × (c_sys/2) × (n_fov/512)² ms par
fenêtre-triplet) : le budget 16.7 ms achète ~10 fenêtres 512²-équivalent à c=8
(~2.6 M cellules fines actives), ~20 à c=4 — contre 27 dans l'enveloppe morte.
CONVERGENCE nommée avec le pilier mesuré S_eff 1.8–3.9 (journal 2026-07-01) : le
raffinement effectif a toujours été épars ; c'est l'épinglage §6 qui était dense
(majorant). La mort tue le majorant, pas le pilier.

**Reste dû** : M-d (génération nuit + load) pour compléter le dossier factuel de
tranche-1 (faits robustes au re-épinglage) ; F0-cloud (prochaine session cloud) ;
puis SÉANCE DE RE-ÉPINGLAGE paper-grade du quadruplet §6 (spec re-versionnée,
addendum — aucun code avant).

### §A15-lecture-M-d + CLÔTURE TRANCHE-1 (2026-07-19)

**Génération E6 (non-verdictale, consignée)** : 3600 commits en 2.84 h (enveloppe
2–5 h tenue), cadence stable 2.83–2.90 s/commit, zéro dérive. Ledger : **1.17
Ko/commit** (4.2 Mo/3600) vs enveloppe gravée ~3.2 Ko — cohérent, nommé : l'enveloppe
chiffrait le commit plein-domaine (cap 409.6), la chaîne générée est fenêtrée
(k_fen=64, §A14). Snapshot : +0.1 Mo (commit plein 64², cohérent §A13-4-1).

**Lecture M-d (run_f1_md.py, natif)** : load SANS snapshot médiane **0.063 s**
(1er/froid 0.051) ≪ 30 s gravé — **MORT-d NON déclenché** (critère re-scopé
« coût du ledger », quasi-imperdable, dit en face dès E7 : il aurait attrapé un
désastre de parse, il n'y en a pas ; 7200 entrées, tous sha256 relus).
**Diagnostic-test de la claim E7 : CONFIRMÉE** — écart avec/sans snapshot 0.011 s
< 0.5 s (attendu ≈ 0 pré-enregistré). La note « compaction = disque/rétention, PAS
du load » (consigne E7-3) est désormais MESURÉE, plus seulement raisonnée.
**Portée nommée** : segment courant = 0 épisode (ledger clos sur un commit) — le
terme replay d'E7 non exercé ; pire cas mi-segment CALCULÉ : ~0.063 + 1 épisode
CPU (~2.8 s, mesuré à la génération) ≈ 2.9 s, ×10 sous seuil. `[MESURÉ à segment
nul ; pire cas = arithmétique]`

**CLÔTURE DU DOSSIER TRANCHE-1 :**
- **M-a : MORT ARCHITECTURALE** (§A15-lecture-M-a′, prononcée) — au quadruplet §6
  majorant, sur 3050 Ti ; modèle de coût ancré 6.43 ns/cellule/frame.
- **M-c : PAS DE MORT** — 2.784 ms < 4.2 ; PASS porté par la remontée seuillée
  (EPS_DETAIL suspect nommé tranche-2).
- **M-d : PAS DE MORT** — 0.063 s ≪ 30 ; claim E7 confirmée au diagnostic.
- **Verdict tranche-1 : AVEC MORT** (une, architecturale, portée stricte).
  Conséquences en vigueur : spec v0 ne fait pas foi ; r_fovea gaté ; tranche-2
  (M-b) non achetable sur cette enveloppe.

**File d'attente (aucun enchaînement) :** (1) séance de RE-ÉPINGLAGE du quadruplet
§6 — entrée : claude/annexe-enveloppe-jeu-2026-07-19.md (fluide-reduit b7bef0a),
sortie : quadruplet-jeu candidat + M-a-ter pré-enregistré (coût : minutes, harnais
existant) ; (2) F0-cloud à la prochaine session cloud (persister l'état divergent) ;
(3) tranche-2/M-b re-scopée à l'enveloppe re-épinglée, SI M-a-ter sans mort.

## §A16 — RE-ÉPINGLAGE du quadruplet §6 : quadruplet-jeu V2 + pré-enregistrement M-a-ter (gravé 2026-07-19)

> **Statut : gravé sur endossement Romain (R1-R4), AVANT toute mesure M-a-ter.**
> Décision prévue par le gate §6 suite au verdict MORT-a (§A15-lecture-M-a′).
> Brouillon de travail : fluide-reduit claude/reepinglage-quadruplet-2026-07-19-
> BROUILLON.md. Spec : addendum §6-rev1 (pointeur, cette section fait foi).

**Changement de STRUCTURE.** L'enveloppe dense (γ₂·n_fov²·N_niv) est remplacée par
un **cap dur d'emplacements** : le budget est une liste de SLOTS (niveau, c), somme
des coûts ≤ 16.7 ms (T2 INCHANGÉ). Ancre : 1 slot ≡ fenêtre 512² = 1.685 ms à c=8,
0.843 ms à c=4 `[MESURÉ M-a′ + linéarité structurelle en systèmes]`.

**R1 — Quadruplet-jeu V2 (c dégressif) :** fovéale 9 slots (2 niveaux fins à c=8,
7 supérieurs à c=4) + **3 slots énergie c=8 fins** (la réserve gameplay, ENFIN un
poste budgétaire explicite). **Coût prédit : 14.33 ms** (9.271 fovéale + 5.055
énergie), marge 14.2 %. Résidence prédite ≪ 1.35 Go (12 slots ≈ 3.15 M cellules).

**R2 — Scénario épinglé v1-jeu :** 2D mono-observateur, FOV 20°, n = 512 cellules
d'arc (≈3.75 px/cellule à 1920), J = 9 sous h0, monde h0 = 500k cellules (~707 de
côté) ⇒ finest ≈ h0/512, cône couvrant le monde entier. Machine : 3050 Ti (GPU
minimal 1920 ; enveloppe 4K sur GPU ×4-5 dérivée en annexe, TRANSPOSITION).

**R3 — Prédiction gravée AVANT run (test du modèle de coût) :** 14.33 ms ± 15 %
⇒ [12.18, 16.48] ms. Hors bande = AUTRE remonté (modèle linéaire en slots faux —
information, pas échec). COHÉRENCE NOMMÉE : la bande entière est SOUS 16.7 — si le
modèle tient, M-a-ter passe ; une mort impliquerait AUSSI un AUTRE du modèle.

**R4 — Dette σ_ω RECONDUITE, condition de réveil AFFINÉE :** le pied n°5
(cadencement temporel — M-a/M-a-ter mesurent UNE application de F par fenêtre par
frame, `[NON-ANCRÉ : hypothèse de design]` ; la CFL locale sur/sous-cadencerait
par niveau) est un CONSOMMATEUR TEMPOREL NOMMÉ. Le fork temporel rouvre QUAND la
spec décide le cadencement, avec la question perceptuelle du sous-cadencement fin
en main (branche (a″) préférée). Aucun réveil silencieux, aucun réveil prématuré.

**M-a-ter — PRÉ-ENREGISTRÉ (chiffres figés ici) :**
- Protocole : harnais tranche-1 + kernel fusionné M-a′, config = V2 (liste de
  slots (niveau, c) paramétrique — extension mineure du driver requise, revue
  avant run), fovéa mobile E4c, chrono B6, vérifs #1/#2 reconduites, natif.
- **MORT : médiane frame-time > 16.7 ms ⇒ le quadruplet-jeu V2 est mort** (seuil
  T2, inchangé — aucun seuil ne bouge).
- Lecture du modèle : médiane hors [12.18, 16.48] ⇒ AUTRE remonté.
- Diagnostic non-verdictal : V1 (c=4 partout, 14 slots) mesurée aussi — coût
  marginal nul, information spec.
- Gate résidence reconduit (< 1.35 Go).

**Portabilité tranche-1 (pas de re-mesure)** : M-c PASS porté a fortiori (moins de
fenêtres ; dépendance EPS_DETAIL reconduite) ; M-d PASS porté tel quel ; M-b/
tranche-2 re-scopée à V2, achetable SEULEMENT si M-a-ter sans mort ; F0-cloud dû.

**La spec re-fait foi quand** : (i) M-a-ter SANS MORT sur V2 ; (ii) F0-cloud lu ;
(iii) M-b sans mort sur V2. (Reformulation du gate iii §8 à l'enveloppe re-épinglée.)

**Les 5 pieds non mesurés consignés** : niveau 0 CPU 500k ; cube 3D ; fenêtres
d'énergie (désormais budgétées V2) ; r_fovea (arbitre n/J) ; cadencement temporel
(R4). Aucun n'est crédité d'avance.

### §A16-complément — Blocage de parité résolu + lecture R3 précisée (2026-07-19, AVANT tout run M-a-ter)

Le build M-a-ter a exposé un blocage réel (remonté, pas contourné — le driver a
refusé fort plutôt que de padder) : V2 = 17 systèmes, impair ; la garde de
`pas_f_fusionne` exigeait des paires. **Décisions Romain (gravées avant run) :**

**S1 — Garde élargie à {1, 2} systèmes, avec TROIS VERROUS exigés :** (a) diff
limité à la garde Python — kernel CUDA à diff vide ; (b) test de BIT-IDENTITÉ du
chemin 2-systèmes avant/après (c'est lui qui porte l'ancre M-a′) ; (c) test
d'équivalence du chemin 1-système contre le jetable mono-système. Motif gravé :
la parité est un artefact d'INSTRUMENT (un assert, pas le calcul) — modifier le
design épinglé pour satisfaire l'instrument serait l'inversion interdite ; c
dégressif produira des comptes impairs en permanence. Le padding (surcompte
0.843 ms = autre config) et la recomposition à 18 (design piloté par l'assert)
sont REJETÉS — un 4e slot énergie reste possible PLUS TARD comme décision de
design à son propre moment, découplée. Précédent de méthode : complément-3
(amender sa propre consigne, avec preuves et trace). Le chemin 1-système est
NEUF : son coût est un objet de la mesure (les 7 slots c=4 SONT des systèmes
seuls — le design), sa correction est testée.

**S2 — Lecture R3 PRÉCISÉE avant run :** le modèle (14.33 ms) prédit du CALCUL
PUR (ancre M-a′) ; la frame du harnais inclut les transferts (~1.2–1.5 ms à V2,
M-c réduit 27→12 fenêtres `[calculé]`). Donc : **bande [12.18, 16.48] appliquée à
la composante calcul** (médiane frame − transferts médians, les DEUX reportés) ;
**MORT sur la médiane frame COMPLÈTE > 16.7** (T2 est un budget de frame,
transferts compris). Sans cette précision, un AUTRE prévu d'avance viderait le
test du modèle. Gravé avant toute donnée = pré-enregistrement.

**S3 — Répartition des 3 slots énergie : 2 au niveau le plus fin + 1 au second** —
seule géométrie compatible B3 (deux offsets par niveau), nommée par Claude Code,
endossée.

### §A16-lecture-M-a-ter (2026-07-19) — MORT V2 + AUTRE DU MODÈLE (×2.13) + sonde d'attribution pré-enregistrée

**Lecture mécanique (run_f1_ma_ter.py, natif)** : V2 frame complète **31.097 ms**
> 16.7 ⇒ **MORT du candidat V2** ; composante calcul **30.484 ms** hors bande
[12.18, 16.48] ⇒ **AUTRE : le modèle linéaire en slots a un terme manquant de la
taille de F lui-même** (×2.13). La cohérence R3 a fonctionné comme gravée : la
mort est venue AVEC la faute du modèle — le prononcé utile est l'AUTRE.
V1_diagnostic : 26.259 ms (prédit 11.802, même faute). Transferts V2 : **0.613 ms**
(≪ 4.2 — le schéma diff à 12 fenêtres excellent) ; résidence 0.311 Go.

**Diagnostic (étiquette : HYPOTHÈSE ARITHMÉTIQUE sur 2 points)** : résidu
(mesuré − F prédit) = 16.2 ms (V2) / ~13.8 ms (V1) ; rapporté aux cellules-champs :
0.91 / 0.94 ms/M — **le résidu scale avec les cellules-champs**, signature d'un
coût par-cellule non modélisé. Suspect : la MACHINERIE DE PYRAMIDE (prédiction
Harten, extraction/seuillage/compaction B4, mise à jour de référence) en CuPy non
fusionné — mesurée F-seul en M-a′, noyée par le naïf en M-a, émergée à égalité
avec F fusionné. La même maladie que le F naïf, au stade précoce.

**Le cap, relu honnêtement** : il interdit toute nouvelle escalade DU F — la
machinerie est une question NEUVE (jamais dans le modèle ni dans M-a′). Mais
« fusionner la chose suivante » est le tapis roulant que le cap craignait : AUCUNE
escalade n'est décidée ici. Décision Romain : SONDE D'ATTRIBUTION d'abord.

**SONDE D'ATTRIBUTION (pré-enregistrée ICI, avant tout code de sonde) :**
- Bras A : config V2, fovéa IMMOBILE, remontée B4 OFF — F fusionné seul dans la
  boucle pyramide. ATTENDU PRÉ-ÉCRIT : calcul ∈ [12.18, 16.48] (le modèle F) ⇒
  attribution CONFIRMÉE ; hors bande ⇒ l'hypothèse machinerie est FAUSSE,
  chercher ailleurs (AUTRE remonté).
- Bras B : config V2, fovéa MOBILE E4c, remontée B4 OFF — isole la prédiction de
  déplacement. Décomposition mécanique : coût(remontée) = V2_full − B ;
  coût(prédiction) = B − A.
- Chrono B6, natif, mêmes vérifs. SONDE = lecture remontée, AUCUN verdict,
  aucune décision d'escalade ou de design embarquée (l'attribution d'abord, le
  remède — design E4d ou borne fusionnée machinerie — en décision séparée).

### §A16-lecture-M-a-ter, addendum sonde (2026-07-19, AVANT le run) — lecture à deux questions

Le build de la sonde (fluide-reduit 38253d4, diffs vides prouvés sur kernel/chrono/
substrats) a nommé AVANT run trois postes per-niveau présents au bras A et absents
de l'ancre M-a′ (per-slot) : 18 lancements de kernel vs 2, 9 réductions CFL
synchrones vs 1, orchestration CPU per-niveau — comptés au JSON
(postes_residuels_bras_a), verrouillés par test. Conséquence : A peut sortir
LÉGÈREMENT au-dessus de la bande sans que l'hypothèse machinerie soit fausse.
**Lecture précisée en DEUX questions distinctes (gravée avant toute donnée) :**
1. **Question modèle-F** (inchangée) : A ∈ [12.18, 16.48] ⇒ le modèle F tient au
   bras A ; sinon, l'écart A − 14.33 est lu CONTRE les postes per-niveau nommés
   (petit ~1–2 ms = postes plausibles ; grand = AUTRE, le modèle F lui-même).
2. **Question attribution** (le but de la sonde) : l'hypothèse machinerie est
   CONFIRMÉE ssi (V2_full − A) ≥ **12 ms** (≈75 % du résidu 16.16) ; en dessous,
   elle est fausse ou partielle — AUTRE remonté. Décomposition reportée :
   prédiction = B − A, remontée = 31.097 − B, somme de contrôle.
Aucun seuil de verdict là-dedans : la sonde reste une lecture remontée.

### §A16-lecture-sonde-attribution (2026-07-19) — décomposition acquise, hypothèse machinerie PARTIELLE

**Lecture mécanique (run_f1_sonde_attribution.py, natif, critères de l'addendum
5eac9bf)** : A_f_seul = **20.438 ms** (p99 21.773) ; B = 22.574 ;
décomposition : **F seul 20.438 / prédiction 2.135 / remontée B4 8.523** ;
somme de contrôle = 31.096 ✓ (vs full gravé 31.097).
- **Q1 (modèle F) : AUTRE** — A hors bande, écart +6.11 ms ≫ les 1–2 ms des
  postes per-niveau nommés au build. Le modèle F per-slot ne tient pas au
  multi-niveaux tel qu'orchestré. Suspects rangés (HYPOTHÈSES) : (i) bloc
  1-système > moitié d'un 2-systèmes (la « linéarité structurelle en systèmes »
  n'a jamais été MESURÉE ; occupancy 262k vs 524k threads) ; (ii) 9 syncs CFL
  (float(max()) = aller-retour device par niveau) ; (iii) 18 lancements non
  batchés — B5 prévoyait lui-même que le moteur réel batcherait, et TOUS les
  niveaux sont à 512² (batchables par construction).
- **Q2 (attribution) : PARTIELLE, non confirmée au seuil** — V2_full − A =
  10.66 ms < 12. La machinerie est réelle (66 % du résidu) mais 6.11 ms vivent
  dans F-multi-niveaux. La lecture arithmétique d'avant-sonde (§A16-lecture-
  M-a-ter) était partiellement fausse — consigné tel quel, c'est ce que la
  sonde devait départager.
- **Constat structurel** : même à machinerie NULLE, A = 20.4 > 16.7 — le chemin
  vers le budget passe OBLIGATOIREMENT par F-multi-niveaux, la remontée seule
  ne suffit pas.

**Décision Romain (2026-07-19) : SÉANCE DESIGN (E4d + orchestration), avec deux
micro-sondes pré-enregistrées en ouverture (§0 de la séance)** :
- **s1 — ancre du bloc 1-système** : micro-mesure M-a′-style (1 niveau, B=1,
  S=1) — mesure la linéarité supposée ;
- **s2 — F batché inter-niveaux + CFL asynchrone** : bras A re-mesuré avec
  l'orchestration que B5 prévoyait (2 groupes de shapes {1,2}-sys, ~4 lancements/
  frame, réduction CFL restant on-device) — départage artefact d'orchestration
  vs coût architectural dans les 6.11 ms.
Les sondes SERVENT la séance (ancrer le design), elles ne la retardent pas —
critères et attendus à figer au brouillon de séance, AVANT tout code.

## §A17 — Séance design F-multi-niveaux + machinerie : OUVERTE (gravé 2026-07-19)

> Brouillon endossé : fluide-reduit claude/seance-design-f-machinerie-2026-07-19-
> BROUILLON.md. Entrée : décomposition mesurée §A16-lecture-sonde-attribution
> (F 20.438 / prédiction 2.135 / remontée 8.523, budget 16.7 T2 inchangé).

**G1 (Romain) — micro-sondes d'ouverture ENDOSSÉES, pré-enregistrées :**
- **s1 — ancre 1-système** : (a) B=1,S=1 isolé ; (b) B=7,S=1 batché. Kernel
  intouché, chrono B6, natif. Les ancres MESURÉES remplacent l'étiquette
  « structurelle » (hors ±10 % de 0.843 ⇒ linéarité fausse, modèle re-calibré
  sur mesures).
- **s2 — F batché inter-niveaux + CFL async** : bras A re-orchestré (B=5 2-sys +
  B=7 1-sys, 4 lancements/frame vs 18 ; CFL payée on-device, sync CPU artefact
  supprimée). LÉGITIMITÉ vs cap endossée : pas une escalade kernel — B5
  prévoyait le batch du moteur réel ; le per-niveau était plomberie de harnais.
  Lecture : A_batché dans ±10 % de (5×1.685 + 7×ancre_s1) ⇒ l'écart +6.11 était
  l'orchestration, modèle re-calibré TIENT ; hors ⇒ AUTRE remonté. **s2 fixe le
  budget machinerie = 16.7 − A_batché.**
**G2 — leviers à l'étude après s1/s2 : L1 (remontée cadencée — le choix de k
touche la condition de réveil σ_ω R4, décision explicite à son moment) + L3
(détails émis par le kernel — build kernel NEUF avec son propre
pré-enregistrement ; la borne M-a′ reste intacte et citée telle quelle).**
L2 (activité — non mesurable sur harnais actuel) et L4 (prédiction 2.135 ms —
falsificateur naturel : bras s2-mobile) restent nommés, non instruits.
**G3 — porte budget 33.3 ms OUVERTE-NOMMÉE** : pas décidée ; ne s'étudierait
que par re-épinglage assumé (la mort de V2 à 16.7 reste gravée).

### §A17-lecture-s1-s2 (2026-07-19) — modèle re-calibré TIENT ; budget machinerie 2.268 ms ; G2 AMENDÉ

**s1 (run_f1_s1_ancre_1sys.py, natif)** : ancre 1-système ISOLÉE 0.968 ms/slot
(+14.8 %, HORS bande — un bloc seul paie sa granularité de lancement) ; BATCHÉE
(B=7) **0.845 ms/slot** (+0.2 %, dans la bande) — la linéarité en systèmes est
VRAIE en batché, l'étiquette « structurelle » est remplacée par deux mesures.
Ancre opérante : 0.845 (le moteur batché ne paie pas l'isolement).

**s2 (run_f1_s2_batche.py, natif)** : **A_batché = 14.432 ms** (p99 15.808) vs
prédiction re-calibrée 14.341 (5×1.685 + 7×0.845) — **+0.6 %, le modèle TIENT**.
Les +6.11 ms du bras A étaient l'orchestration (effet CUMULÉ des trois artefacts,
sans départage — nuance verrouillée respectée). Audit CFL : finies, positives,
identiques par groupe. **Budget machinerie = 16.7 − 14.432 = 2.268 ms.**

**Chiffre inconfortable en évidence** : la prédiction MESURÉE (2.135 ms,
§A16-lecture-sonde) ≈ le budget entier (2.268) — L1+L3 seuls ne peuvent PAS
fermer l'arithmétique même à remontée nulle. MAIS la prédiction a été mesurée
sous l'ANCIENNE orchestration per-niveau (la maladie que s2 vient de guérir
pour F) — falsificateur pré-nommé au §A17 : bras s2-mobile.

**Décision Romain (2026-07-19) — G2 AMENDÉ : L4 instruite via le bras s2-mobile,
pré-enregistré ici** : config V2, orchestration batchée s2, fovéa MOBILE E4c,
remontée OFF. Lecture mécanique : **prédiction_batchée = A_mobile − 14.432** ;
cible L1+L3 = 16.7 − 14.432 − prédiction_batchée, reportée sans interprétation.
La décision (le design est-il jouable à cette cible, ou candidat/porte 33.3 par
la porte de devant) revient à Romain à la lecture. Sonde : aucun verdict.

### §A17-lecture-s2-mobile (2026-07-19) — cible L1+L3 NÉGATIVE ; décision : L4-design prédiction GPU-side

**Lecture (run_f1_s2_mobile.py, natif)** : A_mobile = **17.251 ms** (p99 19.352) ;
prédiction batchée = **2.819 ms** (vs 2.135 per-niveau — elle MONTE) ; **cible
L1+L3 = −0.551 ms : NÉGATIVE**. F + prédiction dépassent 16.7 AVANT toute
remontée — l'arithmétique V2@16.7 est fermée-négative, mesurée terme à terme
(14.432 + 2.819 + remontée>0). Audit CFL propre. Nuance de cadence dyadique
consignée au build (frame typique = 1-2 niveaux fins ; pire cas au p99).

**Hypothèse de mécanisme (étiquetée, non prouvée)** : la marge du bras B vivait
dans un régime déjà serialisé ; dans le pipeline async de s2, chaque intervention
CPU (prédiction + H2D par niveau) cale un GPU libre — coût marginal supérieur.

**Fait de périmètre (vérifiable aux gravés)** : le harnais prédit TOUTES les
colonnes CPU depuis le niveau 0 (B4, choix de harnais) ; la spec §5 prévoit la
descente parent→enfant, parents des niveaux 2..8 SUR GPU (seul le niveau 1 a un
parent CPU). Le 2.819 mesure une SIMPLIFICATION DE HARNAIS — majorant de la
topologie spécifiée. Prédiction GPU-side = design conforme au gravé,
[TRANSPOSITION] tant que non mesurée.

**Décision Romain (2026-07-19) : L4-design — prédiction GPU-side, pré-enregistrée :**
- Build : prédiction des colonnes entrantes SUR GPU depuis les parents GPU
  (niveaux ≥2) ; niveau 1 seul prédit CPU depuis monde 0 (+ son H2D). Chemin
  VIVANT (Option A : f32 GPU légitime) ; verrous : aucune omission (mêmes
  colonnes, mêmes écritures fenêtre+référence — comptage type s2-mobile),
  équivalence à la prédiction CPU à tolérance f32 nommée. Kernel F intouché.
- Mesure (bras s3) : config s2-mobile, prédiction GPU-side. Lecture mécanique :
  prédiction_gpu = A_s3 − 14.432 ; **cible L1+L3 = 16.7 − A_s3**, reportées sans
  interprétation. Sonde — la décision (design jouable / candidat aminci / porte
  33.3) revient à Romain à la lecture.
- Portes 33.3 (motif T2 : où vit le rendu — à traiter explicitement si ouverte)
  et candidat aminci : RESTENT ouvertes-nommées, non instruites.

### §A17-s3, addendum pré-run (2026-07-19) — deux trous de SPEC surfacés par le build, périmètre consigné

Le build s3 (fluide-reduit 8f0f371, kernel intouché, 189 tests) a établi AVANT
run, sur 400 positions de balayage : le pré-enregistrement supposait tout slot
≥2 doté d'un parent GPU — FAUX pour 5 slots. **Deux questions de niveau SPEC
ouvertes, à traiter au paper-grade (pas ici) :**
1. **EMBOÎTEMENT** : la géométrie d'énergie B3 (±n_fov) place les fenêtres hors
   couverture parente — or Harten exige l'emboîtement, propriété JAMAIS écrite
   en §2. En moteur réel, une fenêtre raffinée sous le plafond devrait vivre
   dans la région raffinée de son parent. À graver en spec.
2. **c DÉGRESSIF vs DESCENTE** : niveau 8 (c=8) sur parent niveau 7 (c=4) — les
   champs supplémentaires n'ont pas de parent à prédire. L'interaction du
   levier « c décroissant » (D3/§2) avec la prédiction Harten n'a jamais été
   spécifiée. À graver en spec.
**Périmètre s3 consigné** : 7 slots GPU-side (fovéaux 2-7 et 9), 5 CPU+H2D
(niveau 1 gravé ; fovéal 8 [c-mismatch] ; 3 énergie [pas de parent]) — chaque
repli avec sa raison au JSON, AUCUN contenu parent fabriqué (majorant honnête).
Frame médiane : 1 GPU-side / 2 CPU — **baisse attendue PARTIELLE par
construction**, verrouillée par test. **Décision Romain : COURIR TEL QUEL** —
s3 mesure l'ancre du mécanisme GPU-side ; une V2 ré-emboîtée se prédira sur
ancre mesurée avant d'être construite. Re-périmètre : levier suivant, nommé.

### §A17-lecture-s3 + CLÔTURE DE LA CASCADE DE SONDES (2026-07-19)

**Lecture (run_f1_s3_pred_gpu.py, natif)** : A_s3 = **17.002 ms** (p99 19.673) ;
prédiction GPU-side = **2.570 ms** (−0.249 vs CPU-side 2.819) ; **cible L1+L3 =
−0.302 ms, toujours NÉGATIVE**. Baisse partielle par construction, confirmée
(1 conversion médiane sur 3). Audit CFL propre.

**Garde anti-tapis-roulant EXERCÉE (le critère gravé : quand la prochaine mesure
ne peut plus changer la décision)** : l'ancre est AMBIGUË — 0.249 ms pour une
conversion ne départage pas deux modèles (coût per-slot linéaire vs structure de
stalls s'effondrant d'un coup) qui collent aux mêmes données ; et la marche
suivante (ré-emboîtement) n'est plus une sonde mais un build de géométrie
EXIGEANT d'abord les deux résolutions de spec. **La cascade de sondes est CLOSE.**

**Acquis de la cascade (tous mesurés, tous gravés)** : V2@16.7 fermée-négative
terme à terme (F batché 14.432, modèle +0.6 % ; prédiction ≥ 2.5 dans toute
configuration mesurée ; remontée 8.523 non financée) ; mécanisme GPU-side
fonctionnel et verrouillé ; orchestration batchée = celle du moteur (les 6.11 ms
du per-niveau étaient de la plomberie) ; linéarité en systèmes vraie en batché
(0.845) ; deux trous de spec nommés (emboîtement §2, c dégressif vs descente).

**Décision Romain (2026-07-19) : PROCHAINE VRAIE SÉANCE = PAPER-GRADE** —
(1) graver l'EMBOÎTEMENT en §2 et trancher c-dégressif-vs-descente (les deux
bloquent tout moteur V2, quelle que soit la porte) ; (2) RE-ÉPINGLER un candidat
avec toutes les ancres mesurées — aminci, ré-emboîté, et/ou porte 33.3 avec le
motif T2 (où vit le rendu) traité EN FACE ; (3) pré-enregistrer son M-a-quater.
Aucune sonde, aucun build avant ce paper-grade. F0-cloud reste dû (session cloud).

## §A18 — Séance papier post-cascade : propriété E, champs d'échelle, candidat V4, M-a-quater (gravé 2026-07-19)

> Brouillon endossé : fluide-reduit claude/seance-papier-2026-07-19-BROUILLON.md.
> P1-P5 tranchés (Romain). Spec : addendum §2-rev1 (pointeur, cette section fait foi).

**P1 — PROPRIÉTÉ E (emboîtement), gravée** : l'ensemble actif est un ARBRE —
toute fenêtre active au niveau j ≥ 1 déclare une parente au niveau j−1 couvrant
intégralement son empreinte (empreinte = quart de fenêtre parente : une parente
couvre jusqu'à 4 enfants). RÈGLE DE PLACEMENT : une fenêtre d'énergie ne naît
que dans l'union des parentes ; hors couverture, le routeur monte une TOUR
d'ancêtres COMPTÉE au cap — le budget d'emplacements est un budget d'ARBRE.
Les offsets B3 ±n_fov étaient du harnais, remplacés.

**P2 — CHAMPS D'ÉCHELLE, gravés** : les champs au-delà de c_parent sont DÉFINIS
à contenu grossier nul (nés de l'échelle, comme les détails de Harten) —
prédiction parent→enfant = remplissage-zéro, EXACT par définition. Contrainte
de substrat gravée : un champ à contenu grossier existe à TOUS les niveaux.
Le trou c-dégressif est éliminé, pas bouché.

**P3 — CANDIDAT V4** : fovéale 9 slots (2 fins c=8 + 7 c=4) + 2 slots énergie
c=8, placement emboîté, champs d'échelle. Scénario R2 inchangé. **F batché =
12.655 ms** [ancres MESURÉES : 1.685 / 0.845]. Budget 16.7 (T2 inchangé).
**Porte 33.3 = REPLI PRÉ-NOMMÉ** avec motif T2 traité : physique 30 Hz + rendu
60 fps par interpolation du readout, cohabitation GPU dans les ~13 ms restants
par fenêtre [design nommé, non décidé, non gratuit].

**P4 — L1 EN RÉSERVE pré-enregistrée** : cadence k=2 si L3 ne suffit pas — k
choisi par BUDGET, étiqueté [NON-ANCRÉ perceptuel], réveil σ_ω DIFFÉRÉ (R4).

**P5 — M-a-quater PRÉ-ENREGISTRÉ (chiffres figés)** :
- Protocole : pipeline COMPLET V4 — F batché + prédiction GPU-side emboîtée +
  remontée L3 (kernel-moteur émettant |d| ≥ EPS_DETAIL, 1e-4 reconduit
  [NON-ANCRÉ], suspect tranche-2 inchangé) + transferts ; fovéa mobile E4c,
  chrono B6, vérifs reconduites, natif.
- **MORT : médiane frame complète > 16.7** ⇒ V4 mort ⇒ le repli 33.3 devient LA
  décision à prendre (jamais un enchaînement).
- **Bande à deux modèles : frame prédite ∈ [13.9, 17.2]** — chevauche le seuil,
  assumé : le papier rend la question mesurable, il ne promet pas. Hors bande =
  AUTRE. Diagnostics : parts F / prédiction / remontée / transferts (attribution
  mécanique — la série de sondes N'EST PAS rouverte).
- **Gates de build AVANT tout code** : chiffrage Claude Code remonté — le
  kernel-moteur L3 est un KERNEL NEUF avec son propre pré-enregistrement
  (équivalence de motif au fusionné + émission), la borne M-a′ INTACTE et citée
  telle quelle ; géométrie emboîtée ; champs d'échelle. Chiffrage > 2 séances ⇒
  remonter avant achat.

### §A18-complément — Gate d'achat M-a-quater tranché : DÉCOUPE A/B, achat A seul (2026-07-19)

Chiffrage remonté (fluide-reduit claude/chiffrage-ma-quater.md) : 2.5–3.5
séances > cap 2 ⇒ gate exercé. **Décisions Romain :**

**ACHAT A SEUL (1.1–1.7 séance)** : borne L3 — kernel-moteur émettant
|d| ≥ EPS_DETAIL en épilogue + compaction + transfert, mesurée SEULE (comme
M-a′ borna F), géométrie intouchée. **B (géométrie emboîtée, champs d'échelle,
driver, flag L1) GATÉ sur la lecture de A** : si la borne ne descend pas vers
~1.5 ms, V4 meurt pour un achat au lieu de trois et le repli 33.3 devient LA
décision. Plausibilité consignée (session critique) : les passes mémoire CuPy
≈ 6.5 des 8.523 ms sont ce que l'épilogue supprime ; résidu = compaction
(~0.2–0.5 calculé) + transfert (0.61 MESURÉ à 12 slots) — borne plausible à
1.0–2.0 ms, pas confortable : la mesure tranche.

**Protocole endossé (4 points)** : (1) compteur de compaction lu à RETARD D'UNE
FRAME — pas de sync par frame (la maladie s2 ne rouvre pas) ; lag d'une frame
étiqueté chemin vivant sous-JND ; (2) le deuxième système reste MOUILLÉ au
harnais (majorant E4a reconduit) — seule la PRÉDICTION le traite en champ
d'échelle ; vérification anti-minoration explicite (un système nul = fenêtre
sèche sans divergence de warp, minorerait la mesure) ; (3) parts F/prédiction/
remontée/transferts DÉRIVÉES des ancres gravées + comptage exact — étiquetées
[DÉRIVÉ], la MORT restant sur la frame médiane MESURÉE ; la série de sondes
reste close ; (4) garde-fous L3 : choix d'implémentation FIGÉS au protocole
avant la première ligne, **CAP : L3 = dernière escalade sur la remontée**
(comme M-a′ sur F), borne écrite comme mesure JETABLE et dite telle quelle.

### §A18-complément-2 — Arbitrage compaction L3 (2026-07-19, avant toute ligne de code)

Point d'arrêt honoré à l'étape 1 de l'achat A (prereg kernel L3 rédigé, zéro
code). **Décision Romain : AGRÉGATION PAR WARP** (__ballot_sync + __popc, un
atomique par warp) contre l'atomicAdd naïf par émission. Motifs consignés :
(i) précédent M-a′ — la borne mesure un design compétent, pas une file
d'attente (17.8 M atomiques sérialisés = mesurer la contention, pas L3) ;
(ii) asymétrie du regret sous cap — un échec agrégé se lit proprement, un échec
naïf serait illisible (contention vs design) ET invérifiable à jamais ;
(iii) le naïf ajoute un coût n'appartenant à aucun terme de la question.
Le choix fait partie du PROTOCOLE : un chiffre décevant se lira « ce design,
compacté ainsi, coûte tant » — jamais « il aurait fallu mieux implémenter ».

**Consigne session critique, jointe à la dérogation du tri** : la comparaison
triée (l'ordre atomique n'étant pas garanti) pourrait masquer une DOUBLE
ÉMISSION — le verrou vérifie l'UNICITÉ DES INDICES avant tri (gratuit : chaque
élément n'émet qu'une fois par frame par construction). Dérogation isolée à ce
chemin, motivée, verrouillée. Choix figés sans arbitrage reconduits : émission
par élément (schéma B4), dimensionnement pire-cas (142.6 Mo, gate large),
retard d'une frame avec écart émis/transférés REPORTÉ, deux bras (fusionné
intouché = contrôle de dérive vs 14.432 gravés ; L3), fovéa immobile.
**Étape 2 (build) débloquée.**

### §A18-lecture-borne-L3 + décision achat B (2026-07-19)

**Lecture (run_f1_borne_l3.py, natif)** : remontée_L3 = **5.122 ms**
(A_L3 19.398 − A_REF 14.277) — L3 gagne ×1.66 sur le CuPy (8.523), manque la
cible ~1.5 ×3.4. Dérive machine −1.1 % (contrôle propre). Tailles d'émission
INSTABLES (×2.3, reporté — le retard d'une frame n'est pas neutre, dernière
frame en attente + bursts). **Sec vs mouillé : −16.9 %** — l'anti-minoration a
mordu, la consigne « deuxième système mouillé » était load-bearing ; consignée
pour la lecture de B.

**Arbre pré-nommé appliqué** : L3 seul insuffisant ⇒ réserve L1 (P4).
Arithmétique sur ancres : k=2 ⇒ frame V4 ∈ [15.5, 17.2] (à cheval) ;
**k=4 ⇒ [14.2, 15.9] — sous le seuil sur toute la bande.**

**PIÈGE D'INTÉGRITÉ NOMMÉ ET FERMÉ (session critique)** : une remontée en
rafale toutes les k frames ferait passer la MÉDIANE en laissant un stutter
~20 ms invisible au critère — **L1 DOIT être ÉTALÉE (round-robin, 11/k
fenêtres/frame)** : coût lissé, critère médian honnête, même amortissement.
Consigne OBLIGATOIRE de l'achat B.

**Symétrie consignée** : les deux portes consomment une hypothèse
temporelle-perceptuelle (16.7+L1 : lointain à 15 Hz ; 33.3 : physique 30 Hz
interpolée) — la dette σ_ω rôde des deux côtés, réveil différé tenable tant que
le choix n'est pas design-final.

**Décision Romain (2026-07-19) : ACHAT B + L1 ARMÉE round-robin k=4** (choisi
par budget, [NON-ANCRÉ perceptuel], σ_ω différé R4/P4). M-a-quater tranche sur
mesure. Bande de M-a-quater RE-CALCULÉE avec L1 : **[14.2, 15.9]** (remplace
[13.9, 17.2] — recalibration pré-mesure sur ancres nouvelles, dont borne L3
5.122/4 amortie). MORT inchangée : médiane frame complète > 16.7. Porte 33.3 :
toujours repli pré-nommé.

### §A18-build-M-a-quater endossé (2026-07-19, AVANT run)

Build livré (fluide-reduit 7164d84 ; kernels F et L3, chrono, substrat_jetable à
diff vide ; empreinte F e8fcaad4…7f04 sur 8 commits ; pyramide.py en mode
PARALLÈLE, B3 restant défaut ⇒ sondes et M-a′ reproductibles au bit près).
**Endossé par Romain, avec trois faits consignés :**
1. **Géométrie emboîtée par IDENTITÉ, pas par réglage** : offsets d'énergie à
   ±n_fov/2, identité oy_j − 2·oy_parent = n_fov/2 exacte à tous niveaux ⇒ les
   2 fenêtres tombent aux bornes de la couverture parente, ZÉRO tour (400
   positions). `slots_sans_parente()` rend la propriété E FALSIFIABLE (vide en
   emboîté, non vide en B3) — le trou de s3 est refermé par construction.
   Prédiction : 10 slots GPU-side, niveau 1 seul CPU ; fail-loud si géométrie
   non emboîtée.
2. **Rattrapage d'ancre** : F prédit corrigé 12.641 → **12.655** (l'étiquette
   0.843 remplacée par l'ancre MESURÉE 0.845, s1 batchée) — concordance exacte
   au gravé, ancres nommées comme telles dans le driver.
3. **Choix d'implémentation ENDOSSÉ (remonté par Claude Code)** : L1 étalée
   oblige L3 et le fusionné à cohabiter ⇒ groupes scindés en segments contigus,
   **8–12 lancements/frame au lieu de 4, ≈ +0.03–0.08 ms MESURÉ et reporté par
   frame**. Seule voie ne touchant ni le kernel L3 ni le cap (les alternatives
   coûtaient des copies ou modifiaient L3). Surcoût 3 ordres sous les 6.11 ms
   de s2 — qui étaient des syncs CFL, absentes ici.

**Anti-minoration — précision de lecture gravée avant run** : le −16.9 % de la
borne L3 est un **MAJORANT de l'effet**, pas sa description — un test du build a
montré que le chemin est VIVANT (après prédiction à zéro, F repropage dès la
frame suivante depuis les voisins mouillés ; la colonne nulle ne le reste pas).
La fraction asséchée est MESURÉE et reportée ; l'assèchement réel est plus
faible que la borne ne le suggère. Rien corrigé en douce.

### §A18-lecture-M-a-quater (2026-07-19) — PAS DE MORT, mais V4 AU SEUIL ; AUTRE attribué ; M-b ouverte

**Lecture mécanique (run_f1_ma_quater.py, natif)** : médiane frame complète
**16.589 ms** — MORT (> 16.7) : **NON déclenchée**. p99 **19.652** (+3.063).
Propriété E = True (11 slots / 15 blocs, 10 GPU-side / 1 CPU) ; L1 k=4 étalée
[2,3,3,3], aucune frame ne remonte tout. **Hors bande [14.2, 15.9] : AUTRE
(4e fois).**

**Attribution COMPLÈTE de l'AUTRE (parts [DÉRIVÉ], somme = 16.589 ✓)** :
- F = 12.655 `[ancre, exact par construction]` ;
- **remontée = 1.280** vs 5.122/4 = 1.2805 prédits — **L1 étalée amortit à
  quatre chiffres** : le modèle d'amortissement est CONFIRMÉ ;
- **prédiction ≈ 0** (−0.073 = limite de la dérivation par soustraction, valeur
  non physique) — le modèle « effondrement des stalls » avait RAISON contre le
  modèle « résidu linéaire » : **la prédiction GPU-side emboîtée est GRATUITE**.
  Gain architectural acquis (propriété E + descente parent→enfant GPU) ;
- **transferts = 2.727** vs ~0.6 modélisés — **la TOTALITÉ du dépassement de
  bande est là** : terme repris d'une mesure PÉRIMÉE (0.613 à 12 slots, remontée
  CuPy seuillée, PAS émission L3). Le modèle n'est pas cassé : un terme était
  périmé.
- Assèchement champs d'échelle : **0.2 %** — le −16.9 % de la borne était bien
  un MAJORANT lâche (chemin vivant, F repropage) ; le fil anti-minoration se
  clôt : effet réel négligeable.

**RÉSERVE GRAVÉE, EN ÉVIDENCE (la lecture honnête)** : marge = **0.111 ms =
0.7 %**, INFÉRIEURE à la dérive machine mesurée le jour même (1.1 %,
A_REF 14.277 vs 14.432). Le seuil n'est pas déplacé et le critère n'est pas
franchi — mais ce qui est établi est « **V4 est AU seuil** », PAS « V4 tient le
budget ». Zéro tête pour les 5 pieds non mesurés (niveau 0 CPU 500k, gameplay
au-delà des 2 fenêtres d'énergie, r_fovea, cadencement, 3D), qui tous ajoutent.
**p99 +3.06 NON RÉSOLU** : L1 étalée a supprimé le stutter de CADENCE, pas
celui des BURSTS d'émission (tailles ×2.3, mesurées à la borne) — ~19.65 ms
rate le 60 fps sur ces frames. Nommé, non minimisé.

**Décision Romain (2026-07-19)** : sans-mort prononcé AVEC ces réserves ;
**tranche-2 / M-b OUVERTE** (gate (iii) ; achetable — tranche-1 et M-a-quater
sans mort). **Ajout au pré-enregistrement de M-b : BALAYAGE EPS_DETAIL** —
mêmes émissions, seuils multiples, coût marginal nul : M-b rend son verdict
Δχ live↔rederive ET la courbe qui arbitre le budget transferts + p99. Le
paramètre `[NON-ANCRÉ perceptuel]` suspect depuis §A15-complément consigne 1
cesse d'être réglé par budget : il devient MESURÉ. Un re-run de dispersion est
ÉCARTÉ (ne changerait ni la suite ni le levier — critère tapis-roulant).
Reste dû : F0-cloud (gate ii).

## §A19 — Sonde EPS (N1) + M-b re-scopée V4 (N2) : pré-enregistrées (gravé 2026-07-19)

> Brouillon endossé : fluide-reduit claude/prereg-mb-eps-2026-07-19-BROUILLON.md.
> Q1-Q3 tranchés (Romain). Gate (iii) de la spec = M-b sans mort ; gate (ii) =
> F0-cloud, toujours dû (~2 min, prochaine session cloud, indépendant).

**CONCESSION consignée (session critique)** : la formule du §A18-lecture
(« balayage EPS gratuit dans le run M-b, une mesure deux consommateurs ») était
FAUSSE SUR L'OBSERVABLE — EPS_DETAIL gouverne la fidélité du NIVEAU 0 VIVANT,
pas le contrat live↔rederive ; il n'atteint le Δχ de M-b qu'au second ordre (via
les colonnes entrantes, fovéa mobile) — signal bruité, impropre à arbitrer un
budget. Observable corrigé ci-dessous ; la correction rend N1 AUTONOME et
BON MARCHÉ, ce qui change aussi le séquencement (Q1).

**Q1 — SÉQUENCEMENT : N1 (~0.5–1 séance) AVANT N2 (3–5 séances, risque ÉLEVÉ)** —
N1 peut changer l'enveloppe dans laquelle N2 sera mesurée ; l'inverse est faux.

**N1 — SONDE EPS (pré-enregistrée)** :
- Vérité : niveau 0 par décimation exacte (moyennes Harten) de l'état fin.
  Vivant : niveau 0 reconstruit par la remontée seuillée à EPS, incrémentale,
  **L1 k=4 étalée INCLUSE** (la péremption d'amortissement fait partie du
  régime de production).
- Observable : **Δχ readout (albedo + delta_chi, max_carrier)** entre les deux —
  espace instrument, jamais l'état. Fovéa mobile E4c, série 300 frames, natif.
- Balayage : EPS ∈ {1e-5, 1e-4 (gravé), 3e-4, 1e-3, 3e-3, 1e-2}. Reporté par
  EPS : Δχ max et médian, octets/frame, temps de transfert, médiane ET p99 frame.
- **Q2 — RÈGLE DE DÉCISION PRÉ-ENREGISTRÉE** : EPS retenu = **le plus grand EPS
  dont le Δχ max de série reste < 0.0603 (ic_bas)** — lecture sur l'IC ENTIER
  (discipline §A13). Si aucun EPS du balayage ne satisfait (1e-5 compris) :
  **AUTRE remonté, aucun EPS retenu par défaut** — jamais de choix après courbe.
- **EXCLUSION NOMMÉE — balayage de k INTERDIT ici** : balayer k perceptuellement
  = décider le cadencement avec la question perceptuelle en main = **condition
  de réveil σ_ω (R4) ATTEINTE**. k reste FIGÉ à 4 `[NON-ANCRÉ, par budget]` ;
  bouger k un jour sera une décision explicite de Romain, jamais un effet de
  bord de sonde.

**N2 — M-b RE-SCOPÉE V4 (pré-enregistrée ; seuils gravés NON déplacés)** :
- Chemin vivant = pipeline V4 COMPLET (géométrie emboîtée, prédiction GPU-side,
  L3, L1 k=4 étalée, fovéa mobile) avec le **F FIDÈLE** (portage `run_episode` :
  wetdry O2 CFL-adaptatif, Exner, pulses, bathymétrie réelle) — pas le jetable.
  Chemin rederive = f(registre, seeds) CPU f64, INTOUCHÉ (gel = contrôle).
- Cellule (§A15, reconduite) : 3 seeds {101, 102, 103}, Δt=4, 6 émissions ;
  commits fenêtrés, k_fen aire-proportionnel, cap 10 %.
- **MORT-b (T3, inchangé)** : max de série Δχ live↔rederive **> 0.0733**,
  **PAR-SEED** ⇒ Option A morte, repli Option B = décision neuve.
- **Q3 — CONTRÔLE T1 DEVENU VERDICTAL** : la frame complète mesurée avec le F
  FIDÈLE **> 16.7 ms ⇒ V4 MORT** (critère pré-écrit AVANT tout chiffre — la
  marge M-a-quater étant de 0.7 %, laisser ce contrôle non-verdictal aurait
  permis de constater la mort sans la prononcer). Diagnostic joint : écart
  F fidèle vs proxy 12.655 (représentativité T1).
- **Gate de chiffrage** : chiffrage du portage remonté AVANT achat ; scission
  interne si > 3 séances.

**Portées** : rien sur r_fovea, 3D, multi-vue, niveau 0 CPU à 500k (pied n°1),
qualité produit (pin n=1), coût de l'Option B si M-b meurt.

### §A19-complément — Vérification d'instrument de la sonde EPS (gravé 2026-07-19, AVANT tout run)

Le build N1 (fluide-reduit fef6ef8, aucun fichier de src/ modifié, empreinte
kernel F identique sur 9 commits) a remonté deux faits AVANT run :

**(1) Bug d'offset attrapé et corrigé** : les indices émis par le kernel L3 sont
relatifs au tableau passé (la TRANCHE du tour round-robin), pas au groupe. Un
consommateur reconstruisant sans l'offset écrit tout dans le slot 0 ⇒ le vivant
des autres slots n'avance jamais ⇒ **la sonde aurait remonté AUTRE avec une
cause fausse**. Deux tests l'ont attrapé. **M-a-quater N'ÉTAIT PAS faussé** :
il ne reconstruit rien, il transfère des octets (scoping vérifié).

**(2) Risque de sonde MUETTE PAR CONSTRUCTION** : sur config réduite (non
reportée comme résultat), facteur ~2000 sur les coefficients remontés pour
**0.07 % de variation du Δχ max**, au-dessus du seuil partout. Signature d'un
observable saturé : à k=4 une fenêtre ne remonte qu'une frame sur quatre, et la
dérive de F entre deux remontées écrase la troncature du seuil. Un AUTRE
signifierait alors « à k=4, le niveau 0 vivant est infidèle QUEL QUE SOIT EPS »
— et non « EPS est trop grand » ; confondre les deux ferait resserrer EPS sans
effet, en payant du trafic pour rien.

**DÉCISIONS ROMAIN (avant toute donnée réelle) :**

**(A) VÉRIFICATION D'INSTRUMENT AJOUTÉE — bras k=1, même balayage EPS, DUE
AVANT toute lecture** (précédent §A14 : la sonde muette se détecte avant, pas
après). Lectures PRÉ-ÉCRITES : (i) si k=1 DISCRIMINE (Δχ répond à EPS et au
moins un EPS passe) ⇒ instrument VALIDE, l'AUTRE à k=4 est attribuable à la
PÉREMPTION ; (ii) si k=1 ne discrimine pas non plus ⇒ **sonde MUETTE sur EPS,
AUTRE D'INSTRUMENT — ne rien régler, remonter**. Nota : à k=1 subsiste une
frame de retard (compteur endossé) — la vérification teste la discrimination
sous péremption MINIMALE, pas nulle. Ce bras n'est PAS un balayage de k et ne
décide aucun cadencement ; toute DÉCISION sur k reste gatée σ_ω (R4).

**(B) ÉCHELLE DE LECTURE ÉPINGLÉE** : la règle Q2 lit le **Δχ DÉCIMÉ à
l'échelle du niveau 0** (l'objet qui existe et sera consommé : le grossier
comparé à ce que le grossier devrait être, à sa propre résolution). Le Δχ non
décimé est reporté en DIAGNOSTIC seul — la décimation déplaçant les bandes
porteuses que delta_chi lit, la relation n'est pas monotone.

**(C) LOGIQUE DU CRITÈRE, gravée (fait de logique, pas de préférence)** : le
joueur ne voit JAMAIS la référence. Donc Δχ < 0.0603 ⇒ **innocuité ÉTABLIE** ;
Δχ > 0.0603 ⇒ **innocuité NON ÉTABLIE** — jamais « nocivité établie ». Un AUTRE
ne justifie donc PAS de resserrer EPS ; le seul indice interne dont le joueur
dispose est la cohérence proche/lointain, et c'est une MESURE DIFFÉRENTE
(nommée, non armée).

**(D) BRANCHE PRÉ-ÉCRITE** : si l'instrument est VALIDÉ mais qu'aucun EPS ne
passe à k=4 ⇒ **la décision passe à k, et le RÉVEIL σ_ω devient LA décision
explicite à prendre** (R4 : le cadencement se déciderait avec une question
perceptuelle en main). L'EPS retenu serait alors celui que k=1 valide ; le
budget transferts se traite ensuite. Aucun réveil silencieux, aucune
reformulation après lecture.

### §A19-complément-2 — Table de branches FERMÉE : branche (iii) (gravé 2026-07-19, avant tout run)

Le build (fluide-reduit e54a730, `src/` intact, empreinte kernel F identique sur
10 commits) a rendu la validité STRICTE — conforme au gravé : instrument valide
= k=1 discrimine ET au moins un EPS passe. Cette rigueur expose un cas que ma
table de branches ne couvrait pas ; il est fermé ICI, avant toute donnée.

**BRANCHE (iii) — k=1 DISCRIMINE mais AUCUN EPS ne passe (même 1e-5)** :
lecture pré-écrite = **« MÉCANISME DE REMONTÉE EN QUESTION »** — ni sonde muette
(elle discrimine), ni k coupable (la péremption est minimale). À k=1 avec
EPS=1e-5 la remontée est quasi sans perte : un résidu au-dessus de 0.0603
pointe ailleurs. **Deux attributions NOMMÉES, NON ARMÉES** : (a) le retard d'une
frame suffit à lui seul — falsificateur : comparer contre la vérité DÉCALÉE
d'une frame ; (b) la référence incrémentale DÉRIVE — falsificateur : comparer
contre une remontée PLEINE non incrémentale. L'armement de l'un ou l'autre est
une décision de Romain à la lecture, jamais un enchaînement.

**Table complète et close (aucune lecture ne peut désormais tomber hors table)** :
(i) discrimine + un EPS passe ⇒ INSTRUMENT VALIDE ⇒ si aucun EPS ne passe à
k=4 : branche (D), la décision passe à k, réveil σ_ω explicite ;
(ii) ne discrimine pas ⇒ SONDE MUETTE sur EPS, AUTRE D'INSTRUMENT, ne rien
régler ; (iii) discrimine sans qu'aucun EPS ne passe ⇒ MÉCANISME EN QUESTION,
deux attributions nommées non armées. Ajout au JSON : le LABEL de lecture seul
— aucun build supplémentaire.

### §A19-lecture-sonde-EPS (2026-07-19) — branche (ii) MUETTE ; le bras k=1 FALSIFIE l'explication attendue ; diagnostic d'instrument pré-enregistré

**Lecture mécanique (run_f1_sonde_eps.py, natif)** : vérification d'instrument =
**branche (ii) SONDE MUETTE sur EPS**. Lecture pré-écrite APPLIQUÉE : **rien
n'est réglé, EPS reste à 1e-4** ; aucun EPS retenu, AUTRE remonté. Δχ décimé max
∈ [0.0820, 0.0825] pour un trafic ×173 — **innocuité NON ÉTABLIE** à toute
valeur (jamais « nocivité » : le joueur ne voit pas la référence).

**FAIT NEUF, plus important que la lecture : le bras k=1 FALSIFIE l'hypothèse
« péremption L1 ».** Δχ max par EPS —
k=1 : [0.0825, 0.0820, 0.0820, 0.0822, 0.0823, 0.0825] ;
k=4 : [0.0822, 0.0820, 0.0820, 0.0822, 0.0824, 0.0825] — **identiques à 3
décimales**. Si la péremption pilotait l'observable, k=4 (jusqu'à 4 frames de
retard) serait nettement pire que k=1 ; il ne l'est pas. **La note du driver
(« l'observable est gouverné par la péremption L1 ») est CONTREDITE par son
propre bras de vérification** — elle ne doit pas être portée comme vraie. Ni
EPS ni k ne bougent l'aiguille ; le Δχ non décimé est tout aussi plat (≈0.069).
⇒ **PLANCHER STRUCTUREL** (signature §A14 « muette par construction »). Suspect
principal nommé : le DOMAINE DE COMPARAISON (cellules du niveau 0 que nulle
fenêtre active ne remonte jamais ⇒ écart constant, indépendant de tout).

**Pourquoi ce plancher gate M-b (et non une curiosité)** : M-b partage les
primitives de readout (albedo + delta_chi/max_carrier) et son seuil de mort est
**0.0733 par-seed** — un plancher d'instrument à 0.069 siège à 6 % SOUS le seuil
qui tue Option A. Acheter 3–5 séances de portage fidèle avec ce plancher
inexpliqué, c'est risquer de prononcer MORT-b sur un ARTEFACT.

**Second écart à attribuer** : à EPS=1e-4 la sonde donne **médiane 17.249 ms**
là où M-a-quater donnait 16.589 — **+0.66 ms, quatre fois la marge de V4**, dans
le mauvais sens ; les transferts concordent (2.635 vs 2.727, dérive), l'écart
est ailleurs. Deux causes possibles : appareil de sonde entrant dans le chrono
« nu », ou terme de PRODUCTION non compté par M-a-quater. **Le contrôle T1 de
M-b est VERDICTAL (Q3)** — l'écart doit être attribué avant.

**Décision Romain (2026-07-19) : DIAGNOSTIC D'INSTRUMENT AVANT le chiffrage
M-b — pré-enregistré ici :**
- **D-1 (plancher)** : (a) TEST À BLANC — vivant := vérité, attendu **Δχ = 0
  exact** ; tout résidu = plancher de READOUT (transférable à M-b) ; (b)
  COMPARAISON RESTREINTE aux cellules effectivement couvertes par une fenêtre
  active du cycle — si le plancher s'effondre, c'est un plancher de COUVERTURE
  (propre à cette sonde, non transférable). Les deux lectures pré-écrites ;
  aucune autre conclusion tirée.
- **D-2 (chrono)** : attribuer les +0.66 ms — chrono « nu » de la sonde
  re-mesuré à configuration M-a-quater identique, appareil de reconstruction
  explicitement hors boucle. Lectures pré-écrites : écart absorbé ⇒ APPAREIL
  (sans conséquence) ; écart persistant ⇒ **TERME DE PRODUCTION NON COMPTÉ**,
  la marge de V4 est entamée et le contrôle T1 s'appliquera à ce total.
- Portée : diagnostic d'INSTRUMENT, pas une nouvelle tentative de régler EPS
  (la branche (ii) est appliquée : rien n'est réglé). k reste figé ; aucune
  décision de cadencement. Coût : minutes, harnais existant.
- **Le chiffrage de M-b (N2) est SUSPENDU à la lecture de D-1/D-2.**

**Levier identifié mais VERROUILLÉ (à ne pas surclamer)** : le balayage montre
que passer de 1e-4 à 1e-2 rendrait ~2.5 ms de transferts et ~1.8 ms de médiane —
exactement la tête qui manque à V4. **Inutilisable** : la non-discrimination
signifie que l'observable ne VOIT pas EPS, pas qu'il n'y a rien à voir.
Conclure « puisque rien n'est établi nulle part, prenons le moins cher » serait
du raisonnement motivé — explicitement refusé ici.

### §A19-complément-3 — Branche pré-écrite du diagnostic (gravé 2026-07-19, AVANT le run D-1/D-2)

Build D-1/D-2 livré (fluide-reduit b5be0dd, `src/` intact). **Retrait consigné** :
Claude Code a retiré l'hypothèse falsifiée (« l'observable est gouverné par la
péremption L1 ») des trois endroits où elle vivait — docstring, note de
discrimination, console — et l'a verrouillée par un test d'anti-régression ;
elle ne subsiste qu'en citation de ce qui a été falsifié. Seuils fixés AVANT run
et endossés : FACTEUR_EFFONDREMENT = 0.5, TOLERANCE_ABSORPTION_MS = 0.15.
`exiger_rien_regle` interdit au diagnostic de devenir un réglage déguisé.

**REFRAMING NOMMÉ (session critique, avant les chiffres)** : si D-1(b) confirme
le plancher de COUVERTURE, alors D-1(b) n'est pas qu'un diagnostic — il établit
que **l'observable de la sonde EPS était MAL DÉFINI** : sur les cellules que
nulle remontée ne touche, on comparait le vivant à une « vérité » qui ne
représente pas ce à quoi le grossier sert. L'observable RESTREINT serait alors
le CORRECT, et le levier EPS (~2.5 ms de transferts) n'était pas verrouillé,
seulement mal mesuré.

**DEUX GARDE-FOUS GRAVÉS** : (1) **JAMAIS de re-lecture du balayage existant**
sur l'observable restreint — choisir l'observable après avoir vu lequel donne
la réponse voulue serait fabriquer le verdict ; la lecture §A19 (branche ii,
EPS reste 1e-4) TIENT comme lecture de CET observable-là : superseded, jamais
invalidée. (2) Toute réouverture passe par un **NOUVEAU PRÉ-ENREGISTREMENT**
avec la règle Q2 RÉÉNONCÉE pour l'observable corrigé, puis une RE-MESURE.

**BRANCHE PRÉ-ÉCRITE (décision Romain, avant tout chiffre)** : plancher de
COUVERTURE confirmé ⇒ **EPS EST ROUVERT — nouveau prereg + re-mesure, AVANT le
chiffrage de M-b.** Motifs : le levier vaut ~2.5 ms quand V4 ne tient qu'à
0.111 ms avec p99 non résolu ; la sonde corrigée est bon marché (harnais
existant) ; et si D-2 révèle un terme de production non compté, le levier
devient nécessaire, pas optionnel. Plancher de READOUT (test à blanc non nul)
⇒ lecture inverse : le plancher est transférable à M-b, et c'est M-b dont
l'instrument devient la question (son seuil de mort siège 6 % au-dessus).

### §A19-lecture-diagnostic (2026-07-19) — readout PROPRE, D-2 relu (label faux), attribution (b) armée

**D-1(a) — ACQUIS SOLIDE : Δχ à blanc EXACTEMENT NUL.** Aucun plancher de
readout : les primitives (albedo + delta_chi/max_carrier) ne fabriquent rien.
**Le gate est DÉGAGÉ : M-b héritera d'un instrument sain** — c'était la raison
d'être du diagnostic, elle est satisfaite. Le plancher observé vient d'ailleurs.

**D-2 — LE LABEL DU DRIVER EST LOGIQUEMENT FAUX ; LA FAUTE DE SEUIL EST DE LA
SESSION CRITIQUE.** Médiane nue **16.427** — soit EN DESSOUS de M-a-quater
(16.589) : l'appareil de sonde expliquait **0.822 ms** des 17.249. Résidu
**−0.162 ms**, NÉGATIF : un terme de production non compté rendrait la mesure
PLUS HAUTE, jamais plus basse — le label « TERME DE PRODUCTION NON COMPTÉ » ne
peut pas s'appliquer à un résidu négatif (la règle a été écrite sur |écart|,
sans considération de signe). De plus −0.162 est très exactement la dérive
machine du jour (−1.07 % mesuré ⇒ −0.178 attendus sur 16.589). **Erreur de
pré-enregistrement OWNED par la session critique** : TOLERANCE_ABSORPTION_MS =
0.15 a été endossée alors que la dérive connue valait 0.155–0.18 — le seuil
était SOUS le bruit connu, la règle ne pouvait qu'échouer. **Lecture retenue :
APPAREIL. Aucun terme de production non compté. La marge de V4 tient à
0.111 ms** et le contrôle T1 de M-b reste verdictal sur le total inchangé.

**D-1(b) — la branche TIRE, son interprétation est CONTREDITE.** Complet
**0.0820** ; cycle **0.0109** (la vue spécifiée au gravé) ; cumulé **0.0892**.
La vue cycle s'effondre ⇒ branche « plancher de COUVERTURE » appliquée ⇒
**EPS EST ROUVERT** (complément-3, pré-écrit). MAIS le cumulé DÉPASSE le complet
— les cellules couvertes autrefois et non rafraîchies sont PIRES que la moyenne.
**Conséquence gravée : le nouvel observable ne sera PAS « cellules fraîches du
cycle »** — ce serait un instrument COMPLAISANT par construction (il ne
mesurerait que ce qui vient d'être corrigé, quand le consommateur voit tout).
**RETRAIT consigné (session critique)** : la reformulation « l'observable de la
sonde EPS était mal défini » (complément-3) n'est PAS soutenue par ces chiffres
— elle est retirée ; seule la réouverture d'EPS subsiste, sans son motif.

**Signature dessinée par les trois nombres** (frais bon, ancien mauvais,
indépendant d'EPS ET de k) : **dérive de la référence incrémentale** —
l'attribution (b) de la branche (iii), déjà nommée avec son falsificateur.

**Décision Romain (2026-07-19) : ARMER l'attribution (b) — comparaison contre
une REMONTÉE PLEINE NON INCRÉMENTALE** (harnais existant, minutes). Lectures
pré-écrites : erreur s'effondre ⇒ **la référence incrémentale DÉRIVE** (défaut
de mécanisme, load-bearing pour tout le lointain de la fovéa-z) ; erreur
persiste ⇒ **erreur INHÉRENTE** au grossier, et c'est l'exigence de fidélité
qui est à redéfinir. Dans les deux cas la lecture DÉTERMINE l'observable du
nouveau pré-enregistrement EPS — qui reste dû, et qui ne s'écrira pas avant.
EPS reste figé à 1e-4, k figé à 4 ; rien n'est réglé. Chiffrage M-b : toujours
suspendu, mais son GATE D'INSTRUMENT est désormais dégagé.

## §A20 — LA PROJECTION : image et son comme readouts de z (paper-grade, gravé 2026-07-19)

> Section d'architecture endossée par Romain (2026-07-19). Elle ouvre la **moitié
> PROJECTION** de la thèse, jamais construite jusqu'ici (état des lieux du jour).
> Rien n'est mesuré ici : le coût des deux projections est ENTIÈREMENT OUVERT.

### §A20-1 — Le principe (ENDOSSÉ)

**Image et son sont des projections DÉTERMINISTES de `z`** — optique et acoustique
appliquées à l'état, jamais génération. C'est la ligne de démarcation avec l'approche
générative : chez PERSIST (ICML 2026) le shader neuronal *« can learn arbitrary
rendering functions »* et prédit *« information not provided by 3D latents (texture,
lighting, particle effects…) »*, d'où une **texture qui dérive alors que leur état 3D
reste stable** (leur pas 1296). Une projection optique ne peut pas dériver ainsi :
**`z` stable ⇒ image stable, par construction.** Corollaire acquis : l'image devient
**FALSIFIABLE CONTRE L'ÉTAT** (« ce readout est-il la projection correcte de `z` ? »),
question qu'un monde généré ne peut structurellement pas poser.

**INVARIANT DE PROJECTION (le cœur) : aucun readout ne porte d'état propre
LOAD-BEARING.** Pas de mémoire de readout qui soit porteuse d'histoire ⇒ aucun chemin
de dérive. C'est cet invariant, pas le réalisme, qui protège la thèse.

### §A20-2 — Les deux projections sont HORS de l'échelle de temps de F

Contrainte quantitative posée AVANT qu'elle ne devienne hypothèse tacite. « Leurs
évolutions calculées par F » ne tient pas au sens littéral, dans deux directions
opposées `[calculs d'enveloppe, non mesurés]` :

- **La lumière est trop RAPIDE.** 3·10⁸ m/s ⇒ traverser 1 m = 3,3 ns, contre un pas de
  F de ~16 ms : rapport ~5·10⁶. Le transport lumineux est **à l'équilibre à chaque
  instant de F** — un *solve* par état (elliptique), pas une évolution (hyperbolique).
  Famille technique adossée : radiosité / light propagation volumes en cascade sur
  grille voxel, qui s'accorde bien à une pyramide multirésolution.
- **Le son est trop RAPIDE pour le pas de F.** Signal audible ⇒ ≥ 40 kHz
  d'échantillonnage = **667× la fréquence de frame**. En ondes sur la grille fine
  (h0/512 ≈ 2 mm), la CFL acoustique impose ~170 kHz, soit **~2800 sous-pas par
  frame** : mort à l'arrivée. Coïncidence à NOMMER sans s'en servir de justification :
  le plafond de LOD par distance devient mécaniquement un **plafond de fréquence**
  (~340 Hz au niveau grossier à 1 m) — cela *ressemble* à l'absorption atmosphérique
  des aigus, mais pour la mauvaise raison.

### §A20-3 — L'architecture qui en découle (ENDOSSÉE)

- **`z` porte les champs de matière et d'état** que F fait évoluer — albédo, humidité,
  température, densité, géométrie : tout ce qui *détermine* les propriétés optiques et
  acoustiques. Rien d'autre n'entre en `z` au titre de la projection.
- **Readout optique** = solve d'équilibre, fonction déterministe de `z`.
- **Readout auditif** = synthèse à taux audio, excitée par les ÉVÉNEMENTS et les ÉTATS
  de `z`, propagée dans la géométrie de `z` (littérature mature : synthèse modale
  temps réel, SYMPHONY, DiffSound — intégration, pas invention).
- **État temporel côté readout** (queue de réverbération, adaptation d'exposition,
  historique d'AA temporel) : **ÉPHÉMÈRE PAR DÉFINITION** — recalculable différemment
  d'un replay à l'autre, n'entre JAMAIS au registre, jamais porteur d'histoire. La
  frontière §3 existante le couvre ; aucun champ dépendant de l'observateur n'entre
  dans `z`.

### §A20-4 — FORK OUVERT (question Romain, NON décidée) : le feuilletage temporel

**La question posée** : les projections de `z(t)` parviennent-elles au joueur à `t+1` ?

**Reformulation exacte, qui est plus dure** : ce n'est pas *une* frame de retard (tout
moteur en a ; 2 à 4 est la norme, 1 serait excellent) — c'est que **l'état est DÉJÀ
temporellement hétérogène, par construction gravée** : le compteur à retard d'une frame
(choix 6, borne L3, endossé) met le niveau 0 une frame derrière le fin, et L1 k=4 étale
la remontée ⇒ une fenêtre peut être **à t−4, soit 66,7 ms** derrière la fovéa. Un rendu
qui composite le proche à `t` et le lointain à `t−4` ne projette pas `z(t)` mais un
**état feuilleté**, avec couture possible à la frontière de LOD pour tout ce qui bouge.

**Ce n'est pas spéculatif** : la sonde EPS (§A19-lecture) a mesuré le symptôme —
cellules rafraîchies du cycle **0.0109** vs cumulées périmées **0.0892**. L'attribution
(b) en vol discrimine sa cause. Question et ligne de mesure CONVERGENT.

**DEUX RÉFÉRENTS PERCEPTUELS DISTINCTS, à ne plus confondre :**
1. **Péremption de contenu** — « l'image montre-t-elle un monde perceptiblement
   différent de l'actuel ? ». **Mesurable avec le pin EXISTANT** : Δχ entre readouts
   consécutifs (et entre readout du feuilleté vs readout de `z(t)` homogène) vs
   jnd_sev, sur l'IC entier. **Sonde de minutes — À PRÉ-ENREGISTRER quand elle sera
   armée**, pas maintenant.
2. **Latence d'entrée / couplage moteur-visuel** — « le joueur sent-il le décalage
   entre son geste et la réponse ? ». **Canal perceptuel TOUT AUTRE ; le pin spatial
   n'en dit RIEN.** Référent NEUF, campagne humaine, **GATÉ** au même titre que
   r_fovea. Aucune décision de pipeline (latence vs débit) ne se prend sans lui.

**Contrainte AUDIO nommée** : si les événements d'excitation sont quantifiés au pas de
frame, il en résulte **~16,7 ms de gigue sur les transitoires** — audible, et sous le
seuil de désynchronisation audio-visuelle. **La quantification de `t_sim` au registre
(§4, écrite pour les entrées joueur) est probablement trop grossière pour l'audio** :
à trancher quand la projection auditive sera spécifiée, jamais par défaut.

### §A20-5 — Portées (ce que cette section NE dit pas)

Aucun coût mesuré, **ni pour l'optique ni pour l'auditif** — faisabilité hors de
question (ingénierie connue, pas recherche), chiffrage entièrement ouvert. Le motif de
T2 réserve l'autre moitié de la frame au rendu : **V4 à 16.589 ms signifie « 30 fps
avec ~16,7 ms pour un rendu jamais chiffré »**. « Le moteur tient » reste hors de
portée tant que cette moitié est vide. Rien ici ne modifie un seuil, ne gate ni ne
dégate une mesure en cours.

### §A19-CORRECTION (2026-07-19) — l'attribution du plancher était FAUSSE ; la sonde EPS mesurait un ARTEFACT

> **Entrée de correction append-only. Rien n'est effacé : l'erreur reste lisible.**
> Elle SUPERSÈDE l'attribution de §A19-lecture-sonde-EPS et §A19-lecture-diagnostic.

**Origine** : en armant l'attribution (b) (fluide-reduit ef2b3b2), Claude Code a vérifié
l'identité que le schéma B4 pose — la `reference` côté device est le modèle de ce que le
CPU sait (émettre d = etat − reference, transmettre d, puis reference += d) — et **elle
casse**. Le device fait rouler `fenetres` ET `references` de dx à chaque frame (fovéa
mobile) et y écrit les colonnes prédites GPU-side ; **le reconstructeur de la sonde ne
fait ni l'un ni l'autre**. L'identité casse dès la frame 0, avec ZÉRO coefficient
transféré : écart 0.196, plateau ~1.03.

**Ce que la sonde EPS mesurait réellement** : un **DÉCALAGE SPATIAL**, pas une infidélité
du grossier. Cela explique exactement ce qui était resté inexpliqué — l'indépendance
simultanée à EPS **et** à k : on comparait des régions décalées, que ni le seuil ni la
cadence ne peuvent bouger.

**CE QUI EST FAUX ET EST RETIRÉ** (attribution seulement) :
- « PLANCHER STRUCTUREL, signature §A14 muette par construction, suspect principal : le
  DOMAINE DE COMPARAISON (cellules que nulle fenêtre ne remonte) » — **FAUX**. La cause
  est le roll + la prédiction manquants dans le reconstructeur.
- L'interprétation de D-1(b) (complet 0.0820 / cycle 0.0109 / cumulé 0.0892) tombe avec :
  elle a été calculée sur l'observable défectueux. La branche « plancher de COUVERTURE »
  a tiré sur un observable cassé.

**CE QUI TIENT** : (i) la lecture mécanique « branche (ii) SONDE MUETTE » reste exacte
COMME LECTURE DE CET OBSERVABLE-LÀ — il n'a effectivement pas répondu ; (ii) **rien n'a
été réglé**, EPS est resté à 1e-4 — la lecture pré-écrite a protégé le projet d'un
réglage fondé sur un artefact ; (iii) D-1(a) test à blanc EXACTEMENT NUL : le readout est
propre, le gate d'instrument de M-b reste dégagé (il ne dépendait pas du reconstructeur).

**CE QUE ÇA REND À L'ÉTAT OUVERT** : la question EPS n'est **PAS répondue** — sonde à
réparer et à re-mesurer, avec un NOUVEAU pré-enregistrement. La question de la fidélité
du grossier n'est pas répondue non plus.

**LE FALSIFICATEUR AURAIT PRONONCÉ UN FAUX VERDICT** : armé tel quel, l'attribution (b)
aurait lu « l'erreur persiste » ⇒ **ERREUR INHÉRENTE AU GROSSIER, l'exigence de fidélité
est à redéfinir** — pour un défaut de reconstruction. Sur l'observable reconstruit les
deux bras sont indistinguables (0.082814 courant vs 0.082870 témoin) ; sur le livre de
comptes le témoin est ~8× meilleur. **La vérification d'identité avant run a évité un
verdict architectural erroné.** (Chiffres de config réduite, NON verdictaux.)

**RÉSULTAT POSITIF MAJEUR, à ne pas laisser passer** : le **livre de comptes contre
vérité vaut 0.00011** — soit **~550× SOUS ic_bas (0.0603)**, seuillage à 1e-4 inclus.
L'information TRANSMISE par la remontée est amplement suffisante ; l'inquiétude « le
grossier est infidèle » s'évapore. Ce qui reste ouvert est **la capacité du CPU à s'en
servir**, ce qui est une question de PRODUCTION, pas de harnais.

**QUESTION D'ARCHITECTURE OUVERTE (conséquence directe)** : si le device roule sa
`reference` et y écrit ses prédictions, alors **en production le CPU doit faire le même
roll et la même prédiction**, sinon les deux modèles divergent. Le « miroir CPU » n'est
donc pas un artifice de mesure : c'est **le premier morceau du côté CPU de production**,
et il tombe sur le **pied non mesuré n°3** (niveau 0 vivant à 500k cellules, ×7.6 jamais
chiffré). Les deux réparations candidates ne sont PAS équivalentes : (1) lire la
connaissance du CPU sur `reference` — gratuit, mais suppose la prédiction CPU exacte,
donc **borne inférieure de l'erreur** ; (2) miroir CPU complet — fidèle, mais c'est un
vrai build. **L'écart entre les deux EST l'erreur de prédiction propagée**, candidate au
terme dominant du lointain de la fovéa-z. NON TRANCHÉ.

**MEA CULPA D'INSTRUMENT (session critique)** : l'empreinte `e8fcaad4…7f04`, exigée et
citée sur une dizaine de tours comme preuve que le kernel est intouché, ne correspond
**ni au fichier** (`9f3de30c…`) **ni au `_SOURCE`** (`c30ab2bd…`), et n'apparaît nulle
part dans le kernel ni son test (vérifié 2026-07-19). J'ai accepté un jeton de preuve
sans jamais vérifier ce qu'il hachait — **du rituel, pas de la vérification**. Ce qui
protégeait réellement : le `git diff`, vérifié plusieurs fois. **Règle reconduite : une
preuve doit être re-calculable par celui qui l'exige, sinon elle est décorative.**

**DÉCISION ROMAIN (2026-07-19) : VÉRIFIER LES DEUX BUGS CANDIDATS D'ABORD** — questions
de CORRECTION, indépendantes du fork de réparation, bon marché :
- **(α) troncature du retard d'une frame** : des coefficients seraient perdus
  DÉFINITIVEMENT (`reference += d` ayant déjà tourné côté device) — **cela
  contredirait le nota endossé « le schéma incrémental ne perd rien »** ;
- **(β) queue de tampon** : si une frame émet moins que la précédente, la queue non
  réécrite serait retransférée et réappliquée.
Les deux sont LUS DANS LE CODE, non vérifiés numériquement. Si (α) est réel, le schéma
change et le choix de réparation change avec lui. Le fork de réparation (1) vs (2) se
tranchera APRÈS, informé.

### §A21 — B4 AMENDÉ : les deux bugs du retard d'une frame sont RÉELS ; invariant de transfert gravé (2026-07-19)

> Suite de §A19-CORRECTION. Build de vérification : fluide-reduit fdc1dbf (cas
> construits, attendus écrits d'avance, contre-épreuves incluses). Aucun run de mesure.

**CAUSE COMMUNE, en une phrase** : dans `_appliquer_f`, **la TAILLE du transfert vient
de la frame n−1, les DONNÉES de la frame n**. (α) et (β) sont les deux côtés du même
décalage.

**(α) TRONCATURE — RÉEL, et la perte est DÉFINITIVE.** Cas construit (5 émissions, puis
40, puis trois frames à 40) : la frame qui émet 40 n'en transfère que 5, et les trois
frames suivantes — qui ne tronquent rien et ont toute latitude — **ne rattrapent rien** :
écart de grand livre de **35 couples exactement, strictement inchangé (atol 1e-12)**.
Contre-épreuve : à tailles stables l'écart est **nul** — c'est bien la variabilité des
tailles, pas le retard en soi.

**(β) QUEUE DE TAMPON — RÉEL.** Cas construit (40, 40, puis 5) : `cloturer_frame` remet
le compteur à zéro sans réécrire les buffers ⇒ **identité bit à bit** des positions
[5, 40) avec la frame précédente (valeurs ET indices), et grand livre CPU **en excès de
35 couples**.

**CE N'EST PAS UN CAS LIMITE.** La condition de déclenchement est exactement
`tailles_stables`, que le diagnostic mesure — et elle est **FAUSSE PARTOUT** dans les
runs déjà enregistrés (lecture de `outputs/f1/`, aucun run neuf) : ma_quater
447 965 → 844 768 et 222 755 → 1 064 750 ; borne_l3 deux groupes à 2.3× sur 330 frames.
**Des centaines de milliers de coefficients perdus ou dupliqués à chaque frame.**

**LE NOTA ENDOSSÉ EST RETIRÉ.** « Le schéma B4 étant incrémental, les résidus repassent
le seuil : rien n'est perdu » supposait que le détail non transféré RESTE un résidu côté
GPU. **Il ne le reste pas** : l'épilogue L3 fait `reference[p] += d` pour **tout couple
ÉMIS, transféré ou non**. Le device inscrit dans sa référence une connaissance que le CPU
n'a jamais reçue, et l'écart **ne repassera plus jamais le seuil**.
**Conséquence architecturale, pas seulement de harnais : la garantie de convergence de
B4 est VIDE sous ce transfert — l'erreur du lointain n'est bornée NI par EPS NI par la
cadence, puisqu'elle s'accumule par un chemin que ni l'un ni l'autre ne contrôle.**

**CORRECTION D'UNE SURCLAME DE LA SESSION CRITIQUE** : §A19-CORRECTION énonçait « la
transmission est amplement suffisante, ~550× sous ic_bas ». **Faux comme formulé** : le
livre de comptes mesure la COMPTABILITÉ DU DEVICE, dont on sait désormais qu'elle diverge
en permanence de la réalité CPU. Énoncé correct : **le budget d'information du DESIGN est
amplement suffisant (0.00011) ; c'est l'IMPLÉMENTATION DU TRANSFERT qui le perd.**

**INVARIANT DE TRANSFERT (gravé, paper-grade — au-dessus du correctif) :**
> **La référence du device n'avance QUE sur ce qui a été effectivement TRANSFÉRÉ,
> jamais sur ce qui a été ÉMIS.** Toute implémentation qui viole cet invariant rend
> non bornée l'erreur du chemin grossier, quels que soient le seuil et la cadence.

**DÉCISION ROMAIN (2026-07-19) : CONSTRUIRE LE CORRECTIF PING-PONG.** Frame n écrit dans
le jeu `n mod 2` ; on transfère le jeu de n−1 avec le compteur de n−1 — taille et données
du même tour, **aucune synchronisation ajoutée (le choix 6 est préservé)**, et les deux
défauts tombent ensemble. Coût : **+120 Mio de VRAM préallouée** (buffers actuels 64 +
56 Mio) et **une frame de péremption sur les coefficients** (aujourd'hui les données sont
fraîches, seule leur taille est fausse). **Motif de la décision, consigné : l'arbitrage
est ASYMÉTRIQUE** — perte permanente NON BORNÉE contre péremption BORNÉE d'une frame, sur
un grossier qui en porte déjà jusqu'à quatre (L1 k=4) ; et +120 Mio sur une résidence
mesurée à 0.311 Go (gate 1.35) est immatériel. **Réserve nommée** : l'attribution « (a) le
retard d'une frame suffit à lui seul » reste NON ARMÉE — l'arbitrage est raisonné, pas
mesuré. Voies écartées et pourquoi : lecture synchrone du compteur (rétablit une
synchronisation, viole le choix 6) ; estampillage de chaque coefficient (+33 % de D2H et
modifie le kernel L3).

**STATUT DES MESURES DÉJÀ GRAVÉES (décision Romain) : TEMPS VALIDES, CONTENU INVALIDE.**
Le transfert déplace `taille(n−1)` octets à chaque frame ⇒ sur 300 frames le total est le
même à un terme près, les MÉDIANES DE TEMPS tiennent : **M-a-quater 16.589, M-c 2.784,
borne L3 5.122 sont conservées** avec cette réserve. Tout ce qui portait sur le CONTENU
transféré tombe. Aucune re-mesure de temps n'est ordonnée ; le contenu sera re-mesuré par
la sonde réparée.

**DISJONCTION MAINTENUE** : ces deux défauts et le fork de réparation de la
RECONSTRUCTION (lire `reference` vs miroir CPU complet) sont **disjoints** — corriger
l'un ne répare pas l'autre. Le fork reste NON TRANCHÉ.

**INSTRUMENT — empreinte rétablie et VÉRIFIÉE** : la seule signature que la suite
verrouille réellement est
`sha256(src.f1_gpu.substrat_fusionne._SOURCE) = e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4`
(chaîne CUDA extraite, **8223 caractères**, pas le fichier). **Recalculée
indépendamment par la session critique le 2026-07-19 : CONCORDE.** La citation
`e8fcaad4…7f04` est abandonnée définitivement. **Signalé, non corrigé (cap) : le kernel
L3 n'a AUCUN verrou d'empreinte** — seulement l'équivalence structurelle et l'état
bit-identique. C'est pourtant lui qui porte `reference += d`.

### §A21-complément — Correctif ping-pong LIVRÉ ; invariant AMENDÉ à sa forme implémentable ; fork de reconstruction TRANCHÉ (2026-07-19)

Build : fluide-reduit f5b3999. **Vérifié indépendamment par la session critique** :
`_SOURCE` fusionné 8223 car. **e18015f5…30b4** CONCORDE ; `_SOURCE_L3` 9635 car.
**9533a130…781a** CONCORDE ; diff de `substrat_fusionne.py` VIDE ; sur `substrat_l3.py`
les hunks tombent en 14/57/60 (docstring) puis ≥181 (`CompacteurL3`) — **aucun entre
76 et 175 : le bloc CUDA est intact par POSITION**, pas par affirmation. Le kernel L3 a
désormais son propre verrou d'empreinte (il porte `reference += d`).

**L'INVARIANT §A21 EST AMENDÉ — ma formulation n'était pas implémentable.** J'avais
gravé « la référence du device n'avance QUE sur ce qui a été TRANSFÉRÉ » : or le kernel
fait `+= d` à l'ÉMISSION et n'a aucun moyen de savoir ce qui sera transféré. La forme
retenue, due à Claude Code, atteint la même fin et elle est TESTABLE :

> **INVARIANT DE TRANSFERT (forme exacte) : bijection ÉMIS ↔ REÇU — chaque couple
> transféré une fois et une seule, avec un décalage d'EXACTEMENT une frame :**
> **`livre_cpu(après n) ≡ livre_device(après n−1)`.** Rien de moins (α réglé), rien de
> plus (β réglé). Seule la dernière frame d'une série reste en attente — **borne
> exacte, avec son propre test**. L'erreur du chemin grossier est alors bornée par une
> frame de péremption, jamais accumulée.

Le correctif ne supprime pas le retard : **il le rend honnête.** L'invariant est tenu
par le COMPACTEUR (le kernel ne peut pas le tenir), `_appliquer_f` est intouché.

**DEUX CORRECTIONS QUE LE BUILD IMPOSE À MES GRAVURES :**
1. **Le compteur n'était PAS faux** : `hote[0]` portait déjà le compte de n−1 (choix 6,
   conforme) — c'est la DONNÉE qui venait de la mauvaise frame. Mon « la taille vient de
   n−1, les données de n » nommait le bon symptôme avec le mauvais coupable. **Le
   mécanisme du choix 6 est CONSERVÉ, pas contourné.**
2. **L'argument « temps valides » se RENFORCE** : avant on transférait `taille(n−1)`
   octets depuis `buffer(n)` ; après, depuis `buffer(n−1)` — **volume par frame
   IDENTIQUE, exactement**. M-a-quater 16.589 / M-c 2.784 / borne L3 5.122 ne sont plus
   « valides à un terme près sur 300 frames » : elles sont **inchangées par
   construction**.

**Preuves apportées avec le build** : choix 6 préservé — **prouvé** par piège
(points d'entrée de synchronisation remplacés par des levées, six frames par le chemin
de production) **avec contre-épreuve que le piège est ARMÉ** (une synchronisation
délibérée doit lever d'abord — *« un piège qui n'attrape rien ne prouverait rien »*).
Les deux tests (α) et (β) passent d'« exhibe le défaut » à « prouve son absence »,
mêmes cas construits, mêmes attendus, conditions de déclenchement toujours présentes.
Test d'INVARIANT (pas de régression) sur les amplitudes RÉELLES lues dans `outputs/f1/`
(2.3× borne_l3, 4.8× ma_quater, croissances et décroissances alternées), avec
vérification que `tailles_stables` vaut bien **False** — l'invariant tient MALGRÉ
l'instabilité. Résidence : **+120 Mio = 0.126 Go** chiffrés AVANT run sans allouer,
reportés à part au JSON (résidence V2 0.311, gate 1.35 — marge intacte). Le nota « rien
n'est perdu » est **retiré, pas nuancé** : disparu de la note du JSON (le seul texte
qu'un lecteur de résultats voit), et un test interdit qu'il soit re-porté comme vrai.

**DÉCISION ROMAIN — FORK DE RECONSTRUCTION TRANCHÉ : OPTION 1 (lire `reference`).**
Motifs consignés : (i) le transfert étant corrigé, `reference(n−1)` ≡ connaissance
CPU(n) **exactement** — l'hypothèse est bien plus mince qu'avant le correctif ;
(ii) argument de sûreté : si l'erreur de prédiction domine, EPS n'est de toute façon pas
la contrainte liante ; si elle ne domine pas, la réponse de l'option 1 est la bonne ;
(iii) c'est une RÉPARATION DE SONDE, gratuite, qui débloque la question EPS (levier
~2.5 ms alors que V4 ne tient qu'à 0.111 ms). **L'option 2 (miroir CPU complet) reste
NOMMÉE comme travail de PRODUCTION** — premier jalon du côté CPU, sur le pied non mesuré
n°3 (niveau 0 vivant à 500k, ×7.6) ; l'écart entre les deux options EST l'erreur de
prédiction propagée, candidate au terme dominant du lointain. Non armée.

**CONSIGNE PRÉ-ENREGISTRÉE POUR LA SONDE RÉPARÉE (session critique)** : au frame n, le
CPU connaît l'état de **n−1**. La sonde doit donc **DÉCLARER AVANT RUN contre quelle
vérité elle compare** — `vérité(n)` mesure ce que le joueur voit (canal + péremption
acceptée), `vérité(n−1)` isole la fidélité du CANAL seul. Les deux sont légitimes, elles
diffèrent d'exactement la péremption qu'on vient d'acheter, **et les confondre
refabriquerait un artefact de décalage temporel** — la faute même qui a coûté la journée.

### §A22 — Sonde EPS v2 : le verrou a mordu sur le correctif ; prereg ENDOSSÉ avec trois amendements (2026-07-19)

Build : fluide-reduit 7a40303. **Vérifié indépendamment** : empreintes 8223/e18015f5
et 9635/9533a130 concordent ; hunks de `substrat_l3.py` tous ≥256 dans `CompacteurL3`
(bloc CUDA intact) ; verrou d'endossement `ENDOSSEMENT_PREREG_V2 = False` présent —
**« aucun run avant endossement » est désormais MÉCANIQUE, pas une promesse.**

**LE VERROU A MORDU SUR LE CORRECTIF QUE J'AVAIS DÉCLARÉ LIVRÉ.** En écrivant le test
d'identité, Claude Code a trouvé que le ping-pong (§A21) déplaçait les **données** d'un
tour à l'autre **sans déplacer leur OFFSET D'INDICES** : `_appliquer_f` calculait
l'offset sur la tranche de la frame courante alors que les couples transférés étaient
ceux de n−1 ⇒ **réindexation silencieuse dans le mauvais slot**, divergence grand livre
vs `reference` de **~1.4e-4 dès la deuxième frame**. L'offset relève du même invariant
que la taille et les données ; il est désormais confié au compacteur (`noter_offset` /
`offset_precedent`) et voyage avec elles.

**FAUTE DE LA SESSION CRITIQUE, consignée** : §A21-complément gravait « correctif livré,
vérifié ». J'avais vérifié les EMPREINTES et la POSITION DES HUNKS — cheap et
vérifiable — puis pris l'invariant sur la foi de tests qui éprouvaient la QUANTITÉ
transférée, jamais les INDICES. **Je n'ai pas demandé ce que le jeu de tests ne couvrait
pas.** Même faute que l'empreinte décorative : vérifier ce qu'on me présente au lieu de
chercher ce qui manque. Ordre de grandeur qui donne la mesure du risque : la divergence
1.4e-4 est **exactement celle du signal cherché** (livre de comptes vs vérité 1.1e-4) —
elle l'aurait noyé. **L'invariant, lui, était correctement énoncé** (bijection de
COUPLES (indice, valeur)) : c'est l'implémentation qui le violait, et le test l'a
attrapée. C'est à cela que servent les invariants.

**CE QUE LE PREREG V2 APPORTE (endossé)** : superseded assumé (aucune lecture v1
reconduite — ni le plancher 0.082, ni la branche (ii), ni la comparaison restreinte) ;
connaissance CPU **lue sur `reference`** (option 1) ; **instant de lecture GRAVÉ** —
après le déplacement, avant l'émission, le seul où `reference` est dans les coordonnées
de la frame courante ; **deux vérités déclarées avant run** — `vérité(n)` PORTE LA RÈGLE
(le joueur ne perçoit pas un canal mais un écart au présent ; lire la règle sur la
vérité transportée s'accorderait gratuitement le retard qu'il subit), `vérité(n−1)`
DIAGNOSTIC, leur écart = `prix_peremption_une_frame` ; **portée déclarée qui VOYAGE DANS
LA LECTURE** (`portee_option_1`) — la sonde crédite le CPU d'une prédiction identique à
celle du device, donc elle mesure **le CANAL, jamais le lointain** : aucun résultat ne
pourra s'énoncer « le lointain est fidèle ».

**TROIS AMENDEMENTS EXIGÉS PAR ROMAIN AVANT RUN (session critique) :**
- **(A) Critère de discrimination et PLANCHER DE BRUIT écrits dans le prereg, et le
  plancher MESURÉ PAR RÉPLICAT, pas supposé.** La v1 portait un seuil relatif de 5 % en
  dur dans le code ; l'observable a changé d'échelle (~0.082 artefact → ~1e-4 attendu),
  et un seuil relatif ne se comporte pas pareil sur un signal 800× plus petit.
- **(B) BRANCHE (iii-bis) PÉREMPTION, écrite avant le run.** La règle portant sur
  `vérité(n)`, si aucun EPS ne passe sur `vérité(n)` **mais qu'ils passent sur
  `vérité(n−1)`**, la cause est **LE RETARD, PAS LE MÉCANISME** — la branche (iii) telle
  qu'écrite mal étiquetterait l'issue. Le discriminateur est déjà collecté
  (`prix_peremption_une_frame`), il n'était pas câblé à la table.
- **(C) LA DISCRIMINATION NE GATE QU'EN CAS D'ÉCHEC.** Si tout passe confortablement —
  plausible : canal ~1e-4 contre seuil 0.0603 — Δχ sera **plat PARCE QUE TOUT VA BIEN**,
  la discrimination échouera et (ii) MUETTE bloquerait un PASS parfaitement clair.
  Règle amendée : *au moins un EPS passe ⇒ l'instrument suffit pour CETTE conclusion, la
  discrimination passe en diagnostic ; aucun ne passe ⇒ la discrimination gate, et c'est
  là que (ii) et (iii)/(iii-bis) se séparent.* Plat-et-bas est informatif ;
  plat-et-haut ne l'est pas.

**Sur ces trois amendements intégrés, `ENDOSSEMENT_PREREG_V2` peut être levé** et le
balayage tourner. EPS reste 1e-4 et k reste 4 d'ici là.

**Conséquences consignées** : `run_f1_attribution_b.py` reste GATÉ (bâti sur l'observable
v1 ; le ré-armer est une décision distincte) ; la lecture D-2 gagne en cohérence
(l'option 1 supprime la capture de coefficients dont la v1 avait besoin, ce qui va dans
le sens des +0.822 ms attribués à l'appareil — pas une re-mesure) ; `ReconstructeurNiveau0`
conservé et marqué SUPERSEDED (des lectures acquises en dépendent, on ne réécrit pas le
passé).

### §A22-complément — Prereg v2 FINAL : amendement du plancher, puis levée du verrou (2026-07-19)

Texte final remonté (fluide-reduit 2c1d64b) ; vérifié : amendements A/B/C intégrés,
ordre des branches documenté comme choix remonté, verrou toujours à `False`, empreintes
inchangées (8223/e18015f5, 9635/9533a130), `git diff -- src/` vide.

**CONCESSION DE LA SESSION CRITIQUE sur l'amendement (A)** : Claude Code CONSERVE le
seuil relatif de 5 %, et son argument est **meilleur que l'amendement qui l'a provoqué**
— une amplitude relative est SANS ÉCHELLE, elle ne devient pas fausse quand l'observable
passe de ~0.082 à ~1e-4 ; **en choisir une autre aujourd'hui reviendrait à la former en
CONNAISSANT l'échelle de la donnée**, c'est-à-dire un seuil ajusté à la vue des chiffres.
Le défaut n'était pas sa valeur mais son ISOLEMENT — il est corrigé par l'ajout du
plancher, pas par un changement de valeur. Deux cas limites nommés plutôt que tus :
plancher nul ⇒ chaîne déterministe (pas infiniment précise), le relatif porte seul et le
driver l'annonce ; plancher non évalué ⇒ discrimination INDÉTERMINÉE, jamais prononcée
sur la forme seule.

**DEUX APPORTS AU-DELÀ DE CE QUI ÉTAIT DEMANDÉ :**
1. **L'ATTRIBUTION (a) EST ARMÉE GRATUITEMENT.** Son falsificateur gravé (§A19-
   complément-2) était « comparer contre la vérité DÉCALÉE d'une frame » — c'est
   exactement `vérité(n−1)`. **La branche (iii-bis) EST sa lecture pré-écrite : un bras
   ÉCONOMISÉ, pas ajouté.** Conséquence directe : **la réserve de §A21 est levable** —
   l'arbitrage « péremption bornée d'une frame contre perte non bornée », que Romain
   avait dû trancher au RAISONNEMENT, sera **MESURÉ en sous-produit**. (b) reste non
   armée, son bras gaté.
2. **ORDRE DES BRANCHES — choix remonté et ENDOSSÉ** : (iii-bis) est évaluée AVANT la
   discrimination, parce qu'elle repose sur une MESURE DIRECTE et non sur la pente d'un
   balayage. Une sonde peut être muette sur EPS tout en mesurant parfaitement le prix du
   retard : les deux axes sont indépendants, et « la cause est le retard » est plus
   informatif que « sonde muette ». À k=4, (iii-bis) porte le plus (la péremption L1
   complète s'ajoutant au retard d'une frame) ; à k=1 elle est la plus lisible (la seule
   péremption restante EST celle du compteur). Test de couverture sur les quatre issues.

**DERNIER AMENDEMENT AVANT LEVÉE (décision Romain, sur point remonté par Claude Code
qui a refusé de le préempter)** : le plancher n'avait qu'UN réplicat — deux passes, un
seul écart, **aucune dispersion**. Décider APRÈS coup qu'il en faut plusieurs serait une
décision post-hoc, précisément ce que le pré-enregistrement interdit ailleurs. Donc,
gravé AVANT run :
> **3 répétitions du même EPS** (au lieu de 2) ; **plancher = max des écarts observés** ;
> et règle PRÉ-ÉCRITE : **si l'amplitude du balayage < 3 × plancher ⇒
> INDÉTERMINÉ-INSTRUMENT, AUCUNE branche prononcée.**
Coût : deux passes Δχ de plus (le réplicat coûtait une passe sur douze).

**SUR CET AMENDEMENT INTÉGRÉ, `ENDOSSEMENT_PREREG_V2` EST LEVÉ** — le balayage peut
tourner (natif, machine-instrument). Restent figés d'ici la lecture : EPS 1e-4 et k=4 ne
sont RIEN réglés par le run lui-même (le run PROPOSE un EPS via la règle Q2, il ne
l'applique pas) ; `run_f1_attribution_b.py` reste GATÉ ; le miroir CPU (option 2) reste
travail de production non construit ; toute décision sur k reste gatée σ_ω (R4).

## §A23 — LECTURE SONDE EPS v2 : branche (i) INSTRUMENT VALIDE ; EPS = 1e-2 APPLIQUÉ (2026-07-19)

Build : fluide-reduit c64c198 (amendement du plancher + levée du verrou). Run natif,
machine-instrument. **Première sonde de la série qui mesure ce qu'elle vise.**

**Branche (i) INSTRUMENT VALIDE, aux DEUX cadences.** L'observable VOIT EPS :
amplitude **92.89 %, monotone** — l'exact inverse de la v1.

| EPS | Δχ vérité(n) | Δχ vérité(n−1) | prix retard | octets/frame | médiane | p99 |
|---|---|---|---|---|---|---|
| 1e-5 | 0.00018 | 0.00016 | +0.00002 | 21 291 112 | 19.496 | 34.058 |
| 1e-4 | 0.00030 | 0.00028 | +0.00002 | 12 749 856 | 16.996 | 20.494 |
| 3e-4 | 0.00049 | 0.00048 | +0.00001 | 8 418 304 | 16.507 | 20.941 |
| 1e-3 | 0.00104 | 0.00103 | +0.00001 | 3 630 404 | 15.732 | 18.609 |
| 3e-3 | 0.00211 | 0.00210 | +0.00001 | 1 304 204 | 15.263 | 18.438 |
| **1e-2** | **0.00252** | 0.00251 | +0.00001 | 371 996 | 15.195 | 18.527 |

**Les six EPS passent, 1e-2 compris. Δχ max au pire = 0.00252, soit 24× SOUS ic_bas
(0.0603) : INNOCUITÉ ÉTABLIE.** Règle Q2 ⇒ propose **1e-2**.

**LEVIER MESURÉ (estimation de Claude Code confirmée à 0.01 ms près)** : de 1e-4 à
1e-2, trafic **÷34.3**, transferts **−2.367 ms**, médiane **−1.801 ms**.

**DÉCISION ROMAIN : EPS = 1e-2 APPLIQUÉ** — la règle était pré-enregistrée et elle a
tiré ; ne pas l'appliquer après avoir vu la donnée serait aussi grave que d'en changer.
**TROIS PORTÉES GRAVÉES AVEC L'APPLICATION** :
1. **INNOCENT, pas OPTIMAL.** 1e-2 est la plus GRANDE valeur testée et elle passe : le
   balayage **ne borne pas par le haut**. On ne sait pas où EPS cesse d'être innocent,
   seulement que c'est ≥ 1e-2. Étendre le balayage serait choisir des valeurs **la donnée
   en main** — cela exigerait son propre pré-enregistrement.
2. **Le CANAL, pas le lointain.** La sonde crédite le CPU d'une prédiction identique à
   celle du device (option 1). « Le canal est innocent à 1e-2 » ne dit PAS « le lointain
   est fidèle à 1e-2 ». L'application repose sur l'argument de sûreté endossé avec
   l'option 1 (si l'erreur de prédiction domine, EPS n'est pas la contrainte liante),
   **toujours non mesuré** — le miroir CPU reste non construit.
3. **CE substrat, CE régime.** Voir la réserve ci-dessous.

**RÉSERVE DE LA SESSION CRITIQUE — la plus belle lecture est la plus limitée.**
« La péremption ne coûte quasiment rien » (prix du retard ~1 % de l'observable, k=1
confondu avec k=4 à la 5ᵉ décimale) est vrai **DANS CE RÉGIME**, où *tout* est 24 à 300×
sous le seuil : sur ce substrat, la question du canal est simplement loin de la limite
perceptuelle. Un substrat plus rapide (explosion, front d'eau vif) déplacerait davantage
par frame. **L'attribution (a) est réfutée ICI, pas en général** — et la réserve de §A21
(arbitrage péremption bornée vs perte non bornée) est levée **avec cette portée**.

**CE QUE LE PLANCHER A FAIT : RIEN, et c'est consigné.** Plancher **exactement nul** sur
3 répétitions, aux deux cadences, sur les quatre grandeurs — la chaîne est DÉTERMINISTE.
Donc le critère d'échelle n'a pas travaillé, la règle d'indétermination ne pouvait pas
tirer (3 × 0 = 0), et le critère de FORME a porté seul. **L'amendement du plancher exigé
par la session critique s'est révélé INERTE** ; c'était le cas pré-déclaré par Claude
Code, rien n'est surpris. Nota gravé de sa plume : **un plancher nul dit REPRODUCTIBLE,
pas PRÉCIS — il ne borne aucune erreur systématique.**

**LE STUTTER N'EST PAS RÉSOLU PAR EPS** : p99 passe de 20.494 (1e-4) à 18.527 (1e-2) —
amélioré de ~2 ms, **toujours au-dessus de 16.7**.

**AUCUNE CONSÉQUENCE BUDGÉTAIRE N'EST TIRÉE ICI — décision Romain.** La passe chrono
donne 16.996 ms à 1e-4 contre **16.589 à M-a-quater sur la même config** : **+0.407 ms,
au-dessus de la dérive machine (~0.18 ms), NON EXPLIQUÉ**. Deux hypothèses vivantes :
(A) **appareil résiduel** de la sonde — D-2 avait mesuré 0.822 ms d'appareil en v1
(17.249 vs 16.427 nu) ; l'option 1 en supprime une part (17.249 − 16.996 = 0.253),
il resterait ~0.57 ms contre le nu ; (B) **régression réelle** — le ping-pong et le
correctif d'offset coûtent ~0.4 ms. **(A) est l'hypothèse qui ARRANGE la session
critique** (elle rendrait à V4 son 16.589) : elle est donc énoncée avec sa concurrente,
pas à sa place.
**PROCHAIN PAS ORDONNÉ : re-mesurer le CHRONO NU sur HEAD (méthode D-2, driver
existant, minutes) pour séparer appareil et régression ; PUIS re-run M-a-quater à
EPS = 1e-2 pour lire le budget de V4 sur des chiffres propres.** D'ici là, **aucun
verdict budgétaire** — ni « V4 est mort à 16.996 », ni « V4 respire à 15.195 ».

## §A24 — VERDICT M-a-quater : V4 **SANS MORT** (2026-07-19) — gate (i) satisfait

Builds : fluide-reduit 02f0239 (application EPS), 6c24622 (chrono nu), d73c875
(M-a-quater à 1e-2). Lectures versionnées désormais sous `claude/lectures/`
(décision Claude Code : la LECTURE de chaque run est versionnée — meta,
lecture_mecanique, table compacte, résidence — pas les tableaux par-frame).

**INSTRUCTION ERRONÉE DE LA SESSION CRITIQUE, refusée à raison.** Mon prompt disait
« `EPS_EN_VIGUEUR` passe à 1e-2 ». **Faux** : ce symbole est importé par D-1/D-2
(`EPS_FIGE`) et attribution_b (`EPS_COURANT`) sous garde `exiger_rien_regle`, et
`EPS_DETAIL` est la constante gravée de la borne L3 — bouger l'un ou l'autre aurait
**réécrit des mesures acquises**. Claude Code a refusé et introduit `EPS_PRODUCTION`,
distinct, au seul chemin de production, avec les trois portées §A23 et la citation
b6c8807 au point du réglage ; un test échoue si la valeur bouge ou si citation et
portées disparaissent.

**§A23-2a — LE +0.407 ms EST DE L'APPAREIL, tranché par une règle écrite avant.**
Chrono nu sur HEAD à 1e-4 : **16.297 ms**, soit **−0.292 sous le baseline 16.589**.
**Le SIGNE tranche** : un terme de production non compté ne peut que rendre la mesure
PLUS HAUTE ; le nu est plus BAS ⇒ appareil et dérive, **jamais régression**. C'est la
règle D-2 corrigée (celle qui teste le dépassement vers le haut, et non |écart|) qui
porte la conclusion. **Le ping-pong ne coûte pas de temps : il double la VRAM, pas le
calcul. V4 garde son 16.589.** Consigné : **l'hypothèse qui arrangeait la session
critique s'est trouvée juste SANS que la préférence n'ait décidé** — c'est la règle
pré-écrite qui a tranché. Réserve de Claude Code reconduite : −0.292 dépasse d'un cheveu
la dérive nominale (0.18), dans le sens favorable ; une seule passe, pas de
surinterprétation.

**§A23-2b — BUDGET V4 À EPS = 1e-2 :**

| grandeur | valeur | référence |
|---|---|---|
| **médiane frame complète** | **14.490 ms** | baseline 1e-4 : 16.589 (**−2.099**) |
| **MORT (médiane > 16.7)** | **NON** | marge **2.199 ms** (contre 0.111 auparavant) |
| p99 (en évidence) | 16.853 ms | +2.363 vs médiane |
| transferts | 0.110 ms | contre 2.449 à 1e-4 (trafic ÷ ~22) |
| résidence | 0.564 Go | gate 1.35 — marge intacte |

Le gain vient **entièrement des transferts** ; le calcul est inchangé.

**VERDICT PRONONCÉ (Romain, 2026-07-19) : V4 SANS MORT.** Le critère MORT-a est
pré-enregistré et porte sur la MÉDIANE : 14.490 < 16.7. **Le gate (i) de la spec est
satisfait — et pour la première fois avec une VRAIE marge** (2.199 ms contre 0.111 au
verdict précédent, qui était « au seuil » et sous la dérive machine).

**QUATRE RÉSERVES GRAVÉES AVEC LE VERDICT :**
1. **« Dans la bande [14.2, 15.9] » N'EST PAS UN TEST DE MODÈLE ICI.** Cette bande fut
   pré-enregistrée pour la configuration à EPS = 1e-4, **transferts ~2.4 ms compris** ;
   appliquer 1e-2 en a retiré 2.34. **Une bande qui ne bouge pas quand un terme de
   2.3 ms disparaît ne teste plus rien** — que 14.490 y tombe est une COÏNCIDENCE, pas
   une validation. Recalculée pour 1e-2 elle vaudrait ~14.0 (12.655 F + ~0 prédiction +
   ~1.28 remontée + 0.110 transferts), soit un écart mesuré de **+0.45**. **À refaire
   (arithmétique pure, aucun run) avant tout énoncé sur le modèle.**
2. **p99 = 16.853 > 16.7** : dépasse l'allocation de **+0.9 % sur 1 % des frames** —
   contre +17.7 % il y a quelques heures (19.652). Amélioration massive, **pas propre**.
   Non verdictal par T2 (la mort porte sur la médiane), reporté en évidence.
3. **Les trois portées de §A23 tiennent** : EPS 1e-2 est établi INNOCENT et non OPTIMAL
   (le balayage ne borne pas par le haut) ; la sonde mesure le CANAL et non le lointain
   (l'argument de sûreté de l'option 1 reste non mesuré, miroir CPU non construit) ; le
   résultat est scopé à CE substrat, CE régime.
4. **LA MOITIÉ RENDU DE LA FRAME RESTE NON MESURÉE** (§A20-5). Ce qui est établi est :
   *la physique tient, avec marge, sur un proxy, à cette enveloppe, sur cette machine,
   rendu non compté.* **« Le moteur tient » reste hors de portée.**

**ÉTAT DES GATES DE LA SPEC** : **(i) M-a-quater sans mort — SATISFAIT** ;
(ii) F0-cloud — **toujours dû** (~2 min, prochaine session cloud) ; (iii) M-b sans mort
— non achetée. **DÉCISION ROMAIN : prochain achat = CHIFFRAGE DE M-b** (gate iii,
3–5 séances, risque ÉLEVÉ) — son gate d'instrument est dégagé (readout propre, D-1(a)
Δχ à blanc exactement nul), D-1/D-2 sont lus, la suspension du chiffrage est levée.

## §A25 — Bande recomposée (statut abaissé) ; chiffrage M-b ; T1 tranché ; DEUX TERMES À PRÉ-ENREGISTRER AVANT ACHAT (2026-07-19)

Builds : fluide-reduit ca5ef3a (bande), e798d77 (chiffrage `claude/chiffrage-mb.md`).

**BANDE RECOMPOSÉE — STATUT ABAISSÉ PAR LA SESSION CRITIQUE.** Point plancher :
F 12.655 + remontée 5.122/4 = 1.280 + transferts 0.110 = **14.045** ; incertitude
± (dérive 0.18 + amortissement L1 0.10), prédiction en plancher unilatéral [0, ~0.5]
⇒ **[13.8, 14.8]**. Mesuré 14.490, écart **+0.445, dans la bande**.
**MAIS : cette bande est construite APRÈS la mesure.** Ses composants sont des ancres
indépendantes, mais la fourchette de prédiction [0, ~0.5] est CHOISIE et 14.490 en
exige 0.445. **Un intervalle bâti la donnée en main ne peut pas prononcer « ce n'est
pas un AUTRE »** — il dit seulement : *la mesure est COHÉRENTE avec les ancres*. C'est
plus faible, et c'est suffisant ici. Réserve de Claude Code reconduite : la prédiction
étant un SOLDE, elle absorbe par construction tout résidu non modélisé — le test dit
que 0.445 est un coût plausible, pas qu'il est prédiction et rien d'autre.
**CONSÉQUENCE CONSTRUCTIVE GRAVÉE : toute mesure de cette classe aura désormais sa
bande PRÉ-ENREGISTRÉE pour sa config exacte** — trivial maintenant que les ancres
existent.

**CHIFFRAGE M-b** : total ~2 600–3 680 LOC, **3.6–5.7 séances** > 3 ⇒ **scission
interne risqué-d'abord** (règle 3, comme A/B) : **tranche-1 = gate T1** (~2.1–3.4 :
F fidèle + Exner + pulse/épisode + T1) — *la frame fidèle tient-elle sous 16.7 ?*, le
moins cher qui peut tuer V4-b ; **tranche-2 = fidélité** (~1.5–2.3 : émissions
3 seeds/Δt=4/6, MORT-b par-seed contre 0.0733) — la fidélité ne se mesure que si le
coût passe. Fait structurel remonté : **M-b est le PREMIER build dont l'objet est un
MOTEUR** (le F fidèle) et non un motif de coût — le morceau qui glisse
mesure→prototype, **pire que L3** (L3 était greffé sur F ; ici le moteur EST l'objet,
sans cap possible).

**DÉCISION ROMAIN — T1 PORTE SUR LA MÉDIANE.** Le gravé §A19-Q3 disait « frame
complète > 16.7 » sans lever médiane/p99, là où MORT-a tranche explicitement pour la
médiane. **T1 = médiane frame complète (F fidèle) > 16.7 ⇒ V4 mort** — cohérent avec
MORT-a et avec T2 qui a déclaré le p99 NON-VERDICTAL. **Le p99 reste reporté en
évidence et la question du stutter reste NOMMÉE À PART** — ni fondue dans T1, ni
enterrée. Motif consigné : ajouter aujourd'hui un seuil p99 **en sachant qu'il vaut
16.853** reviendrait à le régler à la vue de la donnée.

**RÉSERVE MAJEURE DE LA SESSION CRITIQUE : L'ANCRE 12.655 N'EST PEUT-ÊTRE PAS LA BONNE
BASE POUR LE FIDÈLE.** Deux termes que l'ancre **n'a jamais portés** :
1. **HALOS DE FENÊTRE.** Le jetable pade en **réfléchissant par fenêtre** — acceptable
   pour un motif de coût, **faux pour la physique**. Le fidèle exige de VRAIS halos
   entre fenêtres, donc un échange non mesuré.
2. **MAPPING ÉPISODE↔FRAME.** L'ancre mesure **UNE** application de F. Si le CFL du
   fidèle impose PLUSIEURS pas par frame pour avancer le temps réel, **le budget se
   multiplie d'autant**. C'est le **pied non mesuré n°7 (cadencement)** qui arrive sur
   le chemin critique, **déguisé en détail de protocole**. Le rederive fait pulse +
   ~600 pas wetdry O2 CFL-adaptatif + ~300 pas Exner par épisode : le rapport
   pas/frame décide de tout ce que T1 signifie.
**Ces deux termes peuvent coûter plus que tout ce qui a été optimisé aujourd'hui, et
aucun n'est dans les 14.490.**

**DÉCISION ROMAIN : LES PRÉ-ENREGISTRER D'ABORD, PAPER-GRADE, AVANT TOUT ACHAT.**
Aucun run, aucune ligne : (a) dériver le nombre de pas de F par frame depuis le **CFL
réel du substrat**, et (b) chiffrer ce que coûtent de vrais halos entre fenêtres.
Motif : **T1 est VERDICTAL** — mesurer sur un mapping choisi PENDANT le build, ce
serait risquer de choisir celui qui passe. Et cela évite d'acheter 2 à 3 séances contre
un budget qu'on découvrirait faux au premier run.

**Reste inchangé** : k = 4 (σ_ω gaté) ; `run_f1_attribution_b.py` gaté ; miroir CPU non
construit ; kernels intouchés ; gate (ii) F0-cloud toujours dû (~2 min).

## §A26 — Ancre fidèle : mon attente FALSIFIÉE ; dt/frame tranché ; échelle et CFL-par-niveau d'abord (2026-07-19)

Document : fluide-reduit 8ede2e7 (`claude/prereg-ancre-fidele.md`), paper-grade, aucun run.

**ATTENTE DE LA SESSION CRITIQUE : FALSIFIÉE, et rapportée telle quelle.** §A25
anticipait un rapport pas/frame **> 1** qui multiplierait l'ancre. Dérivé du CFL réel
(g = 9.81, dx = 1, h_p = 0.363 ⇒ c = 1.886 ; dt_CFL : 0.212 s/pas au pic du pulse,
0.071 au front type Ritter, 0.049–0.112 en moyenne d'épisode ; phase active
0.05–0.07), il vaut **0.24–0.33, donc < 1**. Claude Code le consigne sans l'habiller :
*« je ne l'ai pas retournée pour coller à l'intuition moteur lourd »*. **L'ancre ne se
multiplie pas en temps réel.**

**RECADRAGE DE LA SESSION CRITIQUE : « rapport < 1 » est de la TÊTE, pas un régime
bursty forcé.** La CFL est une borne SUPÉRIEURE — rien n'oblige à prendre des pas de
taille CFL. **DÉCISION ROMAIN : `dt` = TEMPS DE FRAME, UN PAS PAR FRAME.** Sous-CFL
avec ~4× de marge au niveau grossier, lisse, l'ancre 12.655 s'applique directement, et
c'est le point que M-a-quater a mesuré. **Conséquence sur σ_ω : ce choix NE RÉVEILLE
PAS le fork** — c'est le conservateur (plus de mises à jour, pas moins). Le régime
bursty (`dt` = dt_CFL) reste disponible, mais **décider que le saccadé est acceptable
CONSOMMERAIT un référent temporel ⇒ réveil σ_ω (R4), décision explicite.**

**CORRECTION D'UNE DÉCISION DE LA SESSION CRITIQUE (§A25).** T1 avait été fixé sur la
MÉDIANE pour cohérence avec MORT-a. **Cette décision était CONTINGENTE à un choix de
cadencement non encore fait** : à rapport < 1 la médiane serait une frame SANS pas,
donc vide de sens, tout le coût migrant en p99 (et L1 étale la remontée, pas le pas de
F — stutter que L1 ne couvre pas, comme Claude Code le remonte). **T1-médiane garde son
sens SI ET SEULEMENT SI on prend un pas par frame** — ce qui est désormais tranché.
Le point 2 de Claude Code (« p99 nommée critère ») est donc SUSPENDU à ce choix, non
rejeté : il redeviendrait nécessaire si le régime bursty était un jour acheté.

**RÉSERVE MAJEURE AJOUTÉE PAR LA SESSION CRITIQUE : LA DÉRIVATION EST FAITE À UN SEUL
`dx`.** g/dx/h_p ⇒ c ⇒ dt_CFL, à `dx` = 1. Mais **la pyramide a NEUF niveaux dont le
`dx` varie d'un facteur 2⁹** : la CFL est **PAR NIVEAU**, et au plus fin `dt_CFL` est
~512× plus petit. Un schéma hyperbolique multi-niveaux **SOUS-CYCLE** (Berger-Oliger :
le fin fait 2^k pas par pas grossier) et, à ~autant de cellules par niveau, le coût
devient une somme géométrique — potentiellement catastrophique. **C'est le PIED NON
MESURÉ n°7 tel qu'il était écrit** (« CFL locale ⇒ sous-pas par niveau »). La
dérivation le résout pour UN niveau, **pas pour la pyramide**. Formulé comme QUESTION À
TRANCHER, non comme condamnation : la session critique raisonne depuis les principes
généraux du multi-niveaux, pas depuis le code.

**HALOS — chiffrés, et ce n'est pas le runtime.** Halos ±2 ≈ **1.6 % de l'aire**
(4·512·2 cellules/champ), issus du parent via la prédiction GPU-side prolongée de
2 cellules ⇒ **sur GPU, sans H2D/D2H** ; lancements négligeables, **~0.2–0.5 ms**. Le
coût des halos est le **LOC et la fidélité de bord** (halo parent-grossier vs mur
réfléchi du domaine 64² du rederive), pas le temps.

**RE-CHIFFRAGE TRANCHE-1 (budget frame-pas, 1 pas/frame)** : F 12.655 + bathy 0.1–0.3
+ halos 0.2–0.5 + Exner amorti 0.65–1.25 + remontée 1.28 + transferts 0.11 =
**~15.0–16.1 ms**, marge **0.6–1.7** à 16.7. **T1-médiane n'est PAS déjà compromis —
mais plausible SOUS CONDITION**, et la condition est une décision de design, pas de
physique : temps réel < 1 pas/frame ⇒ passe trivialement ; **point M-a-quater
(1 pas/frame, désormais tranché) ⇒ marge 0.6–1.7 ms, MINCE** (quelques fois la dérive) ;
accéléré > ~2–4× temps réel ⇒ compromis. **Le paramètre décisif est la vitesse de jeu
K, que le CFL ne fixe pas.**

**ET LE TERME QUI PEUT TOUT DÉPLACER (point 4 de Claude Code)** : **l'ancre est à
n_fov = 512, le substrat réel vit à 64².** Si M-b tourne à l'échelle 64, l'aire par
fenêtre chute ~64× et **l'ancre F s'effondre (~0.2 ms, T1 trivial)**. Une ancre mesurée
à 512² pour un substrat à 64² : **c'est la réconciliation d'échelle qui décide du
verdict, pas les autres termes.**

**DÉCISION ROMAIN — ORDRE : (1) RÉCONCILIATION D'ÉCHELLE 64² ↔ n_fov et (2) CFL PAR
NIVEAU / SOUS-CYCLAGE, D'ABORD.** Les deux seuls qui peuvent rendre le reste sans
objet, et les deux sont paper-grade. Les trois autres points — **vitesse de jeu K**,
**p99 comme critère** (suspendu au régime, cf. supra), **condition de mur intérieur** —
suivent, informés. **Aucun achat de M-b tranche-1 d'ici là.**

## §A27 — DÉCOUVERTE DE SPEC : le substrat ne remplit aucun niveau ; T1 SÉPARABLE ; le sous-cyclage devient la question décisive (2026-07-19)

Document : fluide-reduit 8d8fb40 (`claude/reconciliation-echelle-cfl.md`), paper-grade.

**LE FAIT GÉOMÉTRIQUE, depuis le code** : `cote_monde(j) = n0·2^j`, n0 = 256,
niveau_fin = 9 ⇒ le monde de V4 au plus fin fait **131 072²** cellules, la fovéa (512)
en couvre **0.39 %**, `dx` varie de **2⁸ = 256** sur les 9 niveaux GPU. Et le fait
décisif : **le substrat réel 64² est PLUS PETIT que le niveau le plus grossier (256²)
— il ne remplit AUCUN niveau de la pyramide.**

**RÉCONCILIATION D'ÉCHELLE** : le rederive est 64² CPU f64 INTOUCHÉ (la vérité-sol).
Vers le haut, un rederive 131 072² n'existe pas et « intouché » l'interdit ; vers le
bas, 64² < 256² fait **dégénérer** la pyramide. Donc toute mesure exigeant le rederive
tourne à 64², où **l'ancre s'effondre de ~960× : 12.655 → ~0.013 ms** (4 096
block-cells contre 3 932 160). Rapporté sans atténuation par Claude Code.

**CE QUE V4 SIGNIFIE ENCORE** : sa victoire (14.490, gate (i) satisfait) est une
**victoire de GRAND MONDE**, réelle à 131 072², établie par M-a-quater sur un F
**coût-représentatif**. Ce qui s'effondre, c'est la capacité d'une mesure arrimée au
rederive 64² à la tester à l'échelle où la fovéation sert.

**CORRECTION DE LA SESSION CRITIQUE — T1 N'EST PAS VACANT, IL EST SÉPARABLE.**
Claude Code concluait « T1 vacant à 64², un gate qui ne peut pas échouer n'est pas un
gate ». **T1 est une mesure de TEMPS** — « la frame complète avec le F fidèle
dépasse-t-elle 16.7 ? » — et **elle n'a PAS besoin du rederive** ; le rederive
n'intervient que dans MORT-b (Δχ live↔rederive). La scission portait déjà cette
structure (tranche-1 = F fidèle + Exner + pulse + T1, sans rederive ; tranche-2 =
émissions + MORT-b). **DÉCISION ROMAIN : DEUX TRANCHES, DEUX ÉCHELLES** —
**tranche-1 à l'ÉCHELLE V4** (coût réel du F fidèle sur la pyramide, T1 verdictal),
**tranche-2 à 64²** (fidélité, MORT-b contre le rederive intouché). Deux échelles
parce que **deux objets**, pas par défaut.
- **INVARIANT LIANT GRAVÉ : le F fidèle doit être LE MÊME CODE dans les deux
  tranches** — sinon l'une chronomètre ce que l'autre n'a pas validé.
- **CAVEAT DE RÉGIME GRAVÉ** : à l'échelle V4 il faudra SYNTHÉTISER bathymétrie et
  état initial ; **ils devront exercer le MÊME RÉGIME** (fronts wet/dry présents, CFL
  sollicitée). Un domaine tout sec serait rapide et **le chrono ne vaudrait rien**.

**ET CETTE CORRECTION RESSUSCITE L'AVERTISSEMENT DE CLAUDE CODE COMME LA QUESTION
DÉCISIVE.** Il écrit : « pas de sous-cyclage » n'est vrai **qu'à l'échelle dégénérée
64²** ; dès qu'on la quitte, si le niveau fin est sub-résolution la CFL y est violée,
**le sous-cyclage est FORCÉ et le coût fin explose ~100–256×** (Σ n_slots(j)·2^(j−1),
dominée par le fin). Or **tranche-1 tourne précisément à l'échelle V4**, où le niveau
fin est massivement sub-résolution. Arithmétique grossière de la session critique, **à
vérifier** : `dx` variant de 256 sur neuf niveaux ⇒ `dt_CFL(fin)` ~0.8 ms contre une
frame de 16.7 ; en sous-cyclant seulement là où c'est nécessaire (j ≥ 4), la somme
donne **~5× sur F, soit ~63 ms**. Le chiffre de Claude Code est pire. **Dans les deux
cas, V4 MEURT.**
Formulation consignée : la question posée en §A26 n'était **pas déplacée, elle était
prématurée d'un cran** — elle ne mord pas à 64², elle mord **à l'échelle où V4 vit**.

**DÉCISION ROMAIN — PROCHAIN PAS : CHIFFRER LE SOUS-CYCLAGE À L'ÉCHELLE V4**,
paper-grade, aucun run, **avant tout achat**. Il peut **tuer V4 sur le papier** avant
qu'une ligne ne soit écrite ; si F se multiplie par ~5 ou davantage, **les 14.490 ms et
tout ce qui en découle sont à rouvrir**. C'est le **pied non mesuré n°7 arrivant à son
échéance**, et il coûte une séance de papier contre trois de build.

**Les trois décisions de spec remontées par Claude Code restent dues, après** : rôle de
M-b à 64² (T1 étant désormais séparé, MORT-b y redevient le gate propre) ; test de coût
grande échelle (résolu par tranche-1 à l'échelle V4, sans rederive) ; **résolution
physique du niveau fin** — qui EST la question du sous-cyclage, et qui est **une
question ouverte de la SPEC, pas un détail de M-b**.

## §A28 — PROPRIÉTÉ MIPMAP gravée ; M = 1 établi ; V4 SURVIT ; tranche-1 ACHETÉE (2026-07-19)

Document : fluide-reduit 69c9de5 (`claude/sous-cyclage-echelle-v4.md`), paper-grade.

**LE DÉSACCORD EST TRANCHÉ, ET IL ÉTAIT STRUCTUREL.** Le multiplicateur sur F ne
dépend que de **`n_9 = dt_frame / dt_CFL(fin)`** :
`n_j = max(1, ⌈n_9 / 2^(9−j)⌉)` ; `M = Σ blocs(j)·n_j / 15` ; **`M ≈ n_9 / 2`**.
**M est une FONCTION, pas une constante** — nos deux chiffres étaient deux points
arbitraires sur la même courbe.
- Le 100–256× de Claude Code : **faux structurellement** — 2^(j−1) appliqué en aveugle
  (M = 128, indépendant de n_9), soit le Berger-Oliger plein, correct seulement si le
  pas de frame est calé sur la CFL du plus grossier (n_9 = 256) ⇒ double hypothèse
  jamais dite (fovéa sur-résolue 256× **et** ~12× le temps réel).
- **CONCESSION DE LA SESSION CRITIQUE, plus précise que celle qui m'est accordée** : ma
  règle (sous-cycler seulement où `dt_frame > dt_CFL(j)`) était BONNE, mais mon
  placement était **À L'ENVERS** — j'ai mis la résolution de la physique au niveau LE
  PLUS GROSSIER (dx = 1 au grossier, fin 256× plus fin), **fabriquant une fovéa
  sur-résolue par construction**. Claude Code l'a placée où elle est : `run_episode`
  EST le substrat, le niveau FIN EST la physique.

**PROPRIÉTÉ DE SPEC GRAVÉE (load-bearing) :**
> **LA PYRAMIDE EST UN MIPMAP.** Le niveau fin est à la RÉSOLUTION DE LA PHYSIQUE ;
> les niveaux **décimment vers l'extérieur**, jamais ne raffinent **sous la maille de
> base**.

Justification (Claude Code) : `run_episode` est à dx = 1, **il n'existe aucune vérité
sub-cellulaire — une fovéa sur-résolue mesurerait sa propre invention.** C'est la
discipline du projet appliquée à la géométrie.

**COHÉRENCE AVEC LE PILIER, et c'en est la bonne lecture** : « la distance fixe un
**PLAFOND** de LOD » — un plafond, **pas un plancher sous la physique**. La fovéa ne
sert pas à être plus fine que le substrat : elle sert à **NE PAS PAYER LE FIN
PARTOUT**. Lointain décimé, proche à la résolution de la physique.

**SON PRIX, gravé avec elle (une propriété load-bearing sans contrepartie chiffrée
ressemble à un arrangement — le projet a déjà payé pour le nota non chiffré de B4)** :
**raffiner un jour sous dx = 1 exigerait un MODÈLE SOUS-MAILLE, ET ramènerait le
sous-cyclage** — M franchit 2–3 dès **~4–12× de sur-résolution** (la marge disparaît),
et à span plein (256×) **M ≈ 11–31 : catastrophique**. La propriété n'est donc pas
révocable sans conséquence.

**CONSÉQUENCE — M = 1, V4 SURVIT.** À dx = 1, `dt_CFL(fin)` = 0.07–0.21 s et
`dt_frame` = 0.0167 s siège **4–12× sous** ⇒ `n_9` = 0.08–0.24 < 1 ⇒ **AUCUN niveau ne
sous-cycle, M = 1**. Le budget tranche-1 (~15.0–16.1 ms) tient **inchangé**. Et le
caveat de régime de §A27 est **satisfait** : la CFL est SOLLICITÉE (fronts wet/dry,
smax 1.9–5.7, marge 4–12×) sans être violée — **le chrono vaut**.

**COROLLAIRE GRATUIT — la VITESSE DE JEU K est tranchée** (point 1 des quatre décisions
dues) : à K× le temps réel il faut `K · 0.0167 ≤ dt_CFL(fin) = 0.07` ⇒ **K ≤ ~4.2×
avant que le sous-cyclage ne démarre.** Au-delà, M croît et le budget se dégrade.

**SI V4 MEURT UN JOUR (par sur-résolution)** — inventaire gravé de ce qui SURVIT, tout
étant indépendant du dt : **le registre, la géométrie emboîtée (propriété E), L3, la
prédiction gratuite, EPS 1e-2**. À rouvrir : l'intégration mono-dt, la victoire 14.490
(vidée sous sous-cyclage, à re-mesurer), et la question même du besoin de raffiner sous
la physique. **La seule issue qui sauverait une fovéa sur-résolue serait une propriété
de STABILITÉ gravée** — schéma implicite, niveau fin non hyperbolique, ou borne
d'énergie — **aucune n'est acquise**. En leur absence, le sous-cyclage est imposé.

**DÉCISION ROMAIN : ACHAT DE LA TRANCHE-1 DE M-b.** M = 1 est établi, le chiffrage
existe (~2.1–3.4 séances), T1 est verdictal sur la MÉDIANE, K ≤ 4.2 est acquis, et les
deux gardes de §A27 tiennent (même code F dans les deux tranches ; état synthétisé
exerçant le régime). **Ce qui reste — le MUR INTÉRIEUR (fidélité de bord : halo
parent-grossier vs mur réfléchi du domaine 64² du rederive) — est une question de
FIDÉLITÉ, donc de TRANCHE-2**, non bloquante pour une mesure de temps. Les choix de
build seront remontés pour endossement selon la cadence maison.

## §A29 — T1 PRÉ-ENREGISTRÉ et ENDOSSÉ ; C1 gaté à 1.8 séance (2026-07-19)

Document : fluide-reduit d7a8414 (`claude/prereg-t1-fidele.md`). Aucune ligne, aucun run.

**FAIT TECHNIQUE QUI DÉCIDE L'ARCHITECTURE** : le kernel figé a la signature
`(q_in, q_base, q_out, dt, w_base, n, total)` — **aucune bathymétrie**, corrections
hydrostatiques **structurellement nulles** (plat). Le F fidèle exige un opérateur
**b-aware** (reconstruction d'Audusse à `b_eff = b0 + s`) : c'est un **KERNEL NEUF**
(`_SOURCE_FIDELE`), pas une réutilisation. **Donc l'ancre 12.655 n'est qu'un PROXY, pas
la mesure** — dit AVANT la mesure, pas après.

**TROIS POINTS PLUS FORTS QUE CE QUI AVAIT ÉTÉ DEMANDÉ** :
1. **Invariant liant par IDENTITÉ D'OBJET** : une seule fonction `pas_f_fidele`,
   l'échelle est un paramètre de DONNÉES (shapes), jamais une branche ; les deux
   drivers importent le même symbole et le test est `t1.pas_f_fidele is
   t2.pas_f_fidele` — **structurellement vrai, pas surveillé**. Seule couture admise :
   le BORD, injecté comme donnée (réfléchissant en t1, halo-parent en t2) ; l'opérateur
   intérieur reste identique.
2. **Confinement anti-dérive par construction** : tranche-1 ne mesure que du TEMPS,
   donc **aucun bouton « le faire passer » n'existe côté fidélité** ; le critère de
   fidélité appartient à tranche-2. Choix gelés d'avance — un chrono décevant lira
   « ce design coûte tant », jamais « il faut mieux l'implémenter ».
3. **Sémantique de champs gravée** : (h, hu, hv) évolués par le wetdry O2 sur `b_eff`,
   `s` par un kernel Exner séparé, **non advecté** — la physique de `run_episode`, pas
   le motif du jetable.

**BANDE T1, PRÉ-ENREGISTRÉE ET GELÉE (règle §A25 : plus jamais de bande recomposée)** :
F fidèle 10.8–15.2 + Exner 1.5–3.0 + (remontée L3 + transferts + dérive) 1.21–1.57 ⇒
**bande [13.5, 19.8]**, centre ~16.4, **qui STRADDLE le seuil 16.7**. C'est
précisément pourquoi T1 est verdictal : **sur le papier, la survie de V4-b est sur le
fil, seule la mesure tranche.** Hors bande = **AUTRE** (modèle de coût faux), distinct
du verdict de mort. **Réserve de halo explicite** : la bande est à bords réfléchissants,
elle SOUS-COMPTE les halos de tranche-2 ; si la médiane dépasse 16.2, cette réserve
seule peut la porter au-delà de 16.7 — le verdict rapportera les deux.

**TROIS AMENDEMENTS EXIGÉS PAR ROMAIN (session critique) :**
- **(A) LA BANDE EST LARGE PARCE QUE SON TERME DOMINANT EST INCONNU — dit AVANT le
  run.** F fidèle : 40 % d'étalement ; bande totale : 47 %. **Une bande aussi large ne
  peut presque pas échouer** ; « dans la bande » ne testera rien (§A25 a déjà coûté une
  gravure pour avoir laissé une bande dire plus qu'elle ne pouvait). Elle reste honnête
  — plus étroite serait de la fausse précision. **DONC : le DIAGNOSTIC PRINCIPAL de T1
  devient `F_fidèle / F_proxy` (12.655), le RAPPORT DE REPRÉSENTATIVITÉ** — dérivable
  comme les parts de M-a-quater (total − Exner − remontée − transferts). **C'est la
  raison d'être originelle de T1** ; la bande passe au second rang.
- **(B) LA MARGE CFL EST VÉRIFIÉE SUR LA SÉRIE, pas seulement à l'initialisation.** Le
  well-balancing d'Audusse en **f32** est délicat ; une C-property cassée engendre des
  vitesses parasites ⇒ `smax` monte, `dt_CFL` descend, et à `smax` ≈ 50 la marge de
  4–12× est mangée : **la CFL est violée et le chrono mesurerait une simulation qui
  DIVERGE**. Fail-loud si la marge s'effondre en cours de série. C'est le lien entre C1
  et la validité de T1.
- **(C) `_SOURCE_FIDELE` prend son VERROU D'EMPREINTE DÈS LE PREMIER COMMIT** — L3 est
  resté sans verrou pendant des jours alors qu'il portait `reference += d`. On ne refait
  pas cela sur un kernel neuf.

**CAVEAT DE RÉGIME (câblage B9, reconduit)** : `verifier_regime` PASS ssi l'état
synthétisé exerce le régime — fronts wet/dry (fraction mouillée ∈ [0.2, 0.8]) et CFL
sollicitée à 4–12× de marge. Hors bande ⇒ **AUTRE d'instrument**. `lecture_t1` refuse
fail-loud sans le PASS.

**CE QUE LE BORD DE t1 IMPOSE À TRANCHE-2** : delta de coût borné (+0.2–0.5 ms, GPU
sans transfert) ; une couture, pas une refonte ; et **interdiction de réutiliser le bord
réfléchissant pour la fidélité** — physiquement faux, tranche-2 doit le remplacer par le
halo-parent. **Tranche-1 est donc liée à rendre le bord PLUGGABLE**, sinon l'invariant
liant tombe.

**CHIFFRAGE ET GATE — DÉCISION ROMAIN** : total ~1 450–2 100 LOC, **2.6–4.1 séances**,
la fourchette haute dépassant le cap 3.4. Ce qui fait basculer : **la reconstruction
bien équilibrée à `b` réel n'a JAMAIS été exercée sur GPU** (le figé est plat) — c'est
le débogage de fidélité qui glisse. **POINT DE CONTRÔLE C1 ENDOSSÉ : kernel b-aware +
C-property vérifiée sur terrain réel, SEUIL PRÉ-ÉCRIT À 1.8 SÉANCE.** Si C1 glisse
au-delà, **on le sait AVANT d'avoir construit le driver** — scission récursive du même
motif que A/B et que la borne L3.

### §A29-C1 — Kernel fidèle b-aware LIVRÉ et ENDOSSÉ ; halo RK2 et cadence Exner (2026-07-19)

Build : fluide-reduit e78e79a. **Vérifié indépendamment** : `_SOURCE_FIDELE` 9248 car.
**6dd207ca…6265d** ; `_SOURCE` fusionné 8223/**e18015f5** ; `_SOURCE_L3`
9635/**9533a130** — les trois CONCORDENT ; arbre propre, kernels figés intouchés.
**Le verrou d'empreinte est posé DÈS le premier commit** (amendement C) : la leçon de
L3 est appliquée.

**FAIT CONFIRMÉ AU CODE** : le kernel figé n'a **aucune entrée de bathymétrie**
(η = h, z* = 0, corrections nulles) — il **ne peut pas** porter la physique réelle.
`_SOURCE_FIDELE` est donc un kernel NEUF ; seul le préfixe b-agnostique
(`#define`, structs, `desing`, `minmod`) est repris tel quel (`_PREFIXE_BASE`, même
motif que L3). Neuf et b-aware : `lire_b`, `pentes_axe_b` (pentes sur η = h + b),
`flux_1d_b` (**Audusse** : z* = max(bL,bR), h* = max(0, η − z*), pression
(g/2)(h² − h*²)), `divergence_axe_b`, étage global.

**C-PROPERTY PROUVÉE SUR TERRAIN RÉEL** — ce que le figé plat ne pouvait pas tester :
lac au repos sur `default_terrain`, **vitesse parasite max ~1e-8** (soit **un ulp de
f32** à h ≈ 0.363 : zéro machine), pleinement mouillé ET au front wet/dry, stable sur
8 pas. **Premier coup sur GPU** — c'est le morceau que Claude Code avait lui-même
désigné comme le plus susceptible de glisser, et il n'a pas glissé.

**BORD PLUGGABLE, INVARIANT LIANT PROUVÉ** : le bord est une DONNÉE (`mode`), pas un
fork — **mode 0 (mur) ≡ mode 1 (halo = réflexion exacte) = 0.000e+00 bit-exact**.
L'opérateur intérieur est le même objet. L'invariant n'est pas surveillé, il est VRAI.

**LA TROUVAILLE DU TOUR — le halo est partagé par les deux étages RK2.** En mode
réfléchissant il est **auto-consistant** (la réflexion est fonction de l'intérieur,
donc fraîche par construction) ; **en halo-parent il est PÉRIMÉ au second étage** ⇒
tranche-2 devra le **rafraîchir entre étages**, la prédiction de halo tournant **deux
fois par pas**. Le 0.186 initial était exactement cette péremption : **diagnostiqué,
pas rustiné**. Conséquences chiffrées gravées :
- poste halo : **0.2–0.5 → 0.4–1.0 ms** ;
- le seuil de déclenchement de la réserve de bande **glisse de 16.2 à ~15.7**.

**CE QUE CELA AFFINE DANS T1 (et qui rend l'amendement (A) plus juste que prévu)** :
**le RAPPORT `F_fidèle / F_proxy` est PROPRE** — les deux tournent à bords
réfléchissants, c'est apples-to-apples ; **le VERDICT ABSOLU SOUS-COMPTE la
production**, qui utilise des halos-parents. Le diagnostic principal est donc le terme
SAIN, et le verdictal porte la RÉSERVE. Note mineure : la C-property est vérifiée sur
**8 pas** quand la série chrono en fait 300+ ; l'amendement (B) couvre la conséquence
(divergence via `smax`), pas la dérive de la C-property elle-même.

**AMENDEMENT ROMAIN AVANT C2 — LA CADENCE EXNER EST DÉRIVÉE, JAMAIS CHOISIE.** Le
rederive fait ~600 pas wetdry pour ~300 Exner, soit **un Exner tous les deux pas**. Or
la bande alloue « Exner (par frame) » 1.5–3.0 ms, ce qui suppose CHAQUE frame. **La
cadence doit être DÉRIVÉE de la structure RÉELLE du rederive** (quelle qu'elle soit),
pas fixée au build — **sinon tranche-2 comparerait deux physiques différentes, et
l'invariant « même code F » ne couvre PAS la cadence.** Si la cadence réelle est une
frame sur deux, le poste tombe à **0.75–1.5** et la bande haute descend à **~18.3**.

**ENVELOPPE** : C1 a tenu sa fourchette (1.0–1.8 séance) — **aucun POINT D'ARRÊT de
coût déclenché**. 386 tests F1 verts, suite complète 979 (gel bit-exact inclus).
**DÉCISION ROMAIN : C1 ENDOSSÉ, poursuite C2–C5** avec les trois consignes ci-dessus.

### §A29-C2/C4/C5 — TRANCHE-1 DE M-b CONSTRUITE ET ENDOSSÉE ; T1 prêt à courir (2026-07-19)

Builds : fluide-reduit f2e1d7c (C2 Exner), d2afbfc (C4 régime), 558e06e (C5 driver).
**Vérifié indépendamment** : QUATRE empreintes concordent — fidèle **6dd207ca**/9248,
Exner **3533fd0b**/1251, fusionné **e18015f5**/8223, L3 **9533a130**/9635 ; kernels
figés et C1 intouchés.

**CADENCE EXNER — LUE, PAS CHOISIE (amendement §A29-C1)** : `run_episode` fait 300 pas
Exner pour 600 wetdry (`save_every = 2`) ⇒ **un Exner tous les deux pas, exactement
un-sur-deux**. `CADENCE_EXNER` **lit `SedimentParams.save_every`**, et une
CONTRE-ÉPREUVE vérifie qu'elle **suivrait** un changement du rederive — « dérivée » au
sens fort, pas recopiée. Conséquence conforme : poste Exner **[1.5, 3.0] → [0.75, 1.5]**.

**C2 — Exner GPU** : portage pointwise de `_exner_step`, `s` évolué par ce kernel (non
advecté) ; équivalence GPU↔CPU à **1.87e-9** (dépôt si θ<θc, érosion si θ>θc, masque
wet, s ≥ 0). Verrou d'empreinte **dès le premier commit**.

**C4 — RÉGIME SUR LA SÉRIE (amendement B)** : l'état synthétisé à l'échelle V4 (512²)
**exerce** le régime — fronts wet/dry, marge CFL **5.42 ∈ [4, 12]**.
`verifier_regime_serie` est **fail-loud si la marge s'effondre sur une SEULE frame**
(C-property cassée ⇒ `smax` monte ⇒ le chrono mesurerait une divergence) ;
`exiger_regime_pass` refuse toute lecture sans PASS. **Passe SÉPARÉE du chrono** pour
ne pas y injecter la synchronisation de réduction — même hygiène que D-2.

**C5 — driver T1, LIVRÉ NON EXÉCUTÉ** : **`F_fidèle / F_proxy` EN TÊTE** (amendement
A), bande au second rang ; **bande RE-GELÉE [12.76, 18.27]** avant tout run et plus
jamais recomposée (§A25) ; réserve de halo à part, seuil glissant **15.7** ; verdict sur
la **MÉDIANE**, seuil 16.7 ; p99 en évidence.

**DEUX VÉRIFICATIONS QUI ONT ÉVITÉ DES DEMANDES INUTILES** (consigné, parce que le
contrôle vaut aussi quand il ne trouve rien) : (i) la bande recompose **exactement**
(10.8+0.75+1.21 = 12.76 ; 15.2+1.5+1.57 = 18.27) ; (ii) **`F_fidèle` est
CHRONOMÉTRÉ DIRECTEMENT** (`_chrono_f_fidele`, passe séparée) et **non dérivé par
soustraction** — le piège de M-a-quater, où la prédiction calculée en SOLDE était
ressortie NÉGATIVE, était déjà évité.

**AMENDEMENT ROMAIN AVANT RUN — LE VERDICT PREND TROIS VALEURS PRÉ-ÉCRITES.** La
réserve était reportée **à côté** du verdict (`verdict_sous_reserve`,
`mediane_plus_reserve_haute_ms`) ; l'information y était, **le label non**. C'est le
cadrage exact qui a rendu le 16.589 de M-a-quater lisible en diagonale comme un succès.
Donc, gravé avant la mesure :
> **médiane > 16.7 ⇒ MORT** ;
> **médiane ∈ [15.7, 16.7] ⇒ SANS MORT SOUS RÉSERVE DE HALO** (T1 passe, mais la
> production avec halo-parent dépasserait) ;
> **médiane < 15.7 ⇒ SANS MORT.**

**PORTÉE À FAIRE VOYAGER AVEC LE RAPPORT** : `F_fidèle / F_proxy ≈ 1` dira que les deux
**COÛTENT** pareil — ce qui est bien la question de représentativité telle qu'elle fut
posée — mais **NON qu'ils FONT la même chose** (le proxy fait 8 mises à jour de champ à
fond plat ; le fidèle en fait 3 plus la reconstruction b-aware). **Un rapport proche de
1 ne validera PAS le raisonnement E4a du « majorant honnête ».**

**DÉCISION ROMAIN : TRANCHE-1 ENDOSSÉE, T1 PEUT COURIR** sur le label à trois valeurs
intégré. 413 tests F1 verts (27 neufs). C'est **la première fois du projet qu'une
physique RÉELLE tourne à l'échelle de la fovéa**.

## §A30 — VERDICT T1 : SANS MORT ; l'ancre était HONNÊTE ; tranche-2 achetée (2026-07-19)

Run : fluide-reduit fb1a700, natif, machine-instrument. **Première fois du projet
qu'une physique RÉELLE (Audusse bien équilibré + Exner + bathymétrie) tourne à
l'échelle de la fovéa.**

**CHIFFRE DE TÊTE — `F_fidèle / F_proxy` = 11.696 / 12.655 = 0.924.** Le moteur réel
coûte **MOINS** que le proxy : le transport 8-champs à fond plat était un peu plus cher
que 3 champs + reconstruction d'Audusse. **L'ancre sur laquelle repose TOUT le budget
de V4 était honnête, et légèrement PESSIMISTE.** (Et cela **ne valide pas E4a** — cf.
portée : coûter pareil n'est pas faire pareil.)

**VERDICT T1 PRONONCÉ (Romain, 2026-07-19) : SANS MORT.** Médiane frame complète
**13.332 ms**, bien sous le seuil de réserve 15.7 ; **même avec la réserve de halo
haute (+1.0) ⇒ 14.332 < 16.7** : SANS MORT tient SOUS la réserve. Le label à trois
valeurs, gravé avant le run, a fait son travail. **Régime PASS sur la série** (301
frames, marge CFL stable [5.21, 6.92] ⊂ [4, 12], fraction wet ~0.45, pas
d'effondrement) : le chrono mesure bien le F fidèle, **pas une divergence**. Bande
(second rang, NON recomposée) : 13.332 ∈ [12.76, 18.27] ⇒ **pas un AUTRE**.
Résidence 160 Mio. **V4-b survit T1 avec marge des deux côtés.**

**QUATRE RÉSERVES GRAVÉES AVEC LE VERDICT :**
1. **LA FRAME COMPLÈTE N'A PAS ÉTÉ MESURÉE BOUT-EN-BOUT : ELLE EST COMPOSÉE.**
   11.942 MESURÉS (F + Exner) + **1.28 et 0.11 IMPORTÉS** de M-a-quater — donc mesurés
   avec le F **JETABLE**, dont la distribution de détails émis n'est pas celle du
   fidèle. Le verdict repose en partie sur des termes non re-mesurés. **Robuste** (il
   faudrait TRIPLER les termes importés pour atteindre 15.7) — mais dit.
2. **LA PLUS LOURDE — le fidèle tourne à 4 CHAMPS quand le budget était épinglé à
   c = 8.** Proxy : 8 mises à jour à fond plat pour 12.655. Fidèle : 3 (plus Exner)
   pour 11.696. **PAR CHAMP, le fidèle coûte ~2.5× le proxy.** Donc **le proxy était
   conservateur EN TOTAL et OPTIMISTE PAR CHAMP** : tout enrichissement du substrat-jeu
   — et le gameplay pousse dans ce sens (fenêtres d'énergie comme ressource) — partira
   d'une base par-champ bien plus chère que l'ancre ne le suggère. **C'est la note de
   §6 (« c = 8 est l'enveloppe de travail, à re-épingler quand le substrat v-jeu se
   fige ») qui devient CONCRÈTE.**
3. **Le poste EXNER était surestimé 3 à 6×** (0.246 mesuré/frame amorti contre
   [0.75, 1.5] pré-enregistré) : la bande passe **en partie par compensation**, et le
   modèle de coût a **un terme faux**. Reporté SANS recomposer — discipline §A25 tenue.
4. **Le p99 = 12.284 N'EST PAS COMPARABLE à celui de M-a-quater** (16.853) : il mesure
   F+Exner sur un **état synthétisé lisse**, sans les **bursts de remontée** qui
   produisaient l'autre. « Pas de stutter » est scopé à cela.

**DETTE NOMMÉE (non armée)** : mesure de la frame BOUT-EN-BOUT avec le F fidèle (pour
remplacer les 1.39 ms importés). **Écarte par le critère du tapis roulant** : il
faudrait tripler ces termes pour changer le verdict — la mesure ne peut pas changer la
décision.

**DÉCISION ROMAIN : ACHAT DE LA TRANCHE-2 DE M-b** (~1.5–2.3 séances) — émissions
3 seeds {101,102,103} / Δt=4 / 6 émissions, **MORT-b PAR-SEED contre 0.0733**, à 64²
contre le rederive INTOUCHÉ. C'est **le contenu réel de M-b et le gate (iii) de la
spec** — celui qui décide **Option A** (quarantaine du non-déterminisme) ou **repli
Option B**. Le **halo rafraîchi entre étages RK2** y est déjà nommé (§A29-C1), ainsi
que l'interdiction de réutiliser le bord réfléchissant pour la fidélité.

## §A31 — TRANCHE-2 pré-enregistrée : réconciliation 64² ENDOSSÉE + BRAS TÉMOIN exigé (2026-07-19)

Document : fluide-reduit 88d3f87 (`claude/prereg-t2-fidelite.md`). Point d'arrêt honoré
sur un choix load-bearing plutôt que préempté.

**LA RÉCONCILIATION D'ÉCHELLE, ENDOSSÉE.** Au 64² du rederive la pyramide de V4
dégénère (§A27), or §A29-C1 interdit le bord réfléchissant pour la fidélité — et un
halo-parent n'existe que si la fovéa est une sous-fenêtre à bord INTÉRIEUR. D'où la
**fovéation à DEUX NIVEAUX** : **grossier** = 64² décimé, **murs réfléchissants au VRAI
bord** (= le rederive, physiquement correct) ; **fin** = fovéa pleine résolution, **bord
halo-parent** depuis le grossier upsamplé. Cohérent avec le mipmap gravé (§2-rev2) : le
fin est à la résolution de la physique. **Fovéa mobile E4c et décimation 2× endossées**
— les plus proches de la production.

**PROBLÈME DE SENS EXPOSÉ PAR CETTE STRUCTURE (session critique)** : l'écart
live↔rederive aura **TROIS sources** — (a) arithmétique f32 GPU + ordre des réductions
(**la cible déclarée**) ; (b) remontée seuillée à EPS = 1e-2, **perte délibérée** ;
(c) **structure fovéale** (lointain décimé, halo-parent, colonnes entrantes). **Or le
remède pré-écrit de MORT-b est Option B (« tout-déterministe »), qui ne corrige que
(a).** Si la série traverse à cause de (b) ou (c), on prononcerait la mort d'Option A
en appliquant un remède **incapable de guérir la cause**. Le contrat É2 portant bien
sur le TOTAL (« tout readout émis est re-dérivable sous JND_sev »), mesurer le total
est juste — **ce qui manquait, c'est de savoir LEQUEL des trois a parlé.**

**AMENDEMENT ROMAIN — BRAS TÉMOIN ISOLANT (a), exigé avant build :** fidèle F sur GPU
f32 au **64² PLEIN DOMAINE, UN SEUL NIVEAU**, murs réfléchissants — configuration
**identique au rederive**, seule l'arithmétique diffère ; **pas de pyramide, pas de L3,
pas de L1**. Nota : à cette configuration **l'objection de C1 tombe d'elle-même** — le
bord y est le **VRAI** bord du domaine, donc réfléchissant est physiquement correct ;
l'interdiction visait les bords **INTÉRIEURS** de fenêtre.
**LECTURE PRÉ-ÉCRITE, décidable :**
> **bras de production traverse ET témoin PASSE ⇒ la cause est (b)+(c) — OPTION B
> N'EST PAS LE REMÈDE** (la décision porte alors sur EPS et/ou la structure fovéale) ;
> **les deux traversent ⇒ c'est (a) — Option B EST le remède**, et MORT-b se prononce
> comme gravé ; **aucun ne traverse ⇒ Option A tient**, gate (iii) satisfait.

**DEUX PORTÉES À FAIRE VOYAGER AVEC LA LECTURE** : (1) à 64² sur deux niveaux, **la
machinerie L1/L3 n'est exercée que MARGINALEMENT** (peu de slots, cadence k=4 quasi
vide) — la mesure **ne les valide pas à l'échelle V4** ; (2) la fovéa mobile parcourra
**~24 cellules sur les 6 émissions**, soit un tiers du domaine — **la contribution des
colonnes entrantes y sera forte**.

**GELÉ PAR AILLEURS (reconduit)** : cellule §A15 (3 seeds {101,102,103}, Δt=4,
6 émissions, commits fenêtrés, k_fen aire-proportionnel, cap 10 %) ; **MORT-b = max de
série Δχ > 0.0733 PAR-SEED** ; rederive `run_history` 64² CPU f64 **INTOUCHÉ** ;
observable = Δχ readout sur la connaissance du CPU (option 1) contre la vérité f64
pleine ; **halo-parent RAFRAÎCHI ENTRE LES ÉTAGES RK2** (trouvaille C1, dont le 0.186
était la signature) — fait **au driver**, jamais en modifiant `pas_f_fidele` ;
**invariant liant `t1.pas_f_fidele is t2.pas_f_fidele`** (identité, pas diff) ; cadence
Exner toujours **lue** de `save_every`.

**CHIFFRAGE** : ~1 120–1 650 LOC, **1.7–2.6 séances** (cohérent §A28), plus le bras
témoin. Poste qui peut glisser : **le refresh halo inter-étage** (objet neuf, jamais
tourné) — **POINT D'ARRÊT si au-delà de 0.3 séance avant le driver**.

### §A31-build — Trois pièces load-bearing livrées ; GATE (iii) RE-SCOPÉ avant le run (2026-07-19)

Builds : fluide-reduit 8710288 (lecture 3 branches), a06ab4a (refresh halo
inter-étage), 74c5188 (bras témoin). **Point d'arrêt AVANT l'assemblage** — le bon
réflexe : un moteur de fidélité assemblé à la va-vite serait le glissement que Claude
Code avait lui-même nommé. 436 tests F1 verts, kernels figés + C1–C5 intouchés
(quatre empreintes tenues).

**LE BRAS TÉMOIN A DÉJÀ FAIT L'ESSENTIEL DE SON TRAVAIL AVANT DE TOURNER :
(a) = 6e-5 d'écart relatif** (témoin f32 plein domaine vs rederive f64, config
identique). Si l'arithmétique f32 contribue à ce niveau, elle siège **trois ordres sous
le seuil 0.0733**. Conséquence sur la lecture pré-écrite : **la branche « les deux
traversent ⇒ (a), Option B est le remède » devient TRÈS IMPROBABLE** ; si la production
traverse, ce sera presque certainement **(b)+(c)** — donc avec un remède qui **n'est pas
Option B**. C'est exactement ce que le bras témoin existait pour établir, et il
l'établit avant la mesure. **RÉSERVE** : 6e-5 est une **vérification d'ÉTAT au build**,
pas la mesure pré-enregistrée en **Δχ** sur la cellule — **le témoin doit tourner comme
prévu, l'instrument n'est pas le même.**

**LE REFRESH HALO CONFIRME LE DIAGNOSTIC DE C1 PAR UNE SECONDE MESURE INDÉPENDANTE**
(0.34 contre figé — la signature du 0.186), avec **C-property 2-niveaux à 1.2e-8** : le
bien-équilibré survit à la structure. Tenu en ~90 LOC, **bien sous le point d'arrêt de
0.3 séance**. Fait à la couche t2, **jamais dans `pas_f_fidele`** — l'invariant liant
tient.

**PROBLÈME DE GATE, POSÉ AVANT LE RUN ET NON APRÈS.** Le gate (iii) est gravé
« **M-b sans mort SUR V4** » (§A18). Or §A27 a établi que M-b **doit** tourner à **64²**
— le rederive y vit et il est intouchable. **Le gate, tel qu'écrit, est STRUCTURELLEMENT
INSATISFIABLE.** Et les deux portées de Claude Code disent pourquoi un PASS ne pourrait
pas le fermer par surclame : à 64² sur deux niveaux, **L1 et L3 — la machinerie même qui
pourrait rompre le contrat — ne sont exercées que MARGINALEMENT**, et le lointain n'a
pas les neuf niveaux de décimation de V4.

**DÉCISION ROMAIN — GATE (iii) RE-SCOPÉ, gravé AVANT la mesure :**
> **(iii) M-b SANS MORT À 64², SUR FOVÉATION 2-NIVEAUX.** Le contrat à l'échelle V4 —
> où L1/L3 travaillent pleinement et où le lointain porte neuf niveaux de décimation —
> devient une **TRANSPOSITION NOMMÉE**, `[TRANSPOSITION-HYPOTHÈSE]`, **non mesurée**.
> Son falsificateur exigerait un rederive à l'échelle V4, que la règle « rederive
> INTOUCHÉ » interdit ; il reste donc **nommé et non armé**. **Un PASS à 64² ne pourra
> pas être lu comme « le contrat tient à l'échelle V4 ».**

**PROCHAIN PAS ENDOSSÉ : l'ASSEMBLAGE** — évolution production (fovéation 2-niveaux via
`pas_deux_niveaux` + structure d'épisode du témoin + remontée L3 à EPS 1e-2 +
connaissance CPU option 1) et driver (3 seeds, Δt=4, 6 émissions, MORT-b par-seed câblé
à la lecture 3 branches), **construit-non-lancé**. Aucun run avant endossement.

### §A31-assemblage — ENDOSSÉ, T2 prêt à courir ; ce qu'une traversée (b) rétroagirait (2026-07-19)

Build : fluide-reduit 5d051f1. **Vérifié indépendamment** : quatre empreintes
concordent (6dd207ca / 3533fd0b / e18015f5 / 9533a130), kernels figés + C1–C5
intouchés. Re-scope du gate (iii) présent dans le champ de lecture.

**LE VERROU DES CENTRES — la meilleure prise de l'assemblage.** Le rederive est
intouché, il ne rend pas ses centres. Plutôt que de SUPPOSER que le tirage de la
production reproduit le sien, un test **rejoue un épisode avec les centres de la
production contre `run_history` lui-même** : bit-pour-bit. **Sans ce verrou, Δχ aurait
mesuré un écart d'HISTOIRE lu comme un écart de FIDÉLITÉ** — exactement le trou qui a
coûté la journée sur la sonde EPS, refermé ici par construction. `cellule_mb.py` fait
partager aux deux bras la cellule, l'observable et le budget : **apples-to-apples
STRUCTUREL, pas déclaratif.** Canal testé aux deux bouts (seuil ∞ ⇒ CPU muet même si le
live a évolué ⇒ c'est bien la `reference` qui est lue, pas le champ ; seuil 0 + budget
large ⇒ CPU exact).

**CINQ CHOIX NON COUVERTS PAR LE GRAVÉ, ENDOSSÉS** : (1) cadence de remontée = une par
épisode (le seuil mord 4× entre deux émissions) ; (2) le cap 10 % borne les détails L3
transportés, plus gros |Δ| d'abord — sans cette lecture le cap serait décoratif ;
(3) dt lockstep = min(CFL fin, CFL grossier) ; (4) **échelle de lecture = 64² PLEIN, pas
décimé — le choix le plus SÉVÈRE, conservateur** (un PASS y est plus fort) ; (5)
géométrie fovéa (côté 32, 1 cellule/épisode ⇒ ~un tiers du domaine parcouru, asserté).

**ENJEU RÉTROACTIF NOMMÉ AVANT LE RUN (session critique)** : **T2 est la PREMIÈRE fois
que la contribution d'EPS = 1e-2 à l'écart live↔rederive est mesurée contre le VRAI
rederive f64.** La sonde EPS (§A23) avait établi l'innocuité du **CANAL** — Δχ contre
une vérité DÉCIMÉE, connaissance CPU comprise. Ici la production porte sa remontée
seuillée à 1e-2 contre la **vérité f64 PLEINE**. Donc **si la série traverse par (b)**,
la lecture ne dit pas seulement « Option B n'est pas le remède » : elle dit que
**l'application d'EPS = 1e-2 à la production (§A24) était trop lâche contre le contrat
réel** — conséquence rétroactive sur une décision gravée. Le bras témoin sépare (a) ;
**rien ne sépare (b) de (c) dans ce build**, mais **EPS est réglable et la structure ne
l'est pas**, donc la branche (b)+(c) **rouvrirait d'abord EPS**. Déjà porté par la
lecture 3 branches — rendu explicite ici, avant les chiffres.

**RÉSERVES D'INSTRUMENT (Claude Code, reconduites)** : `ruff` **absent de
l'environnement** — pas de « ruff clean » revendiqué ; contrôle F401 par AST fait à la
place (rien, hors le faux positif `from __future__`). Coût du run : **~3.5 min**
(3 seeds × 24 épisodes × 3 bras) — l'endossement ne coûte pas une séance.

**DÉCISION ROMAIN : ASSEMBLAGE ENDOSSÉ, T2 PEUT COURIR.** 48 tests tranche-2 verts, 492
sur la suite. Lecture mécanique remontée à trois branches, verdict à Romain — **c'est
le dernier des trois gates de la spec.**

### §A31-run — ÉCHEC D'INSTRUMENT sur T2 (OOM) : le gate (iii) est NON MESURÉ (2026-07-19)

**Le run de T2 n'aboutit pas : OOM, processus tué par le système.** Consigné comme
**ÉCHEC D'INSTRUMENT, PAS COMME RÉSULTAT** : le gate (iii) n'est **pas raté**, il est
**NON MESURÉ**. Aucune lecture, aucun verdict, aucune branche prononcée.

**CE QUE L'INSPECTION DU CODE ÉTABLIT (session critique, avant tout diagnostic
d'exécution)** : **rien dans T2 n'explique un OOM à 64²**. La configuration est
correcte (n = 64, `N_FOVEA` = 32) ; les buffers L3 sont dimensionnés pire-cas mais sur
**4096 cellules** ; la production n'alloue que des champs minuscules ; le driver libère
le mempool (`free_all_blocks`). **T1 tournait à 512² sur 301 frames avec 0.564 Go sans
incident.** ⇒ **le débordement ne vient vraisemblablement PAS du GPU.**

**CE QUI EST NEUF DANS T2** : le **rederive CPU f64 sur 72 épisodes** (3 seeds × 24), et
« le système coupe les runs » est la signature de l'**OOM killer Linux** ⇒ **RAM HÔTE**.
**Candidat précis, issu de Claude Code lui-même** : *« `_relax_episode` intègre toujours
jusqu'à `_T_END_RELAX` puis tronque »* — si cette intégration MATÉRIALISE la trajectoire
complète avant de tronquer, elle alloue un ordre de grandeur au-dessus de ce que la
cellule demande, **72 fois**. `run_history` ne stocke que les checkpoints (vérifié), donc
l'accumulation, si elle existe, est en amont.

**DISCRIMINATEUR À UNE COMMANDE** : `cupy.cuda.memory.OutOfMemoryError` dans la trace ⇒
**VRAM** ; simplement `Killed` (ou `dmesg | grep -i oom`) ⇒ **RAM HÔTE**. Les deux ont
des causes et des remèdes disjoints.

**DÉCISION ROMAIN : DIAGNOSTIQUER L'ALLOCATION, CELLULE INTACTE.** Instrumenter le pic
mémoire **par bras et par phase** (VRAM via mempool, RAM hôte via `resource`/`psutil`),
identifier le poste qui déborde, et **corriger l'INSTRUMENT** (séquencer les épisodes,
libérer entre seeds, streamer le rederive) — jamais la mesure.

> **INTERDIT EXPLICITE, gravé : la cellule §A15 ne bouge pas.** 3 seeds {101,102,103},
> Δt = 4, 6 émissions. **Rétrécir la mesure pour qu'elle entre dans l'instrument est
> l'INVERSION INTERDITE** — refusée à l'identique sur la parité des 17 blocs
> (§A22-complément : « modifier le design épinglé pour satisfaire l'instrument aurait
> été l'inversion interdite »). Et **MORT-b est PAR-SEED** : sur un seul seed, le
> critère perdrait son sens.

**LEÇON D'INSTRUMENT RECONDUITE** : les états d'échec se **persistent** (leçon gravée du
FAIL cloud non persisté). Le diagnostic doit **capturer** le pic et la phase, pas
seulement constater la mort du processus.

### §A31-diagnostic — OOM ATTRIBUÉ (RAM hôte) ; correction d'instrument ENDOSSÉE ; risque neuf nommé (2026-07-19)

Diagnostic : fluide-reduit 2402f78 (`claude/diagnostic-oom-t2.md`). Aucun run de mesure.

**1. DISCRIMINATION — RAM HÔTE, verbatim.** Aucune trace Python (ni
`cupy.cuda.memory.OutOfMemoryError`, ni traceback) : `dmesg` donne
`Out of memory: Killed process 37234 (python) total-vm:33600364kB,
anon-rss:23569692kB` puis un second à **25 088 636 kB**. **Mempool CuPy plafonné à
34 Mo mesurés — le GPU est HORS DE CAUSE**, et une saturation VRAM aurait levé une
exception, pas un SIGKILL. L'inspection de §A31-run est confirmée.

**2. LE POSTE, AU CENTRE PRÈS** — seed 103, épisode 4, centre (0.3524, 0.6131). **Le
`dt` ne s'effondre pas au départ : il s'effondre EN COURS d'épisode** —
t_end 32 : 269 pas, dt médian 1.05e-1, 26 Mo ; t_end 64 : **3 173 pas**, dt médian
5.5e-3, **312 Mo** ; t_end 130 (le réel) : **~254 000 pas, ~25 Go**. Les 601 pas
retenus couvrent t = 48.44 sur 130 : **besoin réel 59 Mo, ~99.8 % calculé puis jeté.**
`run_history` est hors de cause (ne stocke que les checkpoints, vérifié).

**3. CORRECTION — LE REDERIVE RESTE INTOUCHÉ** (`git diff` vide sur `sediment.py` et
`solver_wetdry.py`). `t_end` ne gouverne que **l'arrêt de boucle et l'écrêtage du
dernier `dt`** ⇒ on consomme **au plus petit barreau d'une échelle plafonnée au `t_end`
gravé**, le domaine de résultats restant exactement celui du rederive, dont le
`RuntimeError` sert d'**oracle**. **Piège visé et fermé** : si le compte de pas tombait
exactement à `N_settle`, la dernière entrée retenue serait le pas ÉCRÊTÉ, différent du
rederive non borné — l'échelle cherchant le plus petit barreau qui passe, **ce cas est
structurellement visé** ⇒ marge d'un pas.
**DEUX ERREURS DE CLAUDE CODE, corrigées par la mesure et consignées** : (i) sa
proposition de modifier `simulate_wetdry_o2` — exclue par §A31-run, le protocole fait le
même travail sans toucher la vérité-sol ; (ii) son premier mémo **monotone**, FAUX :
**ep3 exige 130, ep4 se contente de 64** — hériter du 130 tuait le processus ;
**l'escalade repart du bas.** Propriété de sûreté nommée : *un épisode exigeant un
`t_end` élevé a un `dt` large, donc peu de pas — l'épisode coûteux est justement celui
qu'un `t_end` bas satisfait.* Cellule entière sous **RLIMIT_AS = 4 Go : pic 2.2 Go** —
**la garde est une PREUVE, pas un correctif** (le processus n'aurait pas pu dépasser
sans lever).

**AMENDEMENT ROMAIN — GARDE STRUCTURELLE PAR ÉPISODE.** Le test de bit-exactitude à
`N_settle = 600` est un **échantillon** ; la propriété se vérifie **par épisode et
gratuitement** : **asserter que le temps du dernier pas RETENU est strictement inférieur
à `t_end` moins la marge**. Alors le pas écrêté est hors fenêtre **par construction** —
plus besoin de comparer contre une exécution non bornée, **impossible précisément là où
le risque est maximal**. Motif « invariant plutôt que surveillance », reconduit.

**MONKEYPATCH `_T_END_RELAX` — ENDOSSÉ, et la franchise est la bonne réponse.** Seul
point d'entrée du `t_end` ; remplacé le temps de l'appel, restauré en `try/finally`
(testé sur exception). `git diff` vide, sortie prouvée préfixe ⇒ **la vérité-sol n'est
pas altérée**. **Mutation globale, NON THREAD-SAFE — T2 est mono-fil**, gravé tel quel :
invisible, ce serait pire qu'une modification franche.

**4. SONDE PERSISTÉE** (leçon du FAIL cloud non persisté, appliquée) : le SIGKILL ne
déroule aucun `finally` ⇒ début écrit **avant** le travail, battements portant le pic
**pendant**, fin avec exception ; JSON lines + `fsync`, relecture tolérant la dernière
ligne coupée ; pic par bras et par phase. **Un run partiel qui atteint le pic le
reporte.**

**5. RISQUE NEUF NOMMÉ (session critique) — L'EFFONDREMENT DU `dt` DANS LA QUEUE DE
RELAXATION.** Au barreau 64 : les **601 pas retenus couvrent t = 48.44** ⇒ `dt` moyen
**0.081**, **exactement la fourchette de §A26** (0.049–0.112) — **la dérivation qui
fonde M = 1 CONCORDE et n'est PAS invalidée**. Mais les 2 572 pas suivants couvrent
t = 15.56 (`dt` ≈ **0.006**), et atteindre 130 demande ~254 000 pas (`dt` ≈ **5e-4**) :
**le `dt` s'effondre de DEUX ORDRES DE GRANDEUR dans la queue, là où les cellules
S'ASSÈCHENT.**
> **RISQUE GRAVÉ, NON ARMÉ** : le substrat POSSÈDE des régimes où `dt_CFL` tombe très
> en dessous du temps de frame, et **l'assèchement est omniprésent dans un jeu**. La
> marge de 4–12× de T1 a été mesurée sur un **état synthétisé qui n'exerçait pas ce
> régime**. **Falsificateur désigné** : mesurer la marge CFL sur un état qui SÈCHE.
> **Non armé — T2 reste la priorité** (le gate (iii) est à un run de sa lecture).

**524 tests verts. DÉCISION ROMAIN : correction ENDOSSÉE avec la garde structurelle ;
cellule §A15 INTACTE — c'est l'INSTRUMENT qui a plié, pas la mesure.**

## §A32 — VERDICT MORT-b PRONONCÉ (2026-07-19) ; l'amplitude est diagnostique ; décomposition spatiale ordonnée

Run : fluide-reduit 7e25009, natif, après correction d'instrument endossée (§A31-diagnostic).
**Garde structurelle livrée et plus forte que la sonde qu'elle remplace** (testé sur
nombres mesurés : à N_settle=10 / t_end=2.0, l'ancienne acceptait — 11 pas disponibles —
là où la garde REJETTE, fenêtre à 1.9331 contre 2.0). Espion déléguant à
`_relax_episode` sans la modifier (un appel par barreau au lieu de deux) ; **aucune garde
au plafond gravé** — refuser là où l'original accepte créerait une divergence de domaine.
RLIMIT_AS armé **après** l'init CUDA (qui réserve 6.43 Go d'espace d'adressage pour
0.28 Go résident) ⇒ **plafond réglé sur VmSize, jamais sur RSS**, effectif 12.6 Go.
Aucun OOM ; rederive 1.25–1.61 Go, le plus haut des trois bras (cohérent avec le poste
diagnostiqué, et il tourne EN PREMIER donc pas un artefact de high-water).

**LECTURE MÉCANIQUE — pin sévère 0.0733 :**

| seed | production | témoin | branche |
|---|---|---|---|
| 101 | **0.70329** | **0.07731** | les deux traversent ⇒ (a) |
| 102 | **0.70266** | 0.01481 | production seule ⇒ (b)+(c) |
| 103 | **0.86958** | 0.05240 | production seule ⇒ (b)+(c) |

**VERDICT PRONONCÉ (Romain, 2026-07-19) : MORT-b.** Les trois seeds traversent en
production **d'un ordre de grandeur** (0.70–0.87 contre 0.0733) — **pas une marge
manquée, un écart massif**, et **toutes les émissions traversent**, pas seulement le
maximum. **Option A est MORTE par-seed.** Règle d'agrégation gravée appliquée : un seul
seed en (b)+(c) suffit ⇒ **Option B N'EST PAS LE REMÈDE.**

**LA RÉSERVE SUR LE TÉMOIN ÉTAIT LOAD-BEARING, ET LA MESURE LA VALIDE.** §A31-build
prédisait la branche « les deux traversent » très improbable au vu des **6e-5** ; le
témoin donne en réalité **0.0148 à 0.0773** — trois ordres au-dessus, assez pour faire
traverser le seed 101. **L'instrument n'était effectivement pas le même** : 6e-5 était
un écart d'ÉTAT, Δχ est une mesure SPECTRALE. Sans cette réserve, la branche 101 aurait
été lue comme une anomalie.

**DEUX FAITS REMONTÉS SANS LISSAGE (Claude Code)** : (i) le seed 101 traverse de **5.5 %
au-dessus du pin** (0.07731 vs 0.0733), **sur une seule émission (la 20)** — sa branche
est MARGINALE ; c'est le seul seed accusant (a), et il tient à un cheveu. Le verdict
agrégé n'en dépend pas (102 et 103 suffisent), **mais l'attribution de 101 est fragile** ;
(ii) le témoin est **NON MONOTONE** le long de l'histoire (101 : 0.026 → 0.003 → … →
0.077 → 0.034) — **l'écart f32 ne s'accumule pas, il FLUCTUE.**

**L'AMPLITUDE EST DIAGNOSTIQUE (session critique) — et le risque de raisonnement motivé
est nommé d'abord** : le verdict est mauvais et la session critique produit une raison
pour laquelle il pourrait ne pas dire ce qu'il dit. **Ce qui l'autorise, et rien de plus** :
ni l'observable ni le critère ne sont touchés (tous deux pré-enregistrés, le choix le
plus sévère ayant été endossé par la session critique elle-même) ; ce qui est demandé est
un **DIAGNOSTIC SUR DONNÉES DÉJÀ ACQUISES**, pas une re-mesure sous critère plus aimable.

- **FAIT DÉCLENCHEUR : 0.70–0.87 tombe DANS LA PLAGE DU CONTRÔLE DE CORRUPTION
  DÉLIBÉRÉE de la manche 2** — shuf-commit, Δχ₂ = **0.693 / 0.752 / 1.028**
  (§A13-résultat). Quand une mesure atterrit dans la plage de son propre
  contrôle-scramble, la lecture honnête est **« suspecter qu'on mesure une DESTRUCTION,
  pas une dégradation »**.
- **MÉCANISME STRUCTUREL DISPONIBLE** : la fovéa fait 32² sur 64² ⇒ **25 % de l'aire
  fine, 75 % décimée 2×** ; et l'observable lit **`max_carrier`**, des BANDES PORTEUSES
  — qu'une décimation 2× **annihile** dans toute cette zone. Saturation quasi-tautologique.
- **LE TÉMOIN DIT LA MÊME CHOSE PAR L'AUTRE BOUT** : 6e-5 d'écart d'état ⇒ jusqu'à 0.077
  de Δχ, le f32 seul FRÔLE le pin. Si la simple précision est perceptuellement à la
  limite sur ce substrat, ce n'est pas le f32 qui est en cause mais **la
  COMMENSURABILITÉ de Δχ avec le pin DANS CETTE CONFIGURATION**.
- **LE POINT DE SPEC, LE PLUS LOURD** : appliquer le pin **FOVÉAL** (jnd_sev, mesuré en
  ABX au centre du regard) **UNIFORMÉMENT** à un champ **délibérément grossier en
  périphérie** est **précisément la transposition que `r_fovea` devait résoudre** — et
  `r_fovea` est **GATÉ et NON MESURÉ** depuis l'origine (§7, « LE pin manquant
  load-bearing »). **Une architecture fovéale pourrait être STRUCTURELLEMENT incapable de
  passer un critère JND-fovéal uniforme, et ce ne serait pas un échec d'Option A.**

**DÉCISION ROMAIN : DÉCOMPOSITION SPATIALE, SUR LES CHAMPS DÉJÀ CAPTURÉS, AUCUN RUN** —
Δχ **fovéa seule** contre vérité vs Δχ **grossier seul** contre vérité. **Lectures
pré-écrites** : fovéa propre + grossier saturé ⇒ la cause est **(c) la DÉCIMATION**, et
le remède n'est **ni Option B ni EPS** — c'est **le critère appliqué à la périphérie**
qui est en cause, donc **`r_fovea`** ; fovéa également dégradée ⇒ la cause est **(b) EPS
et/ou le canal**, et EPS se resserre. **Rien n'est décidé avant cette lecture** — en
particulier, resserrer EPS maintenant paierait du trafic pour rien si (c) domine :
0.70 ne descendra pas sous 0.0733 en resserrant un seuil qui gouverne le détail
transporté, **pas la résolution du grossier**.

**L'observation §A31-diagnostic sur l'effondrement du `dt` reste portée, non armée.**

### §A32-décomposition — MON HYPOTHÈSE EST FALSIFIÉE ; aucun remède unilatéral ; bras EPS = 0 ordonné (2026-07-19)

Lecture : fluide-reduit `claude/lectures/mb_t2_decomposition.lecture.json`. Aucun run neuf.

**LA MÉTHODE, MEILLEURE QUE LA DEMANDE.** La restriction spatiale demandée par la
session critique était **ILLÉGITIME**, deux fois : (i) le complément n'est pas
rectangulaire (couronne en L), `chi_bands` fait une `fft2` et `_radial_bins` suppose
H = W — zéro-remplir changerait `mean_a`, qui divise χ ; (ii) même la fovéa seule
**change de base** (à 32², la bande (16, ∞) s'arrête au Nyquist 22.6 au lieu de 45.25)
⇒ on comparerait **deux instruments** — le piège §A14 exactement. **Substitut retenu :
MASQUER LA DIFFÉRENCE, PAS LE DOMAINE** — champ hybride (production dans la région,
rederive ailleurs) sur 64² plein, observable appelé tel quel, **référence identique donc
porteuses identiques** dans les trois évaluations ; masque plein ⇒ retour **au bit près**
sur le Δχ mesuré (testé). Réserves : champs non persistés par le driver (scalaires
seuls) ⇒ **re-matérialisés avec preuve d'identité bit-exacte sur les 18 Δχ publiés**,
fail-loud sinon ; le masquage **fuit au bord**, donc les Δχ partiels ne s'additionnent
pas — la partition d'énergie, exactement additive, sert de **second regard**.

**RÉSULTAT — AUCUNE DES DEUX LECTURES PRÉ-ÉCRITES NE S'APPLIQUE :**

| région | min | max | > pin |
|---|---|---|---|
| fovéa seule | 0.109 | 0.572 | **18/18** |
| périphérie seule | 0.148 | 0.678 | **18/18** |

L'hybride étant le **contrefactuel d'une région parfaitement réparée** : réparer
parfaitement la périphérie (`r_fovea`) laisserait Δχ à **0.109–0.572** ; réparer
parfaitement la fovéa (EPS) le laisserait à **0.148–0.678**. **AUCUN REMÈDE UNILATÉRAL
NE RACHÈTE LE GATE (iii).** (Approximation nommée : réparer une région changerait
l'autre par le couplage du halo.)

**L'HYPOTHÈSE DE LA SESSION CRITIQUE EST FALSIFIÉE — concession immédiate et sans
réserve :**
- **« On mesure une destruction, pas une dégradation » : FAUX.** Corrélation
  production/rederive **0.576–0.927** contre **|r| < 0.2** pour un vrai shuffle
  (discriminant testé). **La production est structurellement corrélée à la vérité**,
  avec une erreur d'amplitude et de grande échelle — **rien à voir avec du bruit**.
  **Le verdict dit bien ce qu'il dit : ce n'est PAS un artefact de corruption.**
- **Le mécanisme supposé est FAUX aussi, plus durement** : **la bande 16-31 N'EST PAS
  PORTEUSE** (`porteuses['16-31'] = False`) ⇒ la décimation **ne peut pas** saturer
  `max_carrier` en annihilant des porteuses fines : **elles n'y sont pas**. Bandes
  dominantes : **2-3 et 4-7** — écart de **GRANDE ÉCHELLE**, pas perte de texture fine.
- **Ce qui SURVIT de l'intuition** : l'asymétrie spatiale est réelle — la périphérie
  porte **22× plus d'énergie d'erreur par cellule** (médian, **jusqu'à 169×**). La
  périphérie est bien le lieu ; le mécanisme n'est pas celui qui avait été nommé.
- Le risque de raisonnement motivé avait été nommé d'avance (§A32) ; **la mesure établit
  qu'il était réel et que la session critique y était.** Consigné comme tel.

**CE QUE LA DÉCOMPOSITION SPATIALE NE POUVAIT PAS VOIR (session critique, après
falsification)** : **les deux régions passent par LE MÊME CANAL** — la connaissance du
CPU sur la fovéa vient des coefficients L3 seuillés, celle sur la périphérie aussi, plus
la décimation. **Une décomposition SPATIALE ne peut pas séparer un défaut DE CANAL : il
est partout.** Et un écart de **grande échelle, corrélé à la vérité, d'amplitude
fausse**, est la signature d'une **BANDE MORTE** — un seuil `|d| ≥ EPS` qui ne transmet
jamais ce qui reste sous lui. Si **EPS = 1e-2 est comparable à l'amplitude du champ de
sédiment réel**, le CPU n'apprend presque rien ; et §A23 avait établi l'innocuité d'EPS
**sur un état SYNTHÉTISÉ à l'échelle V4**, pas sur ce substrat. **C'est exactement la
conséquence rétroactive nommée en §A31-assemblage AVANT le run.**

**DÉCISION ROMAIN : BRAS EPS = 0 — LE PLANCHER STRUCTUREL.** Le discriminateur qui
manque n'est pas spatial, il est **PAR CANAL** : même structure 2-niveaux, budget large,
seuil nul (machinerie déjà testée : « seuil nul ⇒ CPU exact »). **Lectures pré-écrites :**
> **il PASSE ⇒ la cause est la BANDE MORTE du seuil — le remède est EPS** (et l'échelle
> à laquelle §A23 l'avait validé est à re-poser) ;
> **il ÉCHOUE ⇒ le plancher est STRUCTUREL** (décimation + halo) — **l'architecture
> fovéale, telle que construite, ne peut pas satisfaire É2**, et c'est une question de
> spec, pas de réglage.
Coût : un run de ~3.5 min. **Le verdict MORT-b n'est pas rouvert** — ce bras attribue,
il ne réhabilite pas.

## §A33 — PLANCHER STRUCTUREL ÉTABLI : É2 n'est pas satisfaisable par le vivant fovéal tel que construit (2026-07-19)

Bras EPS = 0 : fluide-reduit 4295353, `claude/lectures/mb_t2_eps0.lecture.json`.

**LA PRISE AVANT LE RUN, DÉCISIVE.** À EPS = 0, **le cap 10 % mordait à la place du
seuil** — `|d| ≥ 0` vrai partout ⇒ 4096 candidats pour `budget_k_fen(4096) = 409`
places. **Ce bras aurait mesuré LE CAP, pas la structure** : un plancher qui n'en est
pas, qui n'aurait rien tranché. Budget forcé à H·W, transparence vérifiée par test.
C'est exactement la question posée avant de courir, et elle a payé.
Trois portées nommées : le canal ici est **le modèle numpy (option 1), pas le compacteur
GPU L3** — sa capacité de buffer n'est pas exercée ; transparence à EPS = 0 à l'arrondi
f32 près (**~1e-8 relatif, mesuré — sept ordres sous le pin**) ; remontée à chaque
épisode et émissions en fin d'épisode ⇒ **aucune péremption ne se déguise en plancher**.

**RÉSULTAT — BRANCHE « IL ÉCHOUE » : 3/3 seeds, 17/18 émissions au-dessus du pin.**
À seuil nul et budget plein, **le canal ne retire déjà plus rien** ⇒ **resserrer EPS ne
peut pas franchir ce plancher.**
> **LE PLANCHER EST STRUCTUREL (décimation + halo). L'ARCHITECTURE FOVÉALE TELLE QUE
> CONSTRUITE NE PEUT PAS SATISFAIRE É2 — question de SPEC, pas de réglage.**
Part de structure (c) = EPS0 − témoin : **positive partout, 0.056 à 0.664** — terme
dominant devant le témoin (0.015–0.077).

**DEUX AUTO-CORRECTIONS DE CLAUDE CODE, toutes deux DURCISSANT la lecture :**
1. **Attribuer par différence de MAXIMA était faux** — les maxima des trois bras tombent
   à des **émissions différentes** (seed 101 : EPS0 culmine à l'émission 12, la
   production à la 24) ; les soustraire **mélangeait des instants**. Attribution refaite
   **par émission**.
2. **L'EMBOÎTEMENT (a) ⊂ (a)+(c) ⊂ (a)+(b)+(c) N'EST PAS MONOTONE.** Sur **3/18
   émissions, retirer le seuil AGGRAVE Δχ** (seed 101/12 : EPS0 = 0.668 contre
   production = 0.483). **Fait neuf** : le seuil ne fait pas que perdre de l'information
   — **il filtre aussi du désaccord de faible amplitude**, et Δχ étant un **rapport
   spectral**, tout laisser passer peut **injecter du désaccord**. « Part du canal » est
   un **écart SIGNÉ, pas une contribution additive** — les trois bras ne se lisent pas
   comme une décomposition additive propre. Sans effet sur le verdict (EPS0 traverse
   largement), mais interdit une lecture qui l'aurait suggéré.

**MÉCANISME NOMMÉ (hypothèse de la session critique, pas un fait)** : à EPS = 0
l'intérieur de la fovéa est alimenté **EXACTEMENT** ; son erreur résiduelle
(0.109–0.572, 18/18) ne peut donc venir que de son **BORD**, c'est-à-dire du halo dérivé
du grossier. **L'ERREUR DU LOINTAIN NE RESTE PAS DANS LE LOINTAIN : ELLE ENTRE DANS LA
FOVÉA PAR LE HALO.** Conséquence immédiate : **`r_fovea` ne rachèterait rien** — un JND
périphérique généreux ne protège pas contre une contamination qui **traverse**.

**CAVEAT D'ÉCHELLE, nommé AVEC son apparence de sauvetage.** Sur un épisode,
l'information traverse **`c·t = 1.886 × 48.44 ≈ 91 cellules`** ; **la fovéa fait 32**.
Donc **à 64² le halo contamine la fovéa ENTIÈRE**, tandis qu'à l'échelle V4 (fovéa 512²)
il n'atteindrait que **la couronne**. **La configuration 64²/32² est le PIRE CAS possible
pour ce défaut.** **Cela ne rachète PAS le verdict** — le gate a été re-scopé à 64²
précisément pour cela (§A31-build) — mais c'est une question que la séance papier doit
peser, **avec sa contre-épreuve**, et non un motif d'action.

**PORTÉE DU BRAS** : il **ATTRIBUE, il ne RÉHABILITE PAS**. **MORT-b n'est pas rouvert,
Option A reste morte**, la production reste ce qu'elle est.

**DÉCISION ROMAIN : SÉANCE PAPIER SUR É2 ET L'ARCHITECTURE — pas de quatrième
diagnostic** (engagement pris au tour précédent, tenu). Trois questions que la mesure
rend inévitables :
1. **É2 doit-il comparer des ÉTATS ou des PROJECTIONS ?** S'il compare des projections,
   **il est NON MESURABLE tant que la moitié projection n'existe pas** (§A20) — et le
   projet devrait le dire au lieu de mesurer un proxy d'état.
2. **La contamination du halo est-elle un DÉFAUT RÉPARABLE** (recouvrement plus profond,
   fovéa portant son propre bord, bord commis au registre) **ou INHÉRENTE** au
   couplage grossier→fin ?
3. **Le caveat d'échelle** (91 cellules de propagation contre 32 de fovéa) — et **quelle
   contre-épreuve** le rendrait falsifiable sans devenir un quatrième diagnostic
   déguisé.

## §A33-CORRECTION (2026-07-25) — SUSPENSION des attributions de §A32, §A32-décomposition et §A33 : un défaut d'instrument les précède

> **Entrée de correction append-only (tradition §A19-CORRECTION). Rien n'est effacé :
> les FAITS mesurés restent lisibles tels quels ; seules les ATTRIBUTIONS sont marquées
> EN SUSPENS.** Décisions Romain du 2026-07-25 : **D19-a** test unitaire AUTORISÉ ;
> **D19-b** suspension GRAVÉE MAINTENANT ; lectures décisionnelles D19-c/d **volontairement
> NON pré-enregistrées** — Romain décide après le résultat (dérogation nommée à la règle
> des lectures pré-écrites, portée par lui, consignée ici).

**ORIGINE.** Séance É2 (`pocPhysicator/claude/seance-e2-2026-07-24.md`), §0 : soupçon
que le niveau GROSSIER de T2 est avancé au Δx du fin. Vérification demandée par Romain,
**CONFIRMÉE PAR LECTURE DE CODE** le 2026-07-25
(`pocPhysicator/claude/verif-echelle-grossier-2026-07-25.md`, tout re-vérifiable au
grep) : (1) la référence `_rhs_o2` (`solver_wetdry.py`) **divise par Δx** ; le kernel
`_SOURCE_FIDELE` qui déclare la porter **ne reçoit aucune longueur** (ses `dx`/`dy` sont
des décalages de stencil ±1) — portage exact **uniquement à Δx = 1** ; (2)
`reduction_cfl_fidele` calcule le dt du grossier **à Δx = 1** aussi ; le « choix 3 : le
grossier à dx = 2 » de `production_fidele` existe **en prose, nulle part en code** ; (3)
`grep _rhs_o2 tests/` ne retournait **RIEN** — le portage n'avait JAMAIS été comparé à sa
référence ; tous les tests du chemin sont des propriétés d'équilibre (flux nuls) ou
d'identité, **aveugles par construction** à un facteur multiplicatif sur `L`.
**Conséquence si la lecture tient** : `L_grossier` = 2× le correct, dt partagé en
lockstep ⇒ le grossier avance à ~2× la vitesse physique du fin. Le substrat relaxant,
**aucune amplitude n'est prédite ; rien ici n'annonce un PASS** (le témoin frôle déjà le
pin à 0.077).

**CE QUI EST SUSPENDU (attributions seulement)** :
- §A32 : « **MORT-b** » et « **Option A est morte** » — le verdict se prononce sur le
  bras de PRODUCTION, qui EST la structure à deux niveaux ; son Δχ est porté par un
  grossier possiblement à mauvaise échelle. Par la gravure du projet — *« un échec
  d'instrument n'est pas un résultat »*, *« corriger l'INSTRUMENT, jamais la mesure »*
  (§A31) — l'attribution est suspendue, comme l'OOM avait suspendu le gate (iii) au lieu
  de le rendre.
- §A32-décomposition : l'INTERPRÉTATION causale de l'écart (bande morte éliminée ⇒
  structure par élimination) — un grossier à mauvaise échelle est un candidat qui
  n'avait jamais été mis dans la liste des éliminés.
- §A33 : « **LE PLANCHER EST STRUCTUREL** » et « **l'architecture fovéale telle que
  construite ne peut pas satisfaire É2** » — l'attribution à la STRUCTURE est suspendue ;
  un instrument à mauvaise échelle occupe exactement la même case.
- La phrase « **MORT-b n'est pas rouvert** » (§A32-décomposition, §A33) est **RETIRÉE** —
  concession de la session critique, consignée en §5 de la vérification : elle était
  incompatible avec le soupçon de §0 et la formulation confortable avait été gardée.

**CE QUI TIENT, inchangé** : tous les CHIFFRES (production 0.70–0.87 ; décomposition
fovéa 0.109–0.572 / périphérie 0.148–0.678 ; EPS0 0.056–0.664 ; témoin 0.0015–0.077) ;
**le bras témoin est PROPRE** (mono-niveau, Δx = 1, correct par construction) et avec lui
« **Option B n'est pas le remède** » ; **le lieu** (la périphérie porte 22×, jusqu'à
169×, d'énergie d'erreur par cellule — exactement là où le défaut vivrait) ; **la
méthode** (masquer la DIFFÉRENCE, jamais le domaine) ; la prise cap-à-EPS0 ; les deux
auto-corrections (attribution par émission, non-monotonie 3/18). INTACT car sans
grossier fidèle : le pin, §A13/§A14, le modèle de coût F1, M-a-quater/V4, T1 (ancre
0.924), le diagnostic OOM, la propriété E, le mipmap, L3.

**LE FALSIFICATEUR (D19-a, autorisé)** : `pocPhysicator/tests/test_portage_rhs_o2.py` —
le test unitaire manquant depuis C1, PAS un quatrième diagnostic (un pas, aucun chrono,
aucun épisode ; utile quel que soit son résultat). Prédictions factuelles pré-écrites
(verif §6) : lecture tient ⇒ `L_gpu ≡ L_ref(Δx=1)` et `L_ref(Δx=1) = 2·L_ref(Δx=2)`
exactement ; fausse ⇒ `L_gpu ≡ L_ref(Δx=2)`. Les deux branches sont décidables ; le test
n'échoue que si AUCUNE ne tient (troisième fait, à remonter tel quel). Exécution GPU sur
iluin-tworings3 (`pytest -s tests/test_portage_rhs_o2.py`) ; la branche CPU (échelle de
la référence) tourne partout.

**POINT D'ARRÊT OBLIGATOIRE** : le résultat se remonte à Romain ; D19-c/d se décident
sur lui ; AUCUN enchaînement. Tant que le test n'a pas tourné, le gate (iii) se dit :
« l'architecture fovéale telle que construite ne reproduit pas la vérité pleine
résolution sous le pin d'instrument ; que ce plancher soit STRUCTUREL reste à confirmer
(D19) ; qu'il constitue un échec du contrat É2 est l'objet de D14/D15, non tranché. »

### §A33-CORRECTION-résultat (2026-07-25) — le falsificateur a tranché : CONFIRMÉ

**Exécution** : iluin-tworings3, terminal natif,
`.venv/bin/python -m pytest -s tests/test_portage_rhs_o2.py` (pocPhysicator) — 2 passed.
- CPU : `L_ref(Δx=1) = 2·L_ref(Δx=2)` **EXACTEMENT** (`array_equal`), champ à flux non
  nuls (la garde anti-aveuglement §3 a mordu : max|L| > 1e-3 vérifié).
- GPU : **err(Δx=1) = 3.297e-05** (bruit f32) ; **err(Δx=2) = 5.000e-01** (la signature
  exacte du facteur 2 — quatre ordres de grandeur entre les deux branches).

> **Lecture (pré-écrite, prononcée telle quelle)** : `L_gpu ≡ L_ref(Δx=1)` → **le
> grossier de T2 (Δx=2) tourne à MAUVAISE ÉCHELLE. Le soupçon de §0 de la séance É2 est
> CONFIRMÉ PAR EXÉCUTION.** La suspension des attributions de §A32, §A32-décomposition
> et §A33 passe de « en attente du test » à **fondée sur un fait exécuté** : le défaut
> d'instrument est établi, l'attribution « STRUCTUREL » ne peut pas être revendiquée en
> l'état. Le témoin (mono-niveau, Δx=1) reste propre ; « Option B n'est pas le remède »
> et le lieu (périphérie 22×–169×) restent acquis.

**DÉCISIONS ROMAIN (2026-07-25, prises APRÈS résultat — c/d volontairement non
pré-enregistrées, dérogation consignée en §A33-CORRECTION)** :
- **D19-c ENDOSSÉ** : correctif = **kernel Δx-conscient** (paramètre de maille dans la
  signature, comme `_rhs_o2`) ; l'empreinte du CUDA se re-grave via son verrou, avec
  provenance datée ; `reduction_cfl_fidele` reste à Δx=1 (CONSERVATRICE pour le
  grossier — dt plus petit que sa limite, jamais faux) ; **jamais un ajustement de dt,
  jamais la mesure, aucun seuil, aucune bande**.
- **T2 se re-court INCHANGÉ par ailleurs** (mêmes seuils, mêmes seeds, mêmes bandes,
  cellule §A15 intacte), verdict-grade sur iluin-tworings3, APRÈS revue du correctif et
  suite de tests verte.
- **Implémentation : Claude Code, séance dédiée** — ordre de mission :
  `pocPhysicator/claude/mission-correctif-dx-2026-07-25.md`.

**POINT D'ARRÊT** : tests verts → remonter → **Romain lance lui-même le re-run T2** ;
son verdict remonte avant toute suite — D14, D15, D16, D17, D18 se reposent dessus.

### §A33-CORRECTION-exécution (2026-07-25) — correctif appliqué ; portée du re-run T2 gravée AVANT le run

**Mission exécutée** (Claude Code, ordre `pocPhysicator/claude/mission-correctif-dx-2026-07-25.md`,
commit 26d53e2) : kernel Δx-conscient (`inv_dx` dans la signature, grossier à
Δx = DECIMATION aux quatre appels d'étage), `reduction_cfl_fidele` INTOUCHÉE, aucun dt
ajusté, aucun seuil, aucune bande ; `_SOURCE`/L3/EXNER : diff vide vérifié.
**Empreinte re-gravée** — le verrou a parlé, c'est son rôle : 6dd207ca… (9248) →
**aef7237d… (9646)**, onze citations à jour. **1124 tests verts.** Le portage est
désormais GARDÉ en permanence par trois tests : inv_dx = 1.0 → err(Δx=1) = 3.297e-05
(le chiffre exact d'avant correctif : le régime Δx = 1 n'a rien vu passer) ;
inv_dx = 0.5 → err(Δx=2) = 6.331e-05. **Non-régression MESURÉE, pas argumentée** :
ancien kernel reconstruit en mémoire, écart max **0.000e+00 bit à bit** sur champ non
trivial — le témoin de T2, l'ancre T1 et la vérification de régime C4 portent sur le
même objet numérique. (Vérification gardée HORS suite, délibérément : un test qui
reconstruit une source re-verrouillerait ce que la mission vient de corriger.)

**DETTE NOMMÉE — chrono T1 (décision Romain 2026-07-25 : DIFFÉRÉ)** : bit-exact n'est
pas iso-coût (+1 multiplication/cellule/étage) ; l'ancre 0.924 et le coût 11.696 ms
restent **datés sur 6dd207ca…**, et le rapport T1 le dit désormais explicitement.
**Condition de réveil** : re-chrono DÛ avant toute décision qui consommerait le coût du
fidèle à la marge. Aucune décision pendante ne le consomme (V4/gate (i) tourne sur le
PROXY `_SOURCE`, intouché ; le re-run T2 mesure du Δχ, pas du temps).

**PORTÉE DU RE-RUN T2, pré-écrite AVANT toute donnée** : le grossier corrigé évolue
différemment à chaque pas ⇒ `reduction_cfl_fidele(q_c)` rend d'autres valeurs ⇒ la
suite des dt change ⇒ un épisode couvre un AUTRE temps physique. Mécanisme
identifiable ; **direction et amplitude NON prédites**. Conséquences de lecture,
gravées maintenant :
1. **un gros écart de Δχ vs l'ancien run n'est PAS suspect a priori** — c'est le
   comportement attendu d'un instrument réparé ;
2. **la seule comparaison légitime est contre la vérité f64 pleine résolution** —
   jamais contre le run contaminé ;
3. le verdict se lit au **combinateur pré-enregistré de §A15/T2, inchangé** — mêmes
   seuils, mêmes seeds, mêmes bandes, cellule intacte ;
4. **rien n'annonce un PASS** — le témoin propre frôle le pin à 0.077 ; un échec
   confirmé sur instrument juste serait PLUS solide qu'avant, pas moins.

Documents datés (`diagnostic-oom-t2.md`, prereg-*) : **NON réécrits** — ce sont des
relevés d'époque ; la trace de la re-gravure vit ici, dans le verrou et dans le commit.

**Le re-run T2 est LANCÉ PAR ROMAIN, verdict-grade sur iluin-tworings3 ; son verdict
remonte avant toute suite (D14–D18).**

## §A34 — RE-RUN T2 SUR INSTRUMENT VALIDÉ (2026-07-25) : MORT-b RE-PRONONCÉ, PLUS LARGE QU'AVANT

Re-run décidé en §A33-CORRECTION (D19-c), instrument réparé (kernel Δx-conscient,
empreinte aef7237d…, portage désormais GARDÉ par trois tests contre `_rhs_o2`),
protocole INCHANGÉ : mêmes seuils, mêmes seeds, mêmes bandes, cellule §A15 intacte.
Empreinte de config **9dd95077aa3baf40** — le changement d'empreinte a invalidé le
report contaminé EN BLOC (aucun mélange de configs possible) ; l'ancien run est archivé
(`outputs/f1/mb_t2_avant-correctif-dx.json`) ; lecture versionnée :
`pocPhysicator/claude/lectures/mb_t2_rerun_dx.lecture.json`. Gate d'instrument
(`gel_bit_exact`) exigé et passé avant mesure ; iluin-tworings3, terminal natif.

**RÉSULTAT (combinateur pré-enregistré, lecture mécanique, AUCUNE lecture à l'œil)** —
pin sévère 0.0733, seeds traversants : 101, 102, 103 :

| seed | production Δχ_max | témoin Δχ_max | branche par-seed |
|---|---|---|---|
| 101 | 1.46957 | **0.07731** | (a) — **le témoin traverse** |
| 102 | 2.09089 | 0.01481 | (b)+(c) |
| 103 | 2.41833 | 0.05240 | (b)+(c) |

> **VERDICT : MORT-b — Option B N'EST PAS le remède** (règle d'agrégation : 102 et 103
> en (b)+(c)).

**Les lectures pré-écrites de §A33-CORRECTION-exécution s'appliquent telles quelles** :
l'écart vs l'ancien run n'est PAS suspect (la suite des dt du grossier a changé, un
épisode couvre un autre temps physique) ; la seule comparaison légitime est contre la
vérité f64, et c'est celle que le combinateur a faite.

**CHIFFRES INCONFORTABLES, en évidence :**
1. **La production diverge PLUS qu'avant correctif** : 1.47–2.42 contre 0.70–0.87
   (20–33× le pin, contre 9.6–11.9×). L'instrument corrigé AGGRAVE l'écart mesuré —
   direction que personne n'avait prédite, et que personne n'avait le droit de prédire.
2. Les trois valeurs sont **AU-DESSUS de la plage du contrôle de corruption délibérée**
   de la manche 2 (shuf-commit 0.693–1.028) — l'ancien run, lui, tombait DANS la plage.
3. **LE TÉMOIN SEED 101 TRAVERSE : 0.07731 > 0.0733.** La réserve de §A31 (« le témoin
   frôle ») devient une traversée pour un seed sur trois — la cause (a) seule (f32)
   suffit à franchir le pin pour ce seed, lecture mécanique par-seed. Portée, sans
   déplacer aucun seuil : 0.07731 est dans l'IC du pin [0.0603, 0.0867] ; le
   combinateur utilise le point gravé et c'est lui qui fait foi — cette note est une
   note de portée, pas un sauvetage.

**CE QUE LE RE-RUN ÉTABLIT / N'ÉTABLIT PAS :**
- **ÉTABLI**, désormais sur un instrument confronté à sa référence : à 64², fovéation
  2-niveaux, la production diverge de la vérité f64 largement au-dessus du pin
  d'instrument, 3/3 seeds, toutes émissions confondues au max de série. **Le FAIT de
  §A32/§A33 est re-fondé, PLUS SOLIDE qu'avant.**
- **NON ÉTABLI** : les attributions fines restent SUSPENDUES (§A33-CORRECTION) — la
  part (b) vs (c), « le plancher est STRUCTUREL », « l'erreur entre dans la fovéa par
  le halo » : leurs diagnostics (décomposition spatiale, bras EPS = 0) ont tourné sur
  les champs contaminés et n'ont pas été refaits.

**DÉCISION ROMAIN (2026-07-25)** : PAS de re-attribution (décomposition/EPS = 0)
maintenant — règle anti-tapis-roulant : la décision ouverte est une décision de SPEC,
et aucune de D14–D18 ne dépend de b-vs-c ; re-diagnostic seulement si une décision
prise en dépend explicitement. **Prochain pas : les décisions D14–D18 de la séance É2,
paper grade, sur les chiffres du re-run.** Point d'arrêt à chaque décision.

## §A35 — SÉANCE É2 : DÉCISIONS D14–D18 (2026-07-25, sur les chiffres de §A34)

> Séance paper, aucun run. Décisions Romain, gravées avec leurs caveats. La part qui
> contredit §A33 (« gate (iii) MORT ») passe par **entrée de CORRECTION explicite**
> (tradition §A19-CORRECTION) — rien n'est effacé.

**Correction d'inventaire préalable (session critique)** : la fermeture de la branche
(γ) (« critère fovéa-conscient, fermé par la mesure 0.109–0.572 ») reposait sur la
décomposition des champs CONTAMINÉS — elle est **SUSPENDUE avec son bras**. (γ) a été
remise sur la table de D15 en toute connaissance, et n'a pas été retenue.

**(D14) ENDOSSÉE — SCISSION DE É2** : **É2-état** (readout d'instrument vs vérité f64 —
ce que le harnais mesure) / **É2-projection** (clause d'observateur — indistinguabilité
à travers le rendu, gatée sur son existence). La scission est un énoncé sur le TEXTE de
§A13-0, qui quantifie sur un observateur ET se déclare mesurable à une date où aucune
projection n'existait. **CAVEAT gravé, qui voyage avec** : l'atténuation de l'état des
lieux §1.2 — le rendu étant une fonction déterministe de `z`, le pin peut se transporter
PAR CONSTRUCTION ; si le transport est ≈ 1 et sans pondération d'excentricité,
É2-état ≡ É2-projection et la scission se referme d'elle-même. Faits pesés : pondération
spatiale plate sur une périphérie par-conception lointaine ; hypersensibilité
6e-5 → 0.077 ; **témoin 101 traversant (la f32 seule franchit le pin, §A34)**.

**(D15) ENDOSSÉE — α + SUCCESSEUR** :
- **Gate (iii) : ÉCHOUÉ sur É2-état, PRONONCÉ, définitif** — sur instrument validé
  contre sa référence (§A34), 3/3 seeds, 20–33× le pin, Option B n'est pas le remède.
  **La spec fovéa-z (SPEC-FOVEA-Z.md) NE FAIT PAS FOI en l'état.**
- **Gate (iii′) : NÉ — É2-projection**, non mesurable tant que la moitié projection
  n'existe pas ; la construction de la projection (image rendue d'abord, puis son)
  devient **LE CHEMIN CRITIQUE** de sa mesure. Ni fuite (l'échec est prononcé), ni
  surclame (la clause d'observateur précède le résultat).
- α sec, β, γ : consignés NON RETENUS.

**(D16) GRAVÉE — étiquette de portée sur §A34** : la plage production **1.47–2.42 est
mesurée en régime SATURÉ** (64²/32², fovéa entièrement traversée, Π ≫ 1). Le SENS de la
variation vers Π < 1 (V4) est connu ; **l'amplitude ne l'est pas et ne se transpose
JAMAIS**. L'ancienne assiette (0.056–0.664, bras EPS = 0) reste suspendue avec son bras.

**(D17) GRAVÉE — refus du profil radial entériné, et la RÈGLE en toutes lettres** :
*aucun diagnostic sans une décision nommée qui en dépend explicitement.* C'est elle —
et elle seule — qui rouvrirait une décomposition propre si (γ) redevenait vivante.

**(D18) ENDOSSÉE VERBATIM — la formule publique** (remplace « gate (iii) MORT ») :
> « Le gate (iii) est **ÉCHOUÉ sur É2-état** : à 64², sur un instrument validé contre
> sa référence, l'architecture fovéale telle que construite diverge de la vérité pleine
> résolution à 20–33× le pin d'instrument, 3/3 seeds ; Option B n'est pas le remède.
> L'attribution fine de cet écart (seuil vs structure) n'est pas re-mesurée. La clause
> d'observateur du contrat vit dans le **gate (iii′) — É2-projection — non mesurable
> tant que la projection n'existe pas**. La spec fovéa-z ne fait pas foi en l'état ; la
> question de spec ouverte est : **contre quoi la fovéa doit-elle être fidèle.** »

**CE QUE LA SÉANCE OUVRE (décisions neuves, AUCUNE prise ici)** : (1) la spec de la
MOITIÉ PROJECTION — image rendue d'abord, chemin critique du gate (iii′) ; son coût
n'est mesuré nulle part et s'ajoute à un budget V4 déjà « ~16.7 ms pour un rendu jamais
chiffré » ; (2) la question v2 : « contre quoi la fovéa doit-elle être fidèle » ;
(3) le sort des attributions suspendues (b vs c) : dormantes sous la règle D17.

**POINT D'ARRÊT : la séance est close. Aucun chantier n'est lancé sans décision neuve
de Romain.**

## §A36 — ARC PROJECTION OUVERT : P0 TRANCHÉE — LE CONTRAT DE FIDÉLITÉ DE LA FOVÉA (2026-07-25)

> Décisions Romain (P0-a..d), sur `pocPhysicator/claude/seance-fidelite-2026-07-25.md`
> nourrie par `note-orientation-v2-2026-07-25.md`. Paper grade, aucun run, aucun seuil.

**(P0-a) C-uni** (identité-sous-JND à la vérité f64, uniforme) : **MORT comme contrat
de la fovéa** — échoué par la mesure (§A34), fermé par l'arithmétique de ses remèdes —
**VIVANT comme étalon du harnais du registre** (le rederive f64 reste le juge de la
re-dérivabilité des commits ; REGISTRE_FERME y tient inchangé).

**(P0-b) LE CONTRAT DE LA FOVÉA : C-STRAT, VERSION F-UNIQUE.** Trois clauses :
1. **le COMMIS est contracté à l'identité-sous-JND** — tenu, mesuré, borné ;
2. **le VIVANT est contracté au TÉMOIGNAGE** — plausible, et non-contradictoire avec
   le commis et le perçu ; la vérité f64 n'est plus son référent ;
3. **le JUGE est la projection pondérée-observateur** (le gate (iii′), né en §A35) —
   jamais l'espace d'état.

Forme opératoire endossée (reformulation Romain) : **UN SEUL F**, scale-aware à deux
titres (Δx — réglé §A33-CORRECTION — et fermeture sous-maille, OUVERTE) ; **tirage de
naissance** des branches, conditionnel, semé, contraint par le ledger ; **projection**
pour tout ce qui n'est pas état. **Falsifieur unifié : AUCUNE TRANSITION N'EST
DÉTECTABLE PAR L'OBSERVATEUR** (naissance/élagage de branche, frontière
feuille/projection, instant de commit) — l'ABX de substitution. **Le caveat D14
voyage** : si P3 montre un transport ≈ 1 et une pondération plate, la clause 3 utilise
ce pin tel quel — le contrat ne s'affaiblit pas, son juge se durcit.

**(P0-c) P1 LANCÉ** : spec du rendu minimal d'instrumentation (albédo→luminance,
déterministe, calibration cycles/degré héritée d'Arc C), s'ouvrant sur **sa décision
de périmètre** — projeter la structure réelle (fovéa + grossier upsamplé) ou la pleine
résolution d'abord — coûts chiffrés au papier AVANT le choix.

**(P0-d) PRINCIPES v2 ENDOSSÉS, gravés par référence** (détail et étiquetage :
`note-orientation-v2-2026-07-25.md`) : conservation = squelette, plausibilité = chair
(la forme peut être phénoménologique, jamais le bilan) ; `z` = forêt d'arbres de
coefficients élagués (l'élagage est spatial, la politique de raffinement est la
politique de croissance) ; **τ_dec par champ** — la frontière éphémère/persistant
devient une grandeur mesurée (falsifieur nommé, dormant jusqu'à consommateur, règle
D17) ; stockage **snapshot + queue** (le snapshot matérialise, **la queue reste la
vérité** — le ledger porte les entrées ET les commits, et ne se vide jamais dans un
snapshot) ; météo à support grossier natif + canal causal mince vers le fin.
Propositions NON endossées, consignées séparément dans la note : file de priorité
JND-par-FLOP, quantification du commis par format, fovéa en pente.

**LES GATES DE L'ARC** : P0 ✓ (cette entrée) → **P1** spec du rendu-instrument
(papier) → **P2** chiffrage du rendu (première ancre de la moitié manquante du budget
V4) → **P3** transport du pin (falsifie l'atténuation D14 ; donne son échelle au gate
(iii′) et son consommateur à `r_fovea`). Chaque gate sur le précédent ; la règle D17
s'applique à tout l'arc.

**POINT D'ARRÊT : prochain livrable = la spec P1, paper grade, revue avant toute
construction. Aucun enchaînement.**

## §A37 — SPEC P1 ENDOSSÉE : le rendu-instrument R1, Option B ; trois gravures adjointes (2026-07-25)

> Revue Romain de `pocPhysicator/claude/spec-p1-rendu-instrument-2026-07-25.md`
> (paper grade, aucun run). Quatre décisions tranchées ; la spec passe à ENDOSSÉE
> telle qu'amendée. Rien ne se code avant l'ordre de mission (décision séparée).

**FAIT DE CODE GRAVÉ, fondement de la spec** : le pin jnd_sev 7.33 % a été mesuré
à travers `imshow(cmap="viridis", vmin=0, vmax=1)` (`run_arcC_session.py:107`) —
en PSEUDO-COULEUR, avec le rééchantillonnage par défaut d'imshow, sans gestion
d'EOTF. La dette §C4-3 (albédo→luminance) est concrète. Portée double, sans
sauvetage ni alarme : viridis est par conception monotone et quasi-linéaire en
clarté — le transport pourrait être ≈ 1 ; « pourrait » est ce que P3 mesure.

**DÉCISIONS ROMAIN (P1, 2026-07-25)** :
- **D-P1-1 : OPTION B** — pleine résolution d'abord, noyau R1 seul. Fondement D17 :
  les paires ABX de P3 sont mono-niveau par construction ; aucune décision nommée
  ne consomme le compositeur avant P3. Interface gravée périmètre-neutre (champ
  albédo pleine résolution → luminance calibrée) ; le compositeur (structure réelle
  fovéa+grossier, couture) = incrément post-P3, sa décision de couture prise le pin
  transporté en main. CONSÉQUENCE PRÉ-ÉCRITE sur P2 : P2 chiffrera le NOYAU, pas la
  composition — « première ancre de la moitié manquante », pas la moitié entière ;
  le coût de composition reste une dette nommée (à l'échelle V4, gather écran du
  même ordre que le shading [CALCUL]).
- **D-P1-3 : (a)** — chemin lumineux IDENTITÉ Y = A : une nouveauté à la fois, P3
  mesure le transport sur la seule variable changée (pseudo-couleur → luminance).
  Le lambertien (`relief_shaded`) = incrément R2 post-P3, avec sa propre mesure de
  transport s'il est acheté.
- **D-P1-4 : (a)** — inverse-EOTF sRGB gravé en formule dans R1. Linéarité SUPPOSÉE
  sRGB, jamais mesurée au photomètre, jamais surclamée : l'ABX exige le déterminisme
  et la stabilité (biais commun aux deux stimuli d'un essai), pas la linéarité
  absolue. Conditions de session gravées d'Arc C inchangées.
- **D-P1-2 : (a)** — rééchantillonnage BILINÉAIRE gravé comme étage de R1 (numpy,
  testé, déterministe) — plus jamais délégué à une option de bibliothèque
  d'affichage.

**Le contrat R1 en une ligne** : fonction PURE, sans RNG, sans état (plus dur que
§A20-3 : l'état éphémère de readout est interdit tout court dans l'instrument),
albédo [0,1] → sRGB quantifié uint8 affiché à `taille_domaine_px` (calibration §C7
héritée, VALEURS d'écran re-mesurées à chaque session). L'observable Δχ et le pin
sont INTOUCHÉS. Six falsifieurs d'acceptation nommés dans la spec (déterminisme
bit, stabilité z-stable⇒image-stable, test à blanc, cohérence inverse de
calibration, monotonie de chaîne, garde d'acuité) ; ils GATENT l'achat de P2.

**TROIS GRAVURES ADJOINTES (décisions Romain du 25/07)** :
1. **Les six pistes de la note d'orientation v2 (§7), gravées par référence** :
   (i) sauter au lieu de tourner, exception météo globale = ENABLER du saut ;
   (ii) le ledger est aussi l'interface d'auteur [ACTÉ] ; (iii) versionnage de F =
   clause de sauvegarde (dette de spec née du correctif Δx) ; (iv) le rembobinage
   est gratuit ; (v) checklist de plausibilité sans référence [post-P3] ;
   (vi) hystérésis d'élagage + statut des PNJ-témoins [à trancher avant v1.1].
   Aucune n'apporte de solution à P1 (vérifié piste à piste, spec §6) ; chaque
   falsifieur attend sa décision (règle D17).
2. **Lecture différée O-Voxel/TRELLIS.2** (séance dédiée, jamais gravée jusqu'ici) :
   EXCLU comme base de `z` (readout-space, pas state-space ; « field-free » est un
   anti-feature pour des lois de conservation volumiques ; codec de snapshots sans
   sémantique temporelle ni structure de commit) ; candidat couche
   readout/commit-surface de PRODUCTION (conversion →mesh < 100 ms) ; réexamen
   post-P3. Hors de R1 par construction.
3. **Règle d'hygiène des faits d'instrument** (née du résidu « ruff absent »,
   deux occurrences le même jour) : *un fait d'instrument porte sa date ; à la
   reprise, re-vérifier les moins chers (un `ls`, un `--version`) avant de les
   citer.* Ajoutée à la discipline du brief de reprise.

**POINT D'ARRÊT : prochaine étape = ordre de mission Claude Code
(`mission-p1-rendu-instrument.md`), sur décision explicite de Romain. RIEN ne se
code avant ; les tests d'acceptation se développent où l'on veut, le verdict P3
restera iluin-tworings3 natif.**

## §A38 — P1 CONSTRUIT ET CONFORME ; P2 ET P3 PRÉ-ENREGISTRÉS ET ENDOSSÉS (2026-07-25)

> Revue critique de la remontée mission P1, puis endossement Romain des deux
> pré-enregistrements (`prereg-p2-chiffrage-rendu.md`, `prereg-p3-transport-pin.md`,
> pocPhysicator). Aucune mesure dans cette entrée.

**P1 CONSTRUIT (mission exécutée, commit pocPhysicator 2bbcca2)** : `arcC_rendu.py`
pur, ordre d'étages amendé (linéaire → sRGB → uint8 unique), gardes fail-loud,
**empreinte-verrou c5ac8757… (15 770 octets)** ; sélecteur `--rendu` obligatoire,
provenance au sidecar (viridis à sha=None — la vérité du chemin historique) ;
**1168 tests verts (+44)**. Revue critique : **CONFORME, rien à refaire** —
modules protégés vérifiés à diff VIDE, empreinte recalculée exacte, tolérance de
porteuse 0.90 pré-enregistrée AVANT ses mesures (0.962–0.988, fréquence intacte).
Quatre déviations remontées et ACCEPTÉES : sidecar au lieu d'un en-tête JSONL
inexistant (abx intouchable) ; extension `run_arcC_orchestration.py` hors liste
(sans elle, une campagne humaine héritait d'un chemin d'affichage EN SILENCE —
la négation du flag) ; `CHEMINS_RENDU` en source unique ; falsifieur du
court-circuit ajouté. Deux « non faits » adressés par décision : cohérence
inverse étendue = couverte (identité bit-exacte + porteuse) ; garde d'acuité =
BLOQUANTE en pré-requis P3 (D17 : son consommateur est P3).

**FAIT D'INSTRUMENT CORRIGÉ (première application payante de la règle §A37-3)** :
`ruff` vit dans `pocCascade2phys/.venv/bin/ruff` (0.15.20), **ABSENT de
`pocPhysicator/.venv`**, aucune config aux dépôts — le « INSTALLÉ dans .venv » du
brief du matin était imprécis de DÉPÔT bien que « vérifié ». Brief corrigé
(43c7501). Lint effectif du projet : `--line-length 100 --select E,F,W`.

**P2 ENDOSSÉ** (`prereg-p2-chiffrage-rendu.md`) : deux cellules, chacune avec son
consommateur D17 — cellule 1 (R1 numpy, bande 0.05–5 ms) → l'intégrité des
timings du régime sévère, avec décision de pré-calcul des stimuli si dépassée ;
cellule 2 (encodage sRGB à l'échelle V4, GPU, bande 0.02–0.5 ms, build gaté
0.5 séance, équivalence numpy exigée à tolérance zéro sinon remonté chiffré) →
la ligne de budget « ~16.7 ms pour un rendu jamais chiffré ». **Lecture
inconfortable gravée d'avance** : si la cellule 2 rend « quasi gratuit », la
gravure dit « la dette se DÉPLACE vers la composition et l'optique, elle ne
rétrécit pas » — interdiction pré-écrite de conclure « le rendu tient ».

**P3 ENDOSSÉ** (`prereg-p3-transport-pin.md`) : protocole IDENTIQUE à la campagne
du pin (mêmes 20 sources, même escalier, mêmes gardes §C5, même IC, sévère seul),
**un seul changement : `--rendu r1`** ; **BRAS TÉMOIN viridis dans la même
session** avec garde de validité — hors de l'IC gravé [6.03, 8.67] ⇒ session
INDÉTERMINÉE, aucune lecture de transport (sans ce bras, un écart serait
inattribuable entre le chemin et une dérive sujet/écran). Trois lectures
pré-écrites, dont l'ANTI-SURCLAME : « compatible avec 1 » n'établit QUE la
condition transport du caveat D14 — la pondération d'excentricité n'est PAS
mesurée (stimulus fovéal ~2°), **la scission D14 ne se referme pas ici, quoi que
P3 rende** ; « transport ≠ 1 » donne au gate (iii′) son échelle et **AUCUN seuil
d'état ne bouge rétroactivement**. Pré-requis de build gatés : garde d'acuité
BLOQUANTE, liaison sidecar↔log par sha256, pré-calcul conditionnel.

**POINT D'ARRÊT : ordre de mission des pré-requis (`mission-prerequis-p2p3.md`) ;
les cellules P2 sont LANCÉES PAR ROMAIN, verdict-grade natif ; la session humaine
P3 vient APRÈS pré-requis verts, P2 couru, et décision explicite. Chaque verdict
remonte avant toute suite.**

### §A38-CORRECTION (2026-07-25) — le pré-requis « garde d'acuité bloquante » est FALSIFIÉ avant build ; remplacé par une garde de COMPARABILITÉ

> Entrée de correction append-only (tradition §A19/§A33). Remontée de la session
> Claude Code AVANT toute construction — la vérification la moins chère d'abord,
> appliquée à l'ordre de mission lui-même. Vérifiée indépendamment par la session
> critique (arithmétique + manifeste + docstring).

**LE FAIT** : à la géométrie pic-CSF, la taille angulaire d'une cellule ne dépend
pas de l'écran — ppd entre dans `taille_domaine_px` et en ressort :
cellule = (porteuse/c_deg)·60/64 = (5.5/3)·60/64 = **1.71875 arcmin**, contre un
seuil d'acuité de 1.0. `cellule ≥ seuil` est vrai PARTOUT (mesuré de 400 à
1500 mm : 1.713–1.724, seule la troncature entière de `taille_domaine_px` bouge le
chiffre). La garde gravée en §A38 aurait refusé toute session pic-CSF — **y
compris la campagne du 2026-07-05 dont sort le pin** (manifeste : ppd 42.099,
77 px, cellule réalisée 1.715 arcmin, ratio 1.71). Le seul mode passant
(`plafond`, ratio 0.9973 — de 0.27 %, par grâce d'arrondi entier) casserait le
protocole P3 (« un seul changement ») et la comparabilité du bras témoin à l'IC
gravé, mesuré à pic-CSF.

**CONCESSION CONSIGNÉE (session critique)** : la garde bloquante était SA
recommandation, endossée sur sa parole ; le fait était écrit au docstring de
`observation_cellule_pic_csf` (`arcC_calibration.py` l.117-120 : « pic-CSF est
DÉJÀ au-dessus (ou au) plafond d'acuité — tension attendue », « AUCUNE décision
n'est prise ici ») — un non-choix DÉLIBÉRÉ de §C7 pièce 3, converti en gate dur
sans re-dérivation, falsifiable à coût nul. Le seuil d'acuité n'est pas touché :
un seuil se copie, il ne s'ajuste pas.

**DÉCISION ROMAIN : (A) GARDE DE COMPARABILITÉ, bloquante** — sous
`--sujet humain`, la géométrie de session doit être `pic-csf` (celle du pin
gravé) ; `RuntimeError` d'aiguillage sinon. Le report §C7 reste un CHIFFRE
SURFACÉ au sidecar, jamais un booléen (le refus délibéré de la pièce 3 est
PRÉSERVÉ). Cette garde aurait passé la campagne du pin par identité ; elle bloque
le seul scénario dangereux — une session à une AUTRE géométrie, comparabilité
cassée en silence. Options (B) report seul (strictement dominée) et (C) P3 à
`plafond` (deux changements à la fois) : consignées NON RETENUES.

**CE QUI SURVIT de l'inquiétude d'acuité** : la tension est PORTÉE PAR LE PIN
lui-même (mesuré à ratio 1.71, blocs compris dans le stimulus vu) et SYMÉTRIQUE
entre les bras de P3 (même 64², même taille, deux chemins) — une propriété du
référent, pas un confondeur du transport.

**Chantiers 1 (runner P2) et 3 (sidecar↔log) : indépendants du point, LANCÉS tels
quels.** Le prereg P3 porte sa correction explicite ; le chantier 2 de
`mission-prerequis-p2p3.md` est ré-écrit sur le critère retenu.

### §A38-CORRECTION-2 (2026-07-25) — l'équivalence « tolérance zéro » de la cellule 2 est FALSIFIÉE STRUCTURELLEMENT ; critère de NATURE retenu

> Deuxième correction du jour née de la même règle — la vérification la moins
> chère d'abord, appliquée par la session Claude Code à son propre chantier.
> Mission pré-requis exécutée (pocPhysicator d875834) : **1202 tests verts**,
> empreinte R1 c5ac8757… verte (re-vérifiée par la session critique), garde de
> comparabilité en source unique importée par la campagne, bandes verrouillées
> par test, champ `verdict_grade` qui dénonce les runs raccourcis.

**LE FAIT** : l'équivalence f32/f64 de la sonde CuPy (cellule 2) échoue à
tolérance zéro — **13 px sur 2 073 600** (6.3·10⁻⁶), tous de 1 niveau, tous à
< 1.73·10⁻⁵ de la bascule de `np.rint` ; attribution MESURÉE par isolation :
cast f32 de l'entrée = 0 désaccord, **arithmétique f32 de la puissance
1/2.4 = 13** — exactement le risque que le prereg avait nommé. Conformément au
gravé, le runner s'est arrêté, verdict AUTRE, **chrono non lancé, aucune
tolérance relâchée**. Et le confondeur du champ test est démasqué AVANT d'avoir
fabriqué un faux : un champ 64×64 a **au plus 4096 valeurs distinctes par
construction** — le champ réel pavé rend 0 désaccord par PAUVRETÉ
d'échantillons ; la rampe (2.07·10⁶ valeurs, les deux branches sRGB) était le
choix adverse. « Tolérance zéro » est donc structurellement inatteignable sur
tout champ test à couverture dense : le critère devait changer, pas la mesure.

**DÉCISION ROMAIN : (A) — CRITÈRE DE NATURE, aucun seuil inventé** (gravé en
correction explicite du prereg P2) : (1) équivalence structurelle des formules
(recopie + sha — elle juge la FORMULE) ; (2) isolation booléenne : chaîne f64 à
entrée castée f32 ⇒ ZÉRO désaccord, sinon AUTRE ; (3) tout désaccord résiduel =
**bascule pure**, |Δniveau| == 1 exactement, sinon AUTRE ; (4) taux et distance
à la bascule SURFACÉS, jamais jugés. Portée dite : le critère de nature juge
l'ARITHMÉTIQUE, la recopie juge la formule. Options (B) chrono f64 (coût que la
production f32 ne paierait pas, bit-exact non garanti non plus) et (C) abandon
de la cellule (budget sans ancre) : NON RETENUES. **Les 13 désaccords mesurés
satisfont déjà le critère — mais il est gravé AVANT relance, et c'est le runner
qui re-prononcera mécaniquement.**

**GRAVÉ AVEC** : la règle des zones de la cellule 2 (le « ≪ » sans nombre était
un flou de plume de la session critique, consigné) — branche 1 si
médiane ≤ 0.5 ms (haut de bande = 0.227×marge), branche 2 si ≥ 2.199 ms (la
marge V4), entre les deux RIEN n'est prononcé, ratio surfacé. Les cinq
remontées de mission ACCEPTÉES (garde appelée depuis la campagne ; manettes
--n-appels/--n-chauffe avec `verdict_grade` ; `distance_max_a_la_bascule` ;
test de suite n'affirmant que la décidabilité). **Écho consigné, à ne pas
instruire** : même bête que le fait d'instrument CPU/BLAS de la manche 2 — une
pièce au dossier de « quantification du commis par format » (note v2 §5,
proposition non endossée).

**POINT D'ARRÊT : micro-mission d'implémentation du critère (chantier 4 de
`mission-prerequis-p2p3.md`), puis le chrono P2 est LANCÉ PAR ROMAIN,
verdict-grade natif. La session humaine P3 : après P2, décision explicite.**
