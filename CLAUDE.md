# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Nature du projet — lire avant tout

Ce dépôt n'est **pas** un produit logiciel : c'est un **PoC de recherche pré-enregistré** (« Test
T1 »). Il teste un seul claim (un routeur deux-physiques sur interface FSI contestée bat-il un
routeur de Harten codé en dur, en espace perceptuel, à compute inférieur ?). Le code est
l'instrument ; le **livrable est un verdict honnête**, pas une fonctionnalité.

> **PORTÉE DE LA PHRASE CI-DESSUS — lire avant de l'appliquer (précisé 2026-08-03).** Elle est
> vraie **de T1**, une question fermée à laquelle un verdict répond. Elle **ne gouverne pas le
> programme Cascade**, dont le but gravé est *« un moteur de monde voxel pour un jeu sandbox »*
> (note d'orientation v2 §1, ENDOSSÉ) — **un moteur qui tourne, jamais une publication**. Dans un
> programme de moteur, un verdict négatif est un blocage à contourner, pas un livrable.
> Conséquence opératoire : la **rigueur verdict-grade est proportionnelle au coût de se tromper**
> — réservée à ce dont l'erreur oblige à tout reconstruire (enveloppe mémoire, budget de frame,
> « un seul F » / fermeture sous-maille, contrat du registre). Partout ailleurs : **build par
> défaut, mesure quand une décision en dépend**. Cette phrase, appliquée sans sa portée, a produit
> six semaines de verdicts honnêtes et aucun moteur.

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
7. **LES TROIS FABRICATIONS — une seule règle, trouvée trois fois en une semaine (2026-08-03).**
   Un instrument qui ne peut pas produire le verdict contraire ne produit pas de verdict.
   - **Ne pas fabriquer un PASS** : un seuil manqué est un RÉSULTAT. Ne jamais déplacer un seuil,
     ni relâcher une garde, ni relancer une mesure jusqu'à ce qu'elle passe.
   - **Ne pas fabriquer un INDÉTERMINÉ** (§A52) : un critère d'indétermination se vérifie sur le
     cas **FAVORABLE** autant que sur le défavorable — sinon il punit la qualité qu'il contrôle.
     Corollaire : *un chiffre défavorable n'est pas plus sûr qu'un chiffre favorable*.
   - **Ne pas fabriquer un VERT** (§A53) : *un verrou numérique ne garde que ce que son ÉTAT DE
     TEST allume*. Un test qui ne peut pas échouer sur l'objet qu'il prétend tester ne teste rien.
     Et un terme **payé en arithmétique, nul en valeur** (bathymétrie plate, réconciliation
     positivité) ne se garde que **structurellement** — le retirer ne change aucun chiffre, passe
     tous les verrous numériques, et allège le code : la minoration silencieuse parfaite.

> Statut au dernier commit : **T1 clos.** Instrument FSI validé (G0 a/b/c) ; **C2 perceptuellement
> vacant à Re=100 2D** (substrat (quasi-)périodique → mémoïser-partout perceptuellement optimal).
> **T1.5 : consommateur INCERTAIN (2026-08-03).** Sous la note d'orientation v2, il n'y a plus de
> routeur à trois sorties comme objet premier — la question « `descend` jamais sollicitée » se
> reformule en « la politique d'élagage/croissance paie-t-elle son coût en JND-par-FLOP ? », qui
> appartient à la tranche-moteur, pas à un C2-bis sur cylindre. Par D17, **si aucune décision
> moteur n'en dépend, T1.5 se ferme** — décision Romain, non prise. S'il court, ce sera sur le
> substrat qui a déjà de la structure (inondation shallow-water), **jamais sur
> `experiments/mixed_substrate.py`** (scène falsifiée), et après réparation de `vorticity`
> (bords traités par `jnp.roll` sur un canal non périodique — `cascade/experts/regimes.py:28-32`,
> consommée par `experiments/c2_static_null.py:46`).
> Lire la conclusion de `PREREGISTRATION.md` avant de continuer — dernière entrée : **§A55**
> (2026-08-04). §A45 portée du corollaire de falsifiabilité (image seule) ; §A46 re-scoping audio
> impulsif et couplage τ_dec ; §A47 (+ PRÉCISION, + PRÉCISION-2) le régime de correction est à la
> **CLAUSE**, pas au document ; **§A48 la coupe deux-compositeurs** (le compositeur INSTRUMENT reste
> post-P3 ; un **`chemin-de-coût`** jugé au coût SEUL est autorisé avant, sous quatre gardes) ;
> **§A49 P0-son TRANCHÉ : β3** ; **§A50 l'ancre textuelle** ; **§A51 la séance-table tenue** ;
> **§A52 le premier chiffre mesuré** ; **§A53 le coût de F en 3D — ρ = 2,03, V4 meurt à 60 Hz** ;
> **§A54 le vérificateur d'ancres livré** ;
> **§A55 le multiplicateur `c` — la fourchette du cap est FERMÉE**.
>
> **TRANCHÉS — ne pas se réamorcer sur les documents qui les portent encore comme ouverts.**
> **P0-b** depuis §A36 (2026-07-25) : C-STRAT version F-unique — porté comme ouvert pendant neuf
> jours. **P0-son** (β3, §A49) et **l'échange compositeur** (§A48) depuis le 2026-08-03.
>
> **§A51 — SÉANCE-TABLE TENUE.** Vocabulaire épinglé, **c_fin 3D = 7,5 éq-f32**, dérivé ligne à
> ligne (roche creusable ⇒ `b0` = occupation volumique ; `id-matériau` u16 ; `e_ch` dérivé du
> ledger ; `e_th` = **enthalpie**, ce qui rend la fraction de glace gratuite). Enveloppe lue sous
> le **CAP D'EMPLACEMENTS** — *la formule dense est SUPERSÉÉE depuis §A16* — les deux branches en
> **B1** : la VRAM n'est pas la contrainte. **β est mort comme coefficient** (`M = 2·Σb_prim +
> Σb_dér`). **Streaming VRAM↔RAM promu** de levier à question dont dépend un facteur 5,8.
>
> **§A52 — PREMIER CHIFFRE.** Le **gather-plancher coûte 0,449 ms** à 1920×1080 (**2,7 %** du
> budget) ⇒ **G1**. La prédiction VRAM de §A51 tombe **au bit près**. Machine réelle :
> **3 781 Mo** de VRAM, pas 4 096.
>
> **§A53 — LE COÛT DE F EN 3D, l'INDÉTERMINÉE de §A51 LEVÉE.** À nombre de cellules **identique**
> (262 144 : 512² en 2D, 64³ en 3D), un **bloc** coûte 0,845 ms en 2D et **1,714 ms en 3D** ⇒
> **ρ = 2,03**. F seul, pour les 15 blocs de V4, coûte **25,7 ms = 1,54 × le budget de frame
> ENTIER** ⇒ branche pré-écrite **R-3, mort INCONDITIONNELLE** de V4 transposé.
> **Ce qui meurt est le COMPTE DE BLOCS à 60 Hz — ni la 3D, ni V4 en 2D.** Le cap se lit
> aussitôt : **8 blocs à 60 Hz, 18 à 30 Hz** ; V4 en demande 15. **La porte 33,3 cesse d'être un
> repli de gameplay pour devenir l'arbitrage de la dimension — À TRANCHER (Romain).**
> Deux multiplicateurs restent DUS et ne peuvent que réduire les 8 : le **non-F en 3D** (halos
> ~20 % des cellules contre ~2 %) et le **`c` 3D** (un bloc mesuré porte 5 champs ; §A51 en pose
> 7,5) — ce dernier dépend du **schéma eau 3D**, non tranché.
> **CORRIGE §A51:8258** : la marge 2D de V4 n'est pas 0,7 % (chiffre du 19/07, superséé le même
> jour par §A23-2b) mais **13,2 %** — ancre exacte, contenu périmé.
>
> **§A54 — LE VÉRIFICATEUR D'ANCRES EST LIVRÉ** (`verifier_ancres.py`, 20 verrous).
> **Le lancer avant d'écrire une entrée qui cite** : `.venv/bin/python verifier_ancres.py`
> (sortie non nulle sur ancre morte). Il rend huit états, dont `périmée` et `contestée`
> via un **registre de supersessions** déclaré — le troisième état de §A51-7, indétectable
> par le texte. **80 % des ancres du corpus (147/184) sont NUES**, donc invérifiables : la
> règle textuelle date du 03/08 et tout ce qui précède l'ignore. Conversion **au fil de
> l'eau** (toute entrée neuve au format textuel, toute ancre touchée convertie) — NON
> TRANCHÉ, c'est une décision de coût. **Tout verdict qui en supersède un autre doit
> ajouter sa ligne au registre**, sinon le troisième état retombe à la vigilance humaine,
> qui a échoué deux fois sur la même valeur en deux jours.
>
> **RÈGLES LOAD-BEARING, nées le 03/08 :**
> - Toute ancre **SORTANTE** porte son texte : `` `fichier:NNN` « fragment exact » `` — **le texte
>   fait foi, le numéro est le chemin** (§A50 ; 13 ancres nues sur 19 étaient fausses).
> - Une **méta-donnée** n'est jamais autorité contre ce qu'elle décrit — TOC, en-tête, numéro de
>   ligne (§A47-PRÉCISION-2). En cas de doute, **le CORPS**.
> - Le **pré-enregistrement se commit SEUL, AVANT le premier run** (§A52).
> - Une **sortie de console n'est pas un artefact** (§A52, famille §A41).
> - Un **chiffre défavorable n'est pas plus sûr** qu'un favorable ; un **critère d'indétermination
>   se vérifie sur le cas FAVORABLE** aussi (§A52).
> - Un **verrou numérique ne garde que ce que son état allume** (§A53) : un terme **payé en
>   arithmétique et nul en valeur** (corrections de pression à `b ≡ 0`, réconciliation positivité)
>   se retire sans changer un chiffre — seul un **inventaire STRUCTUREL** l'attrape.
>
> **§A55 — LE MULTIPLICATEUR `c` MESURÉ, LA FOURCHETTE EST FERMÉE.** Une fenêtre 64³ au
> **vocabulaire réel** (1 système + 3 scalaires advectés + 2 statiques lus) coûte
> **`C` = 2,4293 ms** ⇒ **`ρ_c` = 1,4623**, juste sous la borne haute ×1,50. **CAP :
> 6 fenêtres à 60 Hz, 12,97 à 30 Hz** ; V4 en demande 11 ⇒ **branche C-A, la porte 33,3
> tient V4** avec 14,3 % de marge (il faudrait que le non-F 3D TRIPLE pour la renverser).
> Décomposition : scalaires seuls ×1,3377 (**sous** le compte d'opérations 1,4348 —
> l'argument « une advection n'a pas de solveur de Riemann » est juste), **statiques
> +9,3 %** que la borne ignorait (elle comptait des champs, pas des octets lus).
> **LE COÛT N'EST PAS AFFINE, IL EST EN MARCHE D'OCCUPANCY** : 72→80→94→96 registres,
> 3 blocs/SM jusqu'à 2 scalaires puis 2 blocs/SM — l'occupancy tombe de 50 % à 33 %
> exactement là où l'incrément triple (+0,167 / **+0,394** / +0,267 ms). La pente est
> reportée (I-c4 passé à 94 % de sa bande) mais **ce n'est pas une loi**. Deux réductions
> nommées, non mesurées, toutes deux favorables : `id-matériau` lu au centre plutôt que
> dans le halo, et f16 sur `e_th`/`ρ_s`.
>
> **SUITE — LA DÉCISION DE CADENCE APPARTIENT À ROMAIN.** Les deux nombres existent
> désormais (6 à 60 Hz, 12 à 30 Hz, V4 en demande 11) et le choix ne se déduit pas d'eux.
> **C-A n'est PAS un feu vert** : l'autre moitié — rendu, ombrage, régime mobile, **non-F
> 3D** — reste inconnue, et le non-F ne peut que réduire le cap. Rien ne s'enchaîne.

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
