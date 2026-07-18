# SPEC FOVÉA-Z — v0 (PAPER GRADE, EN CONSTRUCTION)

> **Statut : BROUILLON.** Gate d'ouverture : décision (d) + §A13-résultat (PREREGISTRATION,
> 2026-07-18). Chaque section est soumise à Romain et marquée `[ENDOSSÉE]` ou `[EN COURS]` ;
> rien ne fait foi avant endossement. Aucun code de spec avant que le paper grade soit
> complet. Appuis mesurés : §A13-résultat (REGISTRE_FERME), §A14-lecture (fenêtrage),
> note budget VRAM (pocPhysicator claude/note-budget-vram-2026-07-18.md), f_p(L) + S_eff
> (journal 2026-07-01). Dettes portées : σ_ω (condition de réveil gravée), pin r_fovea
> (NON mesuré — le référent que la spec doit traiter en paramètre pinnable, jamais choisir).

## Squelette (v0)

- **§1. Objet et claim architectural** — ce que la fovéa-z EST, le claim falsifiable
  qu'elle porte, ce que la spec n'est pas. `[EN COURS — soumise]`
- **§2. Structure de z** — pyramide Harten, plafond LOD par distance, fenêtres actives,
  résidence mémoire (VRAM/RAM/disque). `[À RÉDIGER]`
- **§3. Frontière éphémère/persistant** — qui écrit où ; droits au non-déterminisme ;
  invariants machine-à-états (candidats Hypothesis). `[À RÉDIGER]`
- **§4. Le registre en production** — cadencement des commits (appui ressaut-relaxation
  ×2 ; τ_relax à armer si cette section le demande), commits fenêtrés (appui §A14),
  format/taux du ledger, rejouabilité. `[À RÉDIGER]`
- **§5. GPU-déterminisme (OBLIGATOIRE)** — où la bit-identité est requise vs permise de
  mourir ; stratégie candidate ; ce qui doit être MESURÉ (tranche-moteur). `[À RÉDIGER]`
- **§6. Budget mémoire (OBLIGATOIRE)** — épinglage (c, b, N_niv, n_fov^d) contre
  l'enveloppe de la note VRAM ; scénario normatif. `[À RÉDIGER]`
- **§7. Pins requis et dettes** — r_fovea (prochaine mesure candidate, harnais Arc C),
  JND d'excentricité, σ_ω (dette + condition de réveil). `[À RÉDIGER]`
- **§8. Falsificateurs et gates de sortie** — quelles mesures tuent quelles sections ;
  tranche-moteur avec critères de mort pré-écrits ; règles de dépense. `[À RÉDIGER]`

---

## §1. Objet et claim architectural `[EN COURS — soumise à Romain]`

**Ce que la fovéa-z est.** Le mécanisme qui borne l'échelle fine à la VUE de l'observateur,
pas au monde : la distance fixe un plafond de LOD ; l'énergie raffine SOUS ce plafond ;
le monde n'entre dans le coût qu'en log (pyramide Harten — chiffré, note VRAM). C'est le
pilier load-bearing identifié le 2026-07-01 : sans lui, la fenêtre `1 < G < S` est trop
étroite (2D ~2.5×L₀, 3D ~1.7×L₀) et l'échelle effective sort de la fenêtre dès que le
monde grandit. La fovéa-z change le régime de scaling du problème de fermeture
`f_p ∝ L^(-0.78)` : `L_eff` est borné par l'observation, plus par le monde.

**Le claim falsifiable que la spec porte** (reconduit du journal 2026-07-01, endpoint
honnête) : *Cascade tient au budget-jeu ssi (a) le fin-fovéa tient la frame ET (b) le
routage paie dans la fovéa.* État des maillons : (b) MESURÉ (S≈3 couplage doux ; LOD
multiplie S ~2×, S_eff 1.8–3.9 selon r_fovea) ; (a) NON MESURÉ — c'est l'objet de la
tranche-moteur (§8), dans l'enveloppe mémoire (§6). Le référent qui arbitre (a)↔(b) est
le pin JND/fovéa (r_fovea) : NON mesuré, traité partout ici comme PARAMÈTRE PINNABLE —
choisir sa valeur pour faire passer un budget serait fabriquer le verdict (§7).

**La colonne persistante.** Le registre-commis §A13 (REGISTRE_FERME, k\* ≤ 256 floats
plein-domaine 64², k_fen aire-proportionnel tenu en fenêtré §A14) est le mécanisme par
lequel l'histoire persiste : commits d'émission + ledger d'événements seedés,
reconstruction déterministe É1/É2/É3. La spec l'installe comme LA colonne vertébrale de
l'état persistant — le monde procède sur (seed, registre), conséquence §A13-0.

**Ce que la spec n'est pas.** Pas un game-design (aucun contenu, aucune mécanique de
jeu) ; pas un plan d'implémentation (aucun choix de librairie hors contraintes mesurées) ;
pas une promesse de qualité perceptuelle produit (le pin est n=1 : référent de conception,
pas spec produit). C'est le contrat d'architecture FALSIFIABLE : chaque section nomme ce
qui la tuerait et quelle mesure a le droit de le faire (§8).

**D1+D2 TRANCHÉS (Romain, 2026-07-18) — ÉTAGEMENT :**
- **v1 (ce document) : 2D, mono-observateur** — l'étage normatif, adossé à tout l'appui
  mesuré existant.
- **v1.1 : multi-vue 2D** — étage suivant, GATÉ sur la sonde deux-fenêtres concurrentes
  (commits concurrents sur le même registre : ordre total du ledger, commutativité —
  falsificateur constructible sur le harnais §A14 existant, à pré-enregistrer le moment
  venu).
- **v2 : addendum 3D** — GATÉ sur une sonde 3D minimale + conversion des hypothèses de
  transposition load-bearing en mesures. La fine-fovéa 3D est déjà bornée (~64³, note
  VRAM).

**RÈGLE D'ÉTIQUETAGE (s'applique à tous les étages, non négociable) :** chaque affirmation
de la spec porte `[MESURÉ]` (avec sa source), `[TRANSPOSITION-HYPOTHÈSE]` (nommée, avec le
falsificateur qui la brûlerait) ou `[NON-ANCRÉ]` (choix de design assumé, réversible).
§8 doit acheter le falsificateur le moins cher de chaque `[TRANSPOSITION-HYPOTHÈSE]`
load-bearing avant que l'étage correspondant fasse foi.

`[§1 ENDOSSÉE 2026-07-18]`

---

## §2. Structure de z (v1 : 2D mono) `[EN COURS — soumise à Romain]`

**z est une pyramide de Harten, pas un champ.** Niveaux j = 0 (le plus grossier, monde
entier, TOUJOURS résident) à J (le plus fin, fovéa seule). La fenêtre fovéale au niveau J
a n_fov cellules de côté ; chaque doublement de distance à l'observateur abaisse le
plafond d'un niveau `[NON-ANCRÉ : règle de design par défaut — sa validité PERCEPTUELLE
dépend du JND d'excentricité, non pinné (§7) ; réversible sans casser la structure]`.
Sous le plafond fixé par la distance, c'est l'ÉNERGIE qui décide du raffinement effectif
(pilier existant) `[MESURÉ : S_eff 1.8–3.9 selon r_fovea, journal 2026-07-01]`.

**Coût structurel.** Cellules actives ≈ γ₂·n_fov²·N_niv, γ₂≈3 — le monde n'entre qu'en
log `[MESURÉ-par-calcul : note VRAM, hypothèses nommées à épingler en §6]`.

**Contenu d'une cellule.** z porte les PRIMITIVES seulement ; les dérivés (vorticité,
pression) vivent dans le cache matérialisé, jamais dans z `[NON-ANCRÉ : principe
d'architecture établi (state vs materialized cache), reconduit]`. Le nombre de champs c
au niveau fin est un paramètre de §6 ; c peut DÉCROÎTRE avec j (champs réduits au large)
`[NON-ANCRÉ : levier nommé note VRAM, non compté par défaut]`.

**Résidence.** VRAM = pyramide active (fovéa + fenêtres d'énergie sous plafond) ;
RAM = niveaux froids / régions dormantes (masques wet/dry, repos sédimentaire) ;
disque = ledger (§4) + artefacts gelés. Le streaming VRAM↔RAM hors fovéa est un levier
nommé, pas une exigence v1 `[NON-ANCRÉ]`.

**Le lien au registre (§4).** Les commits d'émission sont fenêtrés et aire-proportionnels
en budget `[MESURÉ : §A14-lecture — k_fen = 64 sur fenêtre 32² tient comme k* = 256 sur
64², pic au même événement, contamination dehors→dedans sous plancher]`. La structure de
z DOIT donc exposer l'extraction de fenêtre alignée-dyadique à tout niveau (le plongement
§A14 (v) est le prototype de cette interface).

**Décisions Romain pour clore §2 :**
- **(D3) n_fov et N_niv cibles de la v1** — proposition par défaut : n_fov = 512,
  N_niv = 10 (monde linéairement ~512× la fovéa, ~0.9 Go à c = 8 f32). C'est un ÉPINGLAGE
  §6, mais l'ordre de grandeur doit être choisi ici pour que §4-§6 chiffrent.
- **(D4) La règle un-niveau-par-doublement** — l'endosser comme défaut v1 réversible, ou
  exiger dès la v1 une règle paramétrique (plafond = f(distance) à pente libre) ?
