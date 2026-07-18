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

**Décisions Romain pour clore §1 :**
- **(D1) Scénario normatif de la v1 : 2D d'abord ou 3D d'abord ?** L'enveloppe VRAM dit :
  2D-512² confortable (~0.9 Go), 3D-64³ serré-mais-tenable (~2.1 Go). La ligne PoC est
  2D ; l'ambition jeu est 3D. La v1 peut se lier au 2D (continuité instrumentale, tout
  l'appui mesuré est 2D) en portant le 3D comme addendum gaté, ou viser 3D directement
  (honnêteté d'ambition, mais AUCUN appui mesuré n'est 3D).
- **(D2) Modèle d'observateur de la v1 : fovéa unique, observateur unique ?** Tout
  l'appui mesuré est mono-observateur. Le multi-observateur (coop) serait porté comme
  dette nommée de §7, pas résolu par la v1.
