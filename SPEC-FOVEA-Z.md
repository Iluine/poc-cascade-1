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

**D3+D4 TRANCHÉS (Romain, 2026-07-18) :**
- **(D3)** n_fov = 512, N_niv = 10 — cibles v1 (monde linéairement ~512× la fovéa,
  ~0.9 Go à c = 8 f32) ; épinglage définitif en §6.
- **(D4)** Règle un-niveau-par-doublement ENDOSSÉE comme défaut v1 RÉVERSIBLE
  `[NON-ANCRÉ]` — le pin d'excentricité (§7) la re-réglera le cas échéant.

`[§2 ENDOSSÉE 2026-07-18]`

---

## §3. Frontière éphémère/persistant (v1 : 2D mono) `[EN COURS — soumise à Romain]`

**Définitions.** ÉPHÉMÈRE : état dont la pertinence perceptuelle décroît — rejouable ou
oubliable, n'entre JAMAIS au registre, peut être recalculé différemment d'un replay à
l'autre tant que la plausibilité perceptuelle tient (jamais de L2 — principe fondateur).
Exemples : vague d'étrave, fumée, éclaboussures. PERSISTANT : état qui porte témoignage —
observé (émis) ou porteur d'invariants (masses). N'entre que par le registre ; son replay
obéit à É1/É2/É3 `[MESURÉ : §A13-résultat + §A14-lecture pour la variante fenêtrée]`.

**La frontière est un ÉVÉNEMENT, pas une typologie.** La fumée qui noircit un mur devient
persistante. Ce qui fait passer la frontière, c'est le COMMIT : l'émission capture
l'état∣fenêtre du moteur (dynamique éphémère incluse — le commit capture le RÉSULTAT,
§A14 (v) le prototype) ; les flux d'invariants passent par leur colonne propre (bus
d'énergie, la magie comme transducteur conservé — pilier existant). Slogan gravé
reconduit : seul le détail réinjecté paie la taxe de déterminisme bit-exact.

**LE NŒUD — le contrat live↔re-dérivation.** Relecture du claim §A13-0, à la lettre :
É2 dit « tout readout émis est re-dérivable SOUS JND_sev » — PAS bit-exact ; c'est É3
qui exige le bit-exact, et il ne l'exige QUE de la reconstruction avec elle-même (double
re-dérivation identique). Le contrat sépare donc DEUX chemins :
- **chemin de reconstruction** (rederive) : f(registre, seeds) uniquement, BIT-EXACT,
  auto-cohérent — c'est LUI qui porte la non-contradiction de l'histoire ;
- **chemin vivant** (la frame jouée) : doit rester SOUS JND_sev du chemin de
  reconstruction aux émissions — c'est tout ce que É2 exige de lui.
Conséquence architecturale : le chemin vivant a DROIT au non-déterminisme (GPU f32,
réductions non ordonnées) sur ses composantes éphémères ; le chemin de reconstruction
n'y a jamais droit. L'écart live↔rederive aux émissions devient un CONTRÔLE D'INSTRUMENT
de production, mesurable en continu `[TRANSPOSITION-HYPOTHÈSE : dans le harnais, les deux
chemins coïncident bit-à-bit (CPU f64) — l'écart sous charge GPU f32 est NON MESURÉ ;
falsificateur : la tranche-moteur mesure cet écart, critère pré-écrit : sous jnd_sev,
sinon repli Option B (tout-déterministe) et son coût de frame mesuré]`.

**Les deux colonnes du ledger.** (1) Événements : entrées seedées (procédurales —
`[MESURÉ]`) et entrées-joueur (inputs quantifiés en temps-simulation, structurellement
identiques — `[TRANSPOSITION-HYPOTHÈSE : jamais testé, §A13-1 l'a nommé ; falsificateur :
sonde événement-joueur sur le harnais existant]`). (2) Commits d'émission : fenêtrés,
budget aire-proportionnel `[MESURÉ : §A14]`. Taux enveloppe : ~3.2 Ko/commit,
~11.5 Mo/h à 1 commit/s `[MESURÉ-par-calcul : note VRAM]`. Le fichier de sauvegarde du
jeu EST le ledger — le monde se recharge par re-dérivation `[NON-ANCRÉ : conséquence de
design de §A13-0, coût de rechargement à mesurer en tranche-moteur]`.

**Invariants machine-à-états (candidats Hypothesis, v1) :**
- **H1 append-only** : aucune mutation du ledger, jamais (propriété d'API).
- **H2 porte unique** : le persistant de z n'est modifié QUE par {pas de physique F,
  ré-ancrage depuis commit} — aucune écriture directe.
- **H3 idempotence de replay** : double re-dérivation bit-identique (É3 — mesuré au
  harnais, devient test de propriété permanent).
- **H4 projection** : S∘R = id sur les fenêtres commises `[MESURÉ : test §A14]`.
- **H5 isolement de l'éphémère** : purger tout l'état éphémère puis re-dériver depuis le
  ledger redonne des readouts sous JND_sev des émissions vécues (le contrôle du nœud
  ci-dessus, en version invariant).

**Décisions Romain pour clore §3 :**
- **(D5) Le contrat live↔rederive** : endosser l'**Option A** (coupure au commit — live
  perceptuel avec droit au non-déterminisme éphémère, rederive bit-exacte seule porteuse
  d'histoire) comme défaut v1 avec son falsificateur pré-écrit en §8, l'Option B
  (tout-déterministe, y compris le vivant) ne devenant obligatoire que si l'écart
  traverse ? Ou imposer B d'emblée (plus sûr, coût de frame inconnu, probablement
  incompatible GPU) ?
- **(D6) Les entrées-joueur dans le ledger v1** : spécifier leur FORMAT dès la v1
  (design pur, étiqueté transposition non testée, brûlé plus tard par une sonde
  événement-joueur) — ou les repousser en v1.1 et garder la v1 strictement procédurale ?
