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

**D5+D6 TRANCHÉS (Romain, 2026-07-18) :** **(D5) Option A** — coupure au commit : le
chemin vivant a droit au non-déterminisme éphémère (GPU f32), le chemin de reconstruction
est bit-exact et seul porteur d'histoire ; falsificateur pré-écrit (tranche-moteur, écart
live↔rederive sous jnd_sev, sinon repli Option B). **(D6)** le format des entrées-joueur
est spécifié dès la v1 (§4), étiqueté `[TRANSPOSITION-HYPOTHÈSE]` jusqu'à la sonde
événement-joueur.

`[§3 ENDOSSÉE 2026-07-18]`

---

## §4. Le registre en production (v1 : 2D mono) `[EN COURS — soumise à Romain]`

**Format du ledger.** Séquence append-only (H1) d'entrées à ordre TOTAL : clé =
(t_sim quantifié, numéro de séquence) — mono-écrivain en v1 ; la concurrence multi-vue
est le problème PROPRE de la v1.1, gatée sur sa sonde (§1). Types d'entrées :
- **(a) événement seedé** `{t_sim, seed, type_procédural}` `[MESURÉ : tout le harnais]` ;
- **(b) entrée-joueur** `{t_sim quantifié, payload d'input, id_observateur}` —
  id_observateur ≡ 0 en v1, champ RÉSERVÉ pour v1.1 `[TRANSPOSITION-HYPOTHÈSE : format
  D6, jamais testé ; falsificateur : sonde événement-joueur sur harnais existant]` ;
- **(c) commit d'émission** `{t_sim, fenêtre (niveau, rect aligné-dyadique), SummaryQT
  (topologie u8, moyennes f64), k}` `[MESURÉ : §A13/§A14]` ;
- **(d) snapshot-racine** `{t_sim, commit PLEIN sans perte du domaine actif}` — voir
  compaction ci-dessous.

**Budget des commits.** k aire-proportionnel à la fenêtre au niveau fin, cap reconduit
(10 % de l'aire observée — le cap 409.6 de la famille 2 en est l'instance plein-domaine
64²) `[MESURÉ : k*=256 plein-domaine ; k_fen=64 fenêtré §A14]`.

**Cadencement des commits.** Défaut v1 : Δt FIXE par émission `[NON-ANCRÉ — c'est le
protocole mesuré]`. Optimisation NOMMÉE, non armée : cadencement-sur-relaxation — appui :
ressaut-puis-relaxation vu 2×, k*(Δt=16) < k*(Δt=4) au pin (plus l'écart est long, plus
le substrat dissipe le bruit de commit) `[MESURÉ comme PHÉNOMÈNE ; NON-ANCRÉ comme
politique]`. Si cette optimisation est achetée un jour : le falsificateur est l'analyse
τ_relax post-hoc sur les registres m2_parts (identifiée, ZÉRO épisode neuf, à armer à
l'achat seulement — discipline anti-tapis-roulant).

**Rejouabilité et sauvegarde.** save = le ledger, INTÉGRALEMENT ; load = re-dérivation
(chemin bit-exact §3) jusqu'à la dernière émission + reprise du chemin vivant. Débit
enveloppe ~3.2 Ko/commit, ~11.5 Mo/h à 1 commit/s `[MESURÉ-par-calcul : note VRAM]` ;
coût de rechargement à mesurer (tranche-moteur, §8).

**Compaction par snapshot-racine (type (d)).** Un ledger de session longue croît
linéairement ; la compaction remplace un préfixe par un snapshot-racine : commit PLEIN
sans perte → le replay repart de là, le préfixe est archivable/tronquable. Appui direct :
le contrôle ferm de la manche 2 a MESURÉ que le commit sans perte donne une chaîne
bit-identique (Δχ = 0.0 exact, tout Δt) `[MESURÉ : §A13-4-1 — le snapshot-racine est
exactement le bras ferm]`. Ce que la troncature COÛTE : l'histoire d'avant-snapshot
n'est plus re-dérivable depuis le ledger tronqué (le témoignage détaillé est remplacé
par son état final exact) `[NON-ANCRÉ : politique de rétention = décision produit,
hors v1 ; le FORMAT (d) est dans la v1, la politique non]`.

**Concurrence — cadrage Romain (2026-07-18, prépare la v1.1).** Dans cette architecture
l'observation N'EST PAS passive : commit ⇒ ré-ancrage ⇒ écriture. Deux émissions
simultanées sur fenêtres recouvrantes ne commutent pas bit-à-bit (projections avec
perte). MAIS cette non-commutativité est SANS SÉMANTIQUE : tout ordre total convient,
il doit seulement exister et être gravé — la clé (t_sim, seq) du ledger suffit. La sonde
v1.1 mesure précisément : écart A∘B vs B∘A sur recouvrement SOUS JND ? (si oui, l'ordre
d'observation est un détail d'implémentation). La PRIMEUR des ACTIONS joueur (qui assigne
seq quand deux joueurs agissent dans le même quantum) est un problème classique
d'horodatage/AUTORITÉ, nommé HORS-PHYSIQUE : la couche réseau/jeu choisit sa politique ;
la spec garantit seulement — et garantit déjà — *ordre total donné ⇒ reconstruction
bit-exacte* (H3). `[NON-ANCRÉ côté politique ; la garantie H3 est MESURÉE au harnais]`

**Décisions Romain pour clore §4 :**
- **(D7) Cadencement v1 = Δt fixe** (défaut mesuré), cadencement-sur-relaxation nommé
  non-armé — endosser tel quel ?
- **(D8) Le type (d) snapshot-racine dans le format v1** (adossé au bras ferm mesuré),
  politique de rétention explicitement HORS v1 — endosser tel quel ?

---

## §5. GPU-déterminisme (OBLIGATOIRE — v1 : 2D mono) `[EN COURS — soumise à Romain]`

**Topologie de calcul (proposition Romain 2026-07-18, amendée).** Le niveau 0 de la
pyramide (+ niveaux froids) vit CPU-RAM/disque ; les fenêtres actives DESCENDENT sur GPU
(prédiction Harten), F fin s'applique sur GPU, la remontée passe PAR LE DIFF — qui, en
Harten, a un nom : les coefficients de détail. L'interface CPU↔GPU est la transformée
elle-même ; le trafic est à l'échelle n_fov², jamais du monde `[NON-ANCRÉ : design ;
falsificateur : la tranche-moteur mesure frame-time ET débit PCIe réel]`.

**Amendement 1 — le niveau 0 CPU est du VIVANT, pas la colonne déterministe.** Le diff
remontant est f32 GPU non-déterministe (droit Option A, §3) : l'appliquer à une colonne
« autoritaire » la rendrait non-rejouable. Le niveau 0 CPU appartient au CHEMIN VIVANT
(perceptuel, dérive sous-JND admise). La colonne déterministe n'est PAS un niveau de la
pyramide : c'est (ledger, rederive). Le diff GPU ne touche JAMAIS le persistant — porte
unique H2 : {F déterministe du rederive, ré-ancrage} seuls écrivent le persistant.
Bonus : la conservation (É1) est vérifiable au point de passage du diff (contrôle
d'instrument de production, coût borné).

**Le chemin de reconstruction.** Rederive = CPU f64 pur, mêmes disciplines que le
harnais `[MESURÉ : tout §A13/§A14 tourne sur ce chemin]`. Coût : hors-frame par
construction (load, replay, audits) — jamais dans la boucle de jeu.

**Amendement 2 — É3 est MACHINE-LOCALE, fait mesuré.** La bit-exactitude ne voyage pas
entre machines : FAIL cloud mesuré 2×, écart max 2.7 % d'état à 10 épisodes, cause
CPU/BLAS `[MESURÉ : ledger SDD 2026-07-15]`. La v1 déclare donc : garantie É3 = par
machine. Inter-machines, la garantie retombe à du sous-JND — **NON MESURÉ en espace
readout** `[TRANSPOSITION-HYPOTHÈSE : falsificateur GRATUIT — relire l'écart du FAIL
cloud existant dans Δχ/JND, zéro calcul neuf ; verse au menu §8. Si sous-JND : le
multi-machine v1.1 respire ; sinon : le multijoueur exige une autorité de simulation]`.

**Ce que la v1 exige donc du GPU : RIEN en déterminisme.** Aucun kernel ordonné, aucune
réduction déterministe — le non-déterminisme GPU est intégralement quarantiné dans le
chemin vivant par la topologie ci-dessus. Le seul contrat chiffré : écart live↔rederive
aux émissions sous jnd_sev (§3, falsificateur tranche-moteur).

**Décisions Romain pour clore §5 :**
- **(D9)** Endosser la topologie CPU-coarse-vivant / GPU-fin / remontée-par-diff, AVEC
  l'amendement 1 (niveau 0 = vivant ; colonne déterministe = ledger+rederive) ?
- **(D10)** Déclarer É3 machine-locale en v1 (fait mesuré), l'hypothèse sous-JND
  inter-machines étiquetée avec son falsificateur gratuit au menu §8 ?

**D7-D10 TRANCHÉS (Romain, 2026-07-18) : les quatre endossés tels quels.**

`[§4 ENDOSSÉE 2026-07-18]` `[§5 ENDOSSÉE 2026-07-18]`

---

## §6. Budget mémoire (OBLIGATOIRE — v1 : 2D mono) `[EN COURS — soumise à Romain]`

**Épinglage v1 (le quadruplet EST le budget) :** c = 8 champs au niveau fin
`[NON-ANCRÉ : le substrat-jeu réel n'est pas figé — 8 est l'enveloppe de travail, à
re-épingler quand le substrat v-jeu se fige]` ; b = 4 (f32, chemin vivant GPU/VRAM ; le
rederive est f64 CPU-RAM, HORS VRAM par construction §5) ; n_fov = 512 ; N_niv = 10
`[D3]`. Enveloppe : M_VRAM ≈ 0.9 Go (formule note VRAM, β = 1.5, double-buffer ×2) —
reste > 3 Go pour rendu + framework sous les 4 Go de la 3050 Ti
`[MESURÉ-par-calcul : note VRAM, hypothèses nommées]`. Niveau 0 CPU : ~256² cellules,
trivial. Ledger : ~3.2 Ko/commit, ~11.5 Mo/h. Leviers nommés NON comptés dans
l'enveloppe (marge cachée, jamais créditée d'avance) : c décroissant par niveau, masques
dormants, streaming, compression des niveaux froids.

**Gate de section :** si la tranche-moteur mesure une résidence réelle > enveloppe ×1.5,
le quadruplet est RE-ÉPINGLÉ (décision, pas glissement) — jamais d'enveloppe ajustée
silencieusement après mesure.

**Décision Romain :** **(D11)** endosser le quadruplet (c=8, f32, n_fov=512, N_niv=10) ?

---

## §7. Pins requis et dettes (v1) `[EN COURS — soumise à Romain]`

Inventaire exhaustif des référents perceptuels — chacun avec statut et consommateur :

- **Pin spatial central** : MESURÉ (Arc C : jnd_sev 7.33 %, IC [6.03, 8.67] % ; laxiste
  11.54 % [9.37, 13.87] %). Portée n=1, gravée. Consommateurs servis : §A12, §A13, §A14.
- **r_fovea / JND d'excentricité** : LE pin manquant load-bearing — S_eff en dépend
  (1.8–3.9 mesuré selon r_fovea) et la règle D4 l'attend. Mesurable : extension
  excentricité du harnais Arc C (ABX + staircase, stimuli décalés du centre) —
  campagne HUMAINE. **GARDE ANTI-TAPIS-ROULANT gravée : cette campagne ne se lance
  QU'APRÈS un verdict tranche-moteur sans mort** — mesurer r_fovea avant de savoir si la
  frame tient serait dé-risquer le mauvais axe (le pin n'a de consommateur que si le
  moteur existe).
- **σ_ω (axe temporel)** : DETTE, condition de réveil gravée (décision (d), 2026-07-18) —
  rouvre si le paper-grade nomme un consommateur temporel précis. Aucun réveil silencieux.
- **Hypothèse inter-machines sous-JND** (§5) : falsificateur GRATUIT au menu §8.
- **Transfert albédo→luminance** : hypothèse nommée héritée (§C4-3), dormante avec σ_ω.

**Décision Romain :** **(D12)** endosser l'ordre — tranche-moteur D'ABORD, campagne
r_fovea SEULEMENT après verdict sans mort ?

---

## §8. Falsificateurs et gates de sortie (v1) `[EN COURS — soumise à Romain]`

La récolte : toute `[TRANSPOSITION-HYPOTHÈSE]` load-bearing du document a ici son billet
de mesure. Ordre de dépense (du gratuit au cher, cheapest-falsifier-first) :

1. **F0 — GRATUIT : relecture perceptuelle du FAIL cloud** (hypothèse inter-machines,
   §5) — repasser l'écart 2.7 % existant dans Δχ/JND. Zéro calcul neuf.
2. **F1 — TRANCHE-MOTEUR** (le seul gros achat v1 ; brûle 4 hypothèses d'un coup) :
   (a) frame-time vs L_eff sur 3050 Ti dans l'enveloppe §6 — le falsificateur du mot
   « moteur » ; (b) écart live↔rederive aux émissions sous charge f32 (contrat D5) ;
   (c) débit PCIe réel du schéma diff (§5) ; (d) coût de load/replay du ledger (§4).
   CRITÈRES DE MORT PRÉ-ÉCRITS au pré-enregistrement de la tranche (la LISTE est ici,
   les chiffres se figent là-bas, AVANT tout run — jamais après).
3. **F2 — sonde événement-joueur** (format (b) du ledger, §4) : harnais existant, coût
   type sonde.
4. **F3 — sonde deux-fenêtres concurrentes** (GATE v1.1) : commutativité sous-JND des
   ré-ancrages recouvrants (cadrage §4).
5. **F4 — sonde 3D minimale** (GATE v2) : conversion des transpositions dimensionnelles.
6. **F5 — τ_relax post-hoc** : armé SEULEMENT à l'achat du cadencement-relaxation (§4).

**Gate de sortie de la spec v1 :** la spec FAIT FOI quand (i) toutes les sections sont
endossées, (ii) F0 est lu, (iii) F1 a rendu un verdict SANS MORT. Alors seulement le
code moteur commence — paper grade → mesure → build, la discipline maison portée au
niveau moteur. Un critère de mort F1 déclenché = remontée + décision, jamais un
contournement.

**Décision Romain :** **(D13)** endosser l'ordre de dépense F0→F1→F2 (F3/F4 gatés à
leurs étages, F5 conditionnel) et le gate de sortie ?
