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
8. **L'APPARIEMENT EST LA CONDITION D'EXISTENCE DU CONTRASTE** (§A59/§A60/§A61 — trois
   fois violée en une nuit). *Une comparaison ne vaut que si ses deux termes ne diffèrent
   QUE par la variable testée* — et « en plus » inclut **le rang dans l'exécution**.
   Nommer un confondant ne le contrôle pas : seul l'appariement le contrôle. Les trois
   occurrences : groupes non appariés en **taille** (§A59) ; **filtrer puis agréger** sans
   re-nommer le périmètre (§A60) ; contrôle non apparié dans le **temps** — 1ʳᵉ mesure
   contre 4ᵉ, +25 °C (§A61).

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
> Lire la conclusion de `PREREGISTRATION.md` avant de continuer — dernière entrée : **§A62-bis**
> (2026-08-04, `:10104`). §A45 portée du corollaire de falsifiabilité (image seule) ;
> §A46 re-scoping audio
> impulsif et couplage τ_dec ; §A47 (+ PRÉCISION, + PRÉCISION-2) le régime de correction est à la
> **CLAUSE**, pas au document ; **§A48 la coupe deux-compositeurs** (le compositeur INSTRUMENT reste
> post-P3 ; un **`chemin-de-coût`** jugé au coût SEUL est autorisé avant, sous quatre gardes) ;
> **§A49 P0-son TRANCHÉ : β3** ; **§A50 l'ancre textuelle** ; **§A51 la séance-table tenue** ;
> **§A52 le premier chiffre mesuré** ; **§A53 le coût de F en 3D — ρ = 2,03, V4 meurt à 60 Hz** ;
> **§A54 le vérificateur d'ancres livré** ;
> **§A55 le multiplicateur `c` — la fourchette du cap est FERMÉE** ;
> **§A56 la table a TROIS monnaies, une seule linéaire** ; **§A57 la porte 33,3 réserve
> ~13 ms au rendu — V4 ne tient pas non plus à 30 Hz** ; **§A58 la machinerie n'exige que la
> parité** ; **§A59 le balayage de tailles est INDÉTERMINÉ** ;
> **§A60 l'instrument lui-même est en cause** ;
> **§A61 la machine n'a pas de point de fonctionnement stable** ;
> **§A62 le rendu est PLACÉ — il décide le côté, pas la cadence** ;
> **§A62-bis quatre points d'une relecture adverse, tous justes — trois SUR-VENTES DE
> PORTÉE qu'aucune garde pré-écrite n'attrape, et `I-r5` écrite mais non codée**.
> *(Ce pointeur est une méta-donnée : le CORPS du journal fait foi contre lui — §A47-
> PRÉCISION-2. Il annonçait §A61 alors que deux entrées suivaient ; corrigé le 23/08.)*
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
> **§A56 — LA TABLE A TROIS MONNAIES, UNE SEULE LINÉAIRE.** Le budget ne compte **ni des
> champs ni des octets lus** : l'artefact de §A55 contient une **paire iso-octets** —
> M-c3 (3 scalaires) et M-c6 (1 scalaire + 2 statiques), mêmes 7 champs, mêmes 1 008
> octets lus, **25,1 % d'écart de coût**. Les trois monnaies : **les champs**
> (arithmétique de flux — un scalaire advecté ajoute 9 minmods et 6 upwinds, **zéro
> solveur de Riemann**) ; **les octets lus** (bande passante — ce qu'un statique ajoute,
> et rien d'autre : +9,3 %) ; **les REGISTRES**, seule monnaie à **FALAISE** (l'occupancy
> tombe de 50 % à 33 % entre le 2ᵉ et le 3ᵉ scalaire, l'incrément triple).
> ⇒ **La réserve de 4 éq-f32 de §A51 a un prix NON LINÉAIRE qui dépend d'où elle
> s'encaisse**, et le découpage monolithique/passes séparées est un levier à taux de
> change mesuré. **Ligne pour la table, pas décision.** Au vocabulaire réel : **1 296
> octets lus** par cellule et par étage (et non 720, qui est le jouet à 5 champs).
>
> **DETTE DE RELECTURE : §A54 est passé sans seconde lecture.** Deux points y méritent
> celle de Romain — les seuls où la session a exercé une **autorité sur le corpus** au
> lieu de le décrire : les **trois citations non verbatim déclarées**, et surtout le
> **REGISTRE DE SUPERSESSIONS**, dont les entrées décident ce qui est périmé dans des
> documents endossés.
>
> **§A57 — LA PORTE 33,3 NE DONNE PAS 33,3 ms À LA PHYSIQUE.** `:4217-4219` réserve ~13 ms
> au rendu 60 fps ⇒ budget physique ~20,3 ms ⇒ **7,61 fenêtres, V4 (11) NE TIENT PAS non
> plus à 30 Hz**. La marge de 14,3 % de §A55 était celle d'un 30 Hz qui n'existe pas — le
> seuil pré-écrit était trop généreux, et **la session l'avait écrit elle-même**.
> **Mais cellules × cadence est CONSERVÉ** (96,2 vs 102,0 M cellules·Hz) : la cadence
> **n'est pas la variable**, elle redistribue. ⇒ **les 11 fenêtres tiennent à 60 Hz si
> elles font ~52³** — d'où `s_max(budget) = 52,6`, **une INVERSION, jamais une constante**.
>
> **§A58 — la machinerie n'exige que la PARITÉ** : elle ne divise jamais `n_fov` plus d'une
> fois (les fenêtres ont la même largeur à tous les niveaux ; c'est le MONDE qui double).
> {admis} ∩ {s ≤ 52,6} = {24…52}. Et un critère inventé par la session rejetait TOUTES les
> tailles, `64` compris — **attrapé par le témoin gravé placé dans le balayage**.
>
> **§A59 — LE BALAYAGE DE TAILLES EST INDÉTERMINÉ, ET C'EST UN RÉSULTAT.** `I-t3`, le seul
> critère braqué sur le cas **FAVORABLE**, a tiré : `s=40` (chevauchant) sort moins cher que
> TOUS les alignés. **Deux causes, toutes deux fautes du protocole** : (1) les groupes
> n'étaient **pas appariés en taille** — le contraste mesurait la taille, et le prereg
> **nommait lui-même le confondant au paragraphe suivant** ; (2) les **médianes sont
> contaminées par la gigue** aux petits côtés (étendue 36 % sur médiane contre 12 % sur
> minimum ; l'anomalie `s=32` se dissout). La lecture n'a **pas** été basculée sur le
> minimum — changer d'instrument après avoir vu le résultat est la faute. ⇒ **l'inversion
> `s_max = 52,6` reste INVALIDÉE**, aucune constante de côté n'en sort. Le plan qui
> répondrait : des **paires appariées** `(62,64) (94,96) (126,128) (34,32)`, et une série
> dimensionnée sur la DURÉE de frame, non sur un compte fixe.
>
> **§A60 — L'INSTRUMENT EST EN CAUSE, ET ÇA TOUCHE §A53/§A55/§A59.** Le plancher gravé
> « série ≥ 300 frames » (`chrono.py:3-6`) a été écrit pour le harnais 2D ; sur les kernels
> 3D il **ne dilue pas le transitoire de montée en fréquence du GPU**. Les neuf grands
> côtés ACCÉLÈRENT pendant leur série (2ᵈᵉ moitié 12–18 % plus rapide) ; les petits, avec
> 2 200–2 900 frames, sont stables à 1 %. ⇒ **les absolus 3D du programme sont
> PESSIMISTES d'environ 15 %** — §A55 donnait 7,288 ms à 64³, la 1ʳᵉ moitié d'aujourd'hui
> 7,457 et la 2ᵈᵉ **6,117**. **Les rapports y échappent peut-être, ce n'est PAS établi**
> (en 2D les frames sont plus courtes, donc 300 frames y couvrent moins de temps : `ρ`
> pourrait être SOUS-estimé). **DÛ NEUF : re-qualifier l'instrument** — caractériser le
> transitoire et en déduire un warmup MESURÉ, pas décrété — **avant toute mesure 3D**.
> §A53/§A55/§A59 ne sont **pas rétractées** (protocoles respectés, témoins reproduits) et
> **aucune conclusion de gate n'en dépend** : la mort de V4 en 3D tenait à 1,54× le budget
> entier, qu'un biais de 15 % ne renverse pas.
> Seul survivant du run : **T32**, puissance 0,56 %, `Δ = −1,99 %` significatif —
> l'alignement paie, **mais 5× trop peu** pour déplacer le candidat. Un triplet ne fait
> pas un verdict.
>
> **§A61 — LA MACHINE N'A PAS DE POINT DE FONCTIONNEMENT STABLE.** Throttle `0x4` = **SW
> Power Cap** dans les cinq séries sondées ; horloge SM de **1 035 à 1 732 MHz** selon la
> mesure ; un « plateau » parfois **plus bas** que le premier relevé ; température
> **54 → 79 °C** sur le run, avec dérive **monotone** des plateaux (6,038 → 6,243 ms à
> 64³). `T_conv` = **2,5 s** à 64³, **19,3 s** à 128³, **22,1 s** à 32³ — **aucune
> constante, ni en secondes ni en frames**. ⇒ **le protocole gravé suppose un processus
> STATIONNAIRE qui ne l'est pas** ; aucun warmup ne converge vers un plateau inexistant.
> **La re-qualification n'est PAS acquise, et aucune mesure 3D ABSOLUE ne devrait être
> produite avant qu'elle le soit.** Reste probablement sain — **à établir, pas à
> supposer** — les rapports **intra-run entre points adjacents dans le temps** (dérive
> ~3 % sur 3 min ; §A53 mesurait 2D et 3D côte à côte, deux runs à 0,63 %).
> **Rien n'est rétracté** : aucune conclusion de gate n'en dépend.
>
> **§A62 — LE RENDU EST PLACÉ, et le dû de `:4142` (19/07) est INSTRUIT.** Aucune
> mesure, aucun GPU : tout est lu dans des artefacts ou **extrait du texte d'une
> ancre**. Quatre placements traités, y compris celui qui meurt (un seul GPU).
> **Le coût d'une image rendue `R` s'ANNULE de l'arbitrage de cadence** —
> sensibilité 1,3·10⁻¹⁵ — parce que 60 images sont rendues par seconde des deux
> côtés. Ce n'est pas une trivialité : au placement « rendu à la cadence
> physique », hors porte, la sensibilité vaut **0,72**. ⇒ **tout l'arbitrage de
> cadence vaut `Δ = 30·(non-F − I)/C`**, soit **+21,84 fenêtres·Hz (6,2 %) à
> `I` = 0** et **zéro au point mort `I` = 1,768 ms**, où `I` est le coût d'une
> interpolation de readout, inconnu et gravé « non gratuit ». **Le rendu décide
> le CÔTÉ** : `s_max` à 11 fenêtres et 60 Hz va de **51,99** (plancher MESURÉ du
> rendu : gather 0,449 + encodage 0,094) à **43,46** (réserve gravée 13 ms / 2
> images) — étendue 16,4 %, **exactement invariante à un biais sur `C`**, donc
> hors d'atteinte de la maladie de §A61. Témoin non planifié : le modèle
> **reproduit le 7,61 de §A57** (7,6148) par un autre chemin.
> ⚠ **L'INTERDICTION PRÉ-ÉCRITE DE P2 MORD ICI** (recopiée par la machine dans
> l'artefact) : *« le coût inconnu du rendu ne vit PAS dans l'encodage ; il vit
> dans la COMPOSITION et dans l'optique au-delà de Y = A. La dette se DÉPLACE.
> INTERDICTION d'en conclure "le rendu tient". »* Au plancher, le cap 60 Hz
> **tombe de 6,12 à 5,90** pour 11 fenêtres demandées. §A57-1 reste vraie sous son
> label « rendu non compté », §A57-3 est expliquée et non corrigée.
>
> **§A62-bis — QUATRE POINTS D'UNE RELECTURE ADVERSE, TOUS JUSTES.** §A62 n'est
> pas fausse, elle est **sous-tracée**, et deux de ses énoncés entrent au registre.
> **(1)** La prédiction (2) du prereg — point mort *à* non-F — a été **CONTREDITE**
> de 3,63 %, et §A62-3 a écrit « les trois prédictions tiennent » tout en publiant,
> trois lignes plus loin, deux énoncés incompatibles (« non-F → ≈ 0 » et « point
> mort 1,768 »). **La cause est l'arrondi, et c'est le mot que §A62 n'écrit
> jamais** : le corpus grave 16,7 (pas 1000/60), donc le côté 60 Hz vit dans une
> seconde de **1002 ms**. Sous `1000/f` le point mort tombe **exactement** sur non-F.
> **(2)** `I-r5` — « calculer des deux façons » — était **pré-écrite et NON
> MÉCANISÉE** ; exécutée depuis, avec branche INDÉTERMINÉE si elle diffère entre les
> deux. Elle établit que **l'arrondi ne touche que ce qui vit PAR SECONDE** (point
> mort et Δ : 3,63 % ; s_max 0,08 %, cap 0,23 %, étendue 0,28 %).
> **(3)** Le « témoin non planifié » de §A62-4 est une **identité** — mêmes trois
> nombres, mêmes opérations que §A57 — pas une reproduction indépendante. Retiré.
> **(4)** ⚠ **LE « ~67³ » EST LE COIN `I` = 0.** À 30 Hz le côté DÉPEND de `I` :
> **66,82** (I=0) · 66,14 (I=non-F/2) · **65,45** (au point mort). Le 60 Hz n'en
> dépend pas. Au point mort le troc est exact : 51,99³×60 ≈ 65,45³×30.
> **(5)** Volet manquant de `I-r4` : un biais ±15 % sur `R_plancher` déplace s_max
> de **0,19 %** — §A61 n'y mord pas. Calculé, plus supposé.
>
> **SUITE — TROIS CHOSES DEVANT, DANS CET ORDRE, ET LA PREMIÈRE INTERDIT DE MESURER.**
>
> **(1) NE PAS LANCER DE MESURE 3D ABSOLUE.** §A61 : l'instrument n'est pas qualifié —
> machine plafondée en puissance, pas de plateau, dérive monotone. Le prochain chiffre
> coûterait plus cher à interpréter qu'à produire. Ce qu'une re-qualification demandera :
> **rang d'exécution apparié** (mesures alternées ou répétées aux deux bouts),
> **non-stationnarité MESURÉE** (dérive du plateau contre température) et **estimateur
> robuste à la dérive** plutôt qu'une médiane sur population mouvante. Chacun des trois
> corrige une faute constatée. *Les rapports intra-run entre points adjacents restent
> plausibles — à ÉTABLIR, pas à supposer.*
>
> **Ce dû est INSTRUIT depuis le 22-23/08, et AUCUN run n'a eu lieu.** Le protocole existe :
> `pocPhysicator/claude/prereg-requalification-instrument-3d-v2-2026-08-22.md`, endossé en
> trois commits (`4a7b9a9` corps, `f9f57fc` et `ab954f6` amendements), et son driver
> `pocPhysicator/run_requalification_instrument_3d_v2.py` (`ec45510`, commit distinct — la v1
> est intacte, elle porte le protocole que §A61 a disqualifié). **Ne pas le réécrire : le lire.**
> Il renverse v1 — ne plus chercher un plateau qui n'existe pas, mais mesurer si un contraste
> apparié en rang est répétable MALGRÉ la dérive. L'interdiction ci-dessus **tient tant que le
> run n'a pas tourné**, et après lui : artefact, entrée au journal, ARRÊT — aucune branche ne
> s'enchaîne, Q-A moins que les autres.
>
> **(2) LES DEUX GRANDEURS QUE LA CADENCE ATTEND**, et **ce n'est plus le rendu** :
> **`I`**, le coût d'une interpolation de readout (point mort de l'arbitrage), et le
> **non-F 3D** — l'un des deux multiplicateurs dus de §A53, auquel `Δ` est proportionnel,
> et dont la valeur retenue (1,835 ms) est un **non-F 2D à l'échelle de l'instrument**
> (`PREREGISTRATION.md:9559`), pas un rendu 3D à 1920. Les deux passent **derrière (1)**.
>
> **(3) LA PORTE 3 DE §A58 — re-dériver la monnaie du slot si le côté quitte 64.**
> Toujours pas levée, et §A62 rend son échéance concrète : aucun des côtés lus n'est 64.
>
> **LA CADENCE APPARTIENT À ROMAIN, et sa forme est maintenant énonçable sans terme
> caché** : à travail quasi constant, **60 Hz avec des fenêtres de ~52³** ou **30 Hz avec
> des fenêtres de 66,8³ (`I` = 0) à 65,5³ (au point mort)** — résolution temporelle contre
> résolution spatiale, pour un écart de **6,2 % qui s'évapore si l'interpolation coûte un
> non-F**. V4 en demande **11**, et
> **aucun cap ne l'atteint** : 5,90 à 60 Hz au plancher mesuré du rendu, 7,61 à 30 Hz sous
> la réserve gravée.
> ⚠ Le « **12,97 à 30 Hz** » de §A55 est **SUPERSÉÉ par §A57** — il accordait à la
> physique les 33,3 ms entières. La ligne est au registre de supersessions.
> **Rien ne s'enchaîne.**

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

# Ancres — À LANCER AVANT D'ÉCRIRE TOUTE ENTRÉE OU TOUT PREREG QUI CITE (§A54).
# Balaie les DEUX dépôts ; sortie non nulle sur une ancre morte.
.venv/bin/python verifier_ancres.py                 # table lisible
.venv/bin/python verifier_ancres.py --json X        # artefact machine
.venv/bin/python verifier_ancres.py --nues          # liste aussi les ancres NUES

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
