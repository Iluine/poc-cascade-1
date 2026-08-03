#!/usr/bin/env python
"""VÉRIFICATEUR D'ANCRES — dû par §A50, réclamé par quatre consommateurs.

Ce que §A50 a établi : **13 des 19 ancres sortantes du journal étaient
fausses**, toutes nées dans le lot de commits qui les déplaçait. La règle
qui en sort — `` `fichier:NNN` « fragment exact » ``, **le texte fait foi,
le numéro est le chemin** — ne protège que si quelqu'un relit. Ce fichier
est ce quelqu'un.

Ce que §A51-7 a ajouté : un **troisième état**. Une ancre peut pointer
juste et citer un fait **superséé** par un verdict ultérieur — « citation
exacte, fait périmé ». Aucune vérification textuelle ne peut le voir : le
texte cité est bel et bien là. Il faut un REGISTRE DE SUPERSESSIONS,
déclaré une fois, relu à chaque passe. Sans lui, le corpus ment sans
qu'aucune machine ne s'en aperçoive — c'est arrivé deux fois en deux jours
sur la même valeur (la marge de V4), et la seconde fois le relecteur
humain l'a validée de bonne foi.

ÉTATS RENDUS :
  exacte           le fragment est bien à la ligne citée
  exacte-emphase   le fragment y est, au balisage markdown près (reporté)
  décalée→N′       le fragment existe, mais AILLEURS — le numéro a dérivé
  introuvable      le fragment a DISPARU : changement de fond, FAIL-LOUD
  périmée          ancre exacte, contenu superséé (registre) — §A51-7
  contestée        ancre exacte, contenu CONTREDIT ailleurs, non arbitré
  nue              ancre sans texte : INVÉRIFIABLE, c'est le danger de §A50
  fichier-absent   la cible n'existe pas

Une `introuvable` fait sortir en code non nul : le texte a disparu, donc
la substance a changé, et §A50 interdit de renuméroter en silence.

Usage :
    .venv/bin/python verifier_ancres.py            # table lisible
    .venv/bin/python verifier_ancres.py --json X   # artefact machine
    .venv/bin/python verifier_ancres.py --nues     # liste aussi les nues
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

RACINE = pathlib.Path(__file__).resolve().parent
RACINE_PHYSICATOR = RACINE.parent / "pocPhysicator"
JOURNAL = "PREREGISTRATION.md"

# Documents SCANNÉS à la recherche d'ancres. Les cibles, elles, peuvent
# être n'importe quel fichier des deux dépôts.
MOTIFS_SOURCES: tuple[str, ...] = ("*.md", "claude/*.md", "docs/*.md")

# ALIAS — le corpus abrège. TROUVÉ PAR LA PREMIÈRE PASSE : sans cette
# table, `SPEC:431` (5 occurrences), `spec-p1:269` (8) et `§A43:7266` (4)
# étaient JETÉS EN SILENCE par l'extracteur, faute d'extension reconnue.
# Une ancre ignorée est pire qu'une ancre fausse : elle ne se signale
# jamais. La faute exacte que §A51-7 consigne — la citation de `SPEC:431`,
# quadruplet mort — vit sous l'un de ces alias, et le vérificateur ne la
# voyait pas.
ALIAS: dict[str, str] = {
    "SPEC": "SPEC-FOVEA-Z.md",
    "spec-p1": "claude/spec-p1-rendu-instrument-2026-07-25.md",
}
# Un préfixe « §Axx » désigne une entrée du journal : la cible est le
# journal, le numéro reste le numéro de ligne.
MOTIF_SECTION = re.compile(r"^§[A-Za-z]?\d+")


# --------------------------------------------------------------------
# LE REGISTRE — déclaratif, vérifié sur disque avant d'être écrit
# --------------------------------------------------------------------
#
# Chaque entrée dit : telle PLAGE porte un fait superséé, par tel verdict,
# et voici EXACTEMENT ce qui est superséé (le reste de la plage peut être
# parfaitement valide — le vérificateur SURFACE, il ne tranche pas).

@dataclass(frozen=True)
class Supersession:
    """Une supersession se déclare par PLAGE **et** par FRAGMENTS.

    TROUVÉ PAR LA PREMIÈRE PASSE, corrigé aussitôt : déclarée par plage
    seule, l'entrée « modèle dense » (SPEC:109-148) marquait PÉRIMÉES des
    ancres parfaitement vivantes de la même section — la descente par
    l'énergie (:116-117), le streaming (:130). Un registre trop large
    fabrique de faux périmés, et un lecteur qui voit de faux périmés cesse
    de lire les vrais. La supersession ne mord donc que si la citation
    reprend l'un des `fragments` déclarés ; `fragments` vide = toute la
    plage (à n'employer que pour une section entièrement morte).
    """
    fichier: str
    ligne_debut: int
    ligne_fin: int
    portee: str              # ce qui est superséé, précisément
    superseant: str          # l'ancre du verdict qui supersède
    fragments: tuple[str, ...] = ()
    etat: str = "périmée"    # ou "contestée" (contradiction non arbitrée)


SUPERSESSIONS: tuple[Supersession, ...] = (
    Supersession(
        fichier="SPEC-FOVEA-Z.md", ligne_debut=425, ligne_fin=439,
        portee="le QUADRUPLET : §6-rev1 grave V2 (« 3 slots énergie c=8 »). "
               "V2 est MORT ; le quadruplet en vigueur est V4 "
               "(PREREGISTRATION.md:4214). Le reste de la section — la "
               "reformulation du gate (iii), σ_ω — n'est pas superséé.",
        superseant="PREREGISTRATION.md:3971 « §A16-lecture-M-a-ter "
                   "(2026-07-19) — MORT V2 »",
        fragments=("3 slots énergie", "quadruplet-jeu V2")),
    Supersession(
        fichier="PREREGISTRATION.md", ligne_debut=4353, ligne_fin=4400,
        portee="les CHIFFRES DE FRAME de M-a-quater : médiane 16.589, marge "
               "0.111 ms = 0,7 %, p99 19.652, transferts 2.727. Superséés "
               "le MÊME JOUR par le run EPS 1e-2 : médiane 14.490, marge "
               "2.199 ms = 13,2 %. L'attribution de l'AUTRE et la réserve "
               "p99 restent valides. C'est la valeur qui a été recopiée à "
               "tort dans §A51:8258 puis dans le prereg de la tranche.",
        superseant="PREREGISTRATION.md:5215 « marge **2.199 ms** (contre "
                   "0.111 auparavant) »",
        fragments=("16.589", "0.111", "0,7 %", "19.652", "2.727",
                   "V4 AU SEUIL")),
    Supersession(
        fichier="SPEC-FOVEA-Z.md", ligne_debut=109, ligne_fin=148,
        portee="le MODÈLE DE COÛT dense (« Cellules actives ≈ γ₂·n_fov²·"
               "N_niv, γ₂≈3 », :119). Remplacé par le CAP DUR "
               "D'EMPLACEMENTS. Toute enveloppe calculée avec la formule "
               "dense est un compte d'un autre modèle — non faux, mais "
               "d'un modèle superséé (précaution (i) de §A51-6).",
        superseant="PREREGISTRATION.md:3890 « L'enveloppe dense "
                   "(γ₂·n_fov²·N_niv) est remplacée par un **cap dur "
                   "d'emplacements** »",
        fragments=("γ₂·n_fov²·N_niv", "Cellules actives")),
    Supersession(
        fichier="SPEC-FOVEA-Z.md", ligne_debut=440, ligne_fin=452,
        etat="contestée",
        portee="CONTRADICTION INTERNE NON ARBITRÉE (§A51-7) : §2-rev1 dit "
               "« Candidat courant : V4 » tandis que §6-rev1, endossée le "
               "MÊME JOUR, grave V2. Deux sections endossées, deux "
               "quadruplets. Arbitrage Romain, NON PRIS — le vérificateur "
               "signale, il ne choisit pas.",
        superseant="PREREGISTRATION.md:8319 « Contradiction interne de la "
                   "spec, NOMMÉE, NON RÉSOLUE »",
        fragments=("Candidat courant",)),
)

# CITATIONS NON VERBATIM DÉCLARÉES — chacune vérifiée à la main, chacune
# avec sa raison. Ce n'est PAS une liste d'exceptions de confort : une
# citation déclarée reste affichée à chaque passe, dans sa propre section.
# Ce qu'elle change, c'est le code de sortie — pour qu'un problème NEUF
# ne se noie pas dans un problème connu. Un vérificateur qui échoue
# toujours n'est plus lu, et un vérificateur qu'on ne lit plus ne garde
# rien.
#
# LE DÉFAUT COMMUN AUX DEUX PREMIÈRES, qui mérite son nom : **la citation
# qui referme une parenthèse**. On tronque au milieu d'une incise et on
# ferme proprement — le fragment obtenu est bien formé, a l'air verbatim,
# et n'a JAMAIS existé dans la source. Ni un grep ni un lecteur ne le
# retrouvent. C'est une ancre de la famille §A50 : elle a l'apparence de
# la rigueur et n'en a pas la substance.
@dataclass(frozen=True)
class CitationDeclaree:
    source: str
    ligne_source: int
    cible: str
    raison: str


CITATIONS_NON_VERBATIM: tuple[CitationDeclaree, ...] = (
    CitationDeclaree(
        source="PREREGISTRATION.md", ligne_source=7976,
        cible="src/summary_quadtree.py",
        raison="§A50 cite « COPIE float64 (jamais une vue -- anti-fuite) » ; "
               "la source (:79) écrit « … -- anti-fuite : `summarize_qt` ne "
               "doit » — la citation FERME une parenthèse que la source "
               "n'y ferme pas. L'ancre pointe juste et le sens est intact ; "
               "le fragment, lui, est inventé. Consignée, non amendée : "
               "§A50 est une entrée de journal APPEND-ONLY."),
    CitationDeclaree(
        source="claude/prereg-tranche-cout-rendu-2026-08-03.md",
        ligne_source=61, cible="src/f1_gpu/backend.py",
        raison="même défaut : « (pas de mesure valide) » referme une "
               "parenthèse que la source (:6) poursuit en « (pas de mesure "
               "valide, kill 45 s) », plus une capitale initiale abaissée. "
               "Ancre juste, sens intact. Clause de PROVENANCE d'un "
               "pré-enregistrement daté : couverte en bloc, non amendée "
               "(régime §A47-PRÉCISION-2)."),
    CitationDeclaree(
        source="PREREGISTRATION.md", ligne_source=8178,
        cible="PREREGISTRATION.md",
        raison="troisième variante du même défaut : §A51 cite `:4221` "
               "« P4 — L1 EN RÉSERVE : cadence k=2 » là où la source écrit "
               "« **P4 — L1 EN RÉSERVE pré-enregistrée** : cadence k=2 si L3 "
               "ne suffit pas ». Un mot SUPPRIMÉ sans marque d'élision — la "
               "citation se resserre en silence. Ancre juste, sens intact, "
               "fragment inexistant. Entrée append-only : consignée."),
)


# Plages où des ancres périmées sont citées DÉLIBÉRÉMENT — tables de
# correction, entrées qui consignent une faute. Les y signaler serait
# crier au loup, et un vérificateur qu'on apprend à ignorer ne garde
# plus rien. Chaque exclusion nomme sa raison.
ZONES_NON_NORMATIVES: tuple[tuple[str, int, int, str], ...] = (
    ("PREREGISTRATION.md", 8867, 8900,
     "§A54-2 CONSIGNE les trois formes de citation non verbatim, donc il "
     "les CITE. Une entrée qui documente une faute doit pouvoir la "
     "reproduire sans que l'outil la lui reproche — et sans que le "
     "fragment fautif, désormais présent à deux endroits du journal, "
     "fasse croire à une ancre simplement décalée. Zone tenue AU PLUS "
     "SERRÉ : la seule section qui cite des fragments fautifs."),
)


# --------------------------------------------------------------------
# Extraction et résolution
# --------------------------------------------------------------------

# `fichier.ext:NNN` ou `fichier.ext:NNN-MMM` ou `:NNN` (journal implicite),
# suivi — ou non — d'un fragment entre guillemets français.
MOTIF_ANCRE = re.compile(
    r"`(?P<cible>[^`\s]*?):(?P<debut>\d+)(?:-(?P<fin>\d+))?`"
    r"(?P<suite>\s*«(?P<fragment>[^»]*)»)?",
    re.DOTALL)


def _cible_plausible(nom: str) -> bool:
    """Un préfixe désigne-t-il un fichier ? Les extensions connues, plus
    les noms sans extension qui existent réellement sur disque (`.gitignore`)."""
    if nom.endswith((".md", ".py", ".json", ".txt", ".jsonl", ".toml")):
        return True
    return any((racine / nom).exists()
               for racine in (RACINE, RACINE_PHYSICATOR))


def _normaliser(texte: str) -> str:
    """Aplatit les blancs (un fragment cité se replie sur plusieurs
    lignes du markdown source, la cible non)."""
    return re.sub(r"\s+", " ", texte).strip()


def _sans_emphase(texte: str) -> str:
    """Retire le balisage markdown d'emphase — une citation ajoute
    parfois un gras que la source n'a pas, et l'inverse."""
    return re.sub(r"[*_`]", "", texte)


@dataclass
class Ancre:
    source: str
    ligne_source: int
    cible: str
    debut: int
    fin: int | None
    fragment: str | None
    etat: str = ""
    detail: str = ""
    ligne_trouvee: int | None = None
    cible_heritee: bool = False
    racine_source: str = ""
    candidats: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        plage = f"{self.debut}" if self.fin is None else f"{self.debut}-{self.fin}"
        return f"{self.cible}:{plage}"


def extraire(chemin: pathlib.Path, relatif: str,
             racine_source: str = "") -> list[Ancre]:
    texte = chemin.read_text(encoding="utf-8")
    debuts_de_ligne = [0]
    for i, c in enumerate(texte):
        if c == "\n":
            debuts_de_ligne.append(i + 1)

    def ligne_de(offset: int) -> int:
        bas, haut = 0, len(debuts_de_ligne) - 1
        while bas < haut:
            milieu = (bas + haut + 1) // 2
            if debuts_de_ligne[milieu] <= offset:
                bas = milieu
            else:
                haut = milieu - 1
        return bas + 1

    ancres = []
    # RÉSOLUTION CONTEXTUELLE DES ANCRES NUES DE FICHIER (`:NNN`).
    # Le corpus les emploie de DEUX façons dans la même entrée — §A51
    # écrit `:4214` pour le journal et `:116-117` pour la spec, en
    # héritant du dernier fichier NOMMÉ. C'est un piège de la même
    # famille que §A50 : la même syntaxe désigne deux cibles selon ce
    # qui précède. On reproduit la règle d'héritage et on REPORTE la
    # cible inférée, pour qu'un lecteur puisse la contester.
    dernier_nomme = JOURNAL
    for m in MOTIF_ANCRE.finditer(texte):
        cible = m.group("cible")
        heritee = False
        if cible:
            if cible in ALIAS:
                cible = ALIAS[cible]
            elif MOTIF_SECTION.match(cible):
                cible = JOURNAL
            elif not _cible_plausible(cible):
                continue                  # `12:30` d'une horloge, etc.
            dernier_nomme = cible
        else:
            cible, heritee = dernier_nomme, True
        ancres.append(Ancre(
            source=relatif, ligne_source=ligne_de(m.start()),
            cible=cible, debut=int(m.group("debut")),
            fin=int(m.group("fin")) if m.group("fin") else None,
            fragment=m.group("fragment"), cible_heritee=heritee,
            racine_source=racine_source,
            candidats=([cible] if not heritee
                       else list(dict.fromkeys([cible, JOURNAL])))))
    return ancres


def _resoudre_cible(cible: str, racine_source: str = "") -> pathlib.Path | None:
    """Résout une cible en préférant LE DÉPÔT DU DOCUMENT CITANT.

    TROUVÉ PAR LA PREMIÈRE PASSE : `.gitignore` existe dans les DEUX
    dépôts. Le prereg 3D, qui vit dans pocPhysicator, citait `.gitignore:5`
    « `outputs/` » — résolu contre pocCascade2phys, le fragment était
    « introuvable ». Une ancre relative se lit depuis chez elle."""
    ordre = (RACINE, RACINE_PHYSICATOR)
    if racine_source == "physicator":
        ordre = (RACINE_PHYSICATOR, RACINE)
    for racine in ordre:
        chemin = racine / cible
        if chemin.exists():
            return chemin
        trouves = sorted(racine.rglob(pathlib.PurePath(cible).name))
        trouves = [t for t in trouves
                   if ".git" not in t.parts and ".venv" not in t.parts]
        if len(trouves) == 1:
            return trouves[0]
    return None


def _normaliser_typographie(texte: str) -> str:
    """Uniformise le séparateur décimal entre chiffres.

    TROUVÉ PAR LA PREMIÈRE PASSE : `PREREGISTRATION.md:4217` porte
    « Porte 33.3 = REPLI PRÉ-NOMMÉ » (point) ; §A51 puis §A53 l'ont cité
    « Porte 33,3 » (virgule française). L'ancre est JUSTE, la citation
    n'est pas verbatim. §A50 dit que le texte fait foi — une citation qui
    corrige la typographie de sa source affaiblit ce principe sans qu'on
    s'en aperçoive. On la détecte et on la NOMME, on ne l'absout pas.
    """
    return re.sub(r"(?<=\d)[.,](?=\d)", "\u00b7", texte)


# Les trois lectures, de la plus stricte à la plus permissive. Une
# correspondance obtenue par un mode permissif DANS la plage citée vaut
# mieux qu'une correspondance stricte HORS plage : la première dit « la
# citation n'est pas verbatim », la seconde dirait faussement « l'ancre
# a dérivé ».
MODES = (
    ("exacte", lambda s: s),
    ("exacte-emphase", _sans_emphase),
    ("exacte-typographie", lambda s: _normaliser_typographie(_sans_emphase(s))),
)


def _verifier_contre(ancre: Ancre, cible_nom: str) -> Ancre:
    if ancre.fragment is None:
        ancre.etat = "nue"
        ancre.detail = ("aucun texte : INVÉRIFIABLE — une ancre nue décalée "
                        "fait RÉUSSIR la vérification sur la mauvaise ligne")
        return ancre

    chemin = _resoudre_cible(cible_nom, ancre.racine_source)
    if chemin is None:
        ancre.etat = "fichier-absent"
        ancre.detail = f"cible introuvable sur disque : {cible_nom}"
        return ancre

    lignes = chemin.read_text(encoding="utf-8").splitlines()
    haut = ancre.fin if ancre.fin is not None else ancre.debut
    memes_fichiers = pathlib.PurePath(ancre.source).name == chemin.name

    def positions_pour(transforme):
        """Lignes où le fragment apparaît, sous une lecture donnée.

        LES CITATIONS À ÉLISION — trouvées par la première passe, dont
        elles faussaient deux « introuvable » et deux « auto-citation ».
        Le corpus écrit couramment « début […] fin » pour couper un
        passage. Cherché tel quel, un fragment élidé ne se trouve JAMAIS :
        il devient une ancre nue déguisée, qui a l'air vérifiée et ne
        l'est pas — précisément le danger que §A50 nomme. On découpe donc
        sur l'élision et on exige que chaque morceau apparaisse, DANS
        L'ORDRE, la ligne rendue étant celle du premier."""
        bornes, position, morceaux = [], 0, []
        for numero, ligne in enumerate(lignes, start=1):
            morceau = _normaliser(transforme(ligne))
            if morceau:
                bornes.append((position, numero))
                morceaux.append(morceau)
                position += len(morceau) + 1
        botte = " ".join(morceaux)
        brut = _normaliser(transforme(ancre.fragment))
        if not brut:
            return []
        parts = [p for p in (m.strip() for m in
                             re.split(r"\[\s*(?:…|\.\.\.)\s*\]|…", brut))
                 if p]
        if not parts:
            return []

        def ligne_de_offset(offset):
            numero = bornes[0][1] if bornes else 1
            for debut_offset, n in bornes:
                if debut_offset <= offset:
                    numero = n
                else:
                    break
            return numero

        trouvees = []
        for m in re.finditer(re.escape(parts[0]), botte):
            curseur = m.end()
            complet = True
            for suite in parts[1:]:
                suivant = botte.find(suite, curseur)
                if suivant < 0:
                    complet = False
                    break
                curseur = suivant + len(suite)
            if complet:
                trouvees.append(ligne_de_offset(m.start()))
        # Les zones qui CONSIGNENT des fautes en citent les fragments.
        # Les compter comme correspondances ferait passer une ancre morte
        # pour une ancre décalée — l'entrée qui documente le défaut
        # deviendrait la preuve qu'il n'existe pas.
        trouvees = [n for n in trouvees
                    if not _dans_zone_non_normative(chemin.name, n)]
        if memes_fichiers:
            hors_site = [n for n in trouvees
                         if abs(n - ancre.ligne_source) > 3]
            return hors_site if hors_site else ["SITE"]
        return trouvees

    hors_plage = None
    auto_seulement = False
    for nom_mode, transforme in MODES:
        trouvees = positions_pour(transforme)
        if trouvees == ["SITE"]:
            auto_seulement = True
            continue
        if not trouvees:
            continue
        # Tolérance d'UNE ligne : une citation peut démarrer en milieu de
        # phrase, donc sur la ligne précédant celle qu'on retient.
        dans_plage = [n for n in trouvees if ancre.debut - 1 <= n <= haut + 1]
        if dans_plage:
            ancre.etat = nom_mode
            ancre.ligne_trouvee = dans_plage[0]
            if nom_mode == "exacte-typographie":
                ancre.detail = ("ancre JUSTE, citation NON VERBATIM : le "
                                "séparateur décimal diffère de la source")
            elif nom_mode == "exacte-emphase":
                ancre.detail = ("ancre juste ; le balisage markdown de la "
                                "citation diffère de la source")
            return ancre
        if hors_plage is None:
            hors_plage = (nom_mode, trouvees[0])

    if hors_plage is not None:
        ancre.etat = "décalée"
        ancre.ligne_trouvee = hors_plage[1]
        ancre.detail = (f"le texte est en :{hors_plage[1]}, "
                        f"l'ancre dit :{ancre.debut}")
        return ancre
    if auto_seulement:
        ancre.etat = "auto-citation"
        ancre.detail = ("le fragment n'existe qu'au site de citation — "
                        "la cible ne le porte pas")
        return ancre

    ancre.etat = "introuvable"
    ancre.detail = ("le fragment a DISPARU de la cible — changement de fond, "
                    "jamais une renumérotation (§A50)")
    return ancre


def verifier(ancre: Ancre) -> Ancre:
    """Vérifie l'ancre contre ses cibles CANDIDATES.

    LE PIÈGE DE LA FORME NUE DE FICHIER (`:NNN`), trouvé par cette passe
    même : le corpus l'emploie de deux façons dans la MÊME entrée — §A51
    écrit `:4214` pour le journal et `:116-117` pour la spec. Ni « toujours
    le journal » ni « hériter du dernier nommé » n'est vrai partout. Le
    vérificateur ESSAIE les deux et NOMME la cible retenue ; si les deux
    résolvent, il le dit plutôt que de choisir en silence. C'est un danger
    de la famille §A50 que §A50 n'avait pas nommé.
    """
    if ancre.fragment is None or not ancre.candidats:
        return _verifier_contre(ancre, ancre.cible)
    essais = []
    for nom in ancre.candidats:
        essai = Ancre(source=ancre.source, ligne_source=ancre.ligne_source,
                      cible=nom, debut=ancre.debut, fin=ancre.fin,
                      fragment=ancre.fragment,
                      cible_heritee=ancre.cible_heritee,
                      racine_source=ancre.racine_source,
                      candidats=ancre.candidats)
        essais.append(_verifier_contre(essai, nom))
    bons = [e for e in essais if e.etat.startswith("exacte")]
    if bons:
        retenu = bons[0]
        if len(bons) > 1 and ancre.cible_heritee:
            retenu.notes.append(
                "ancre NUE DE FICHIER résolue par plusieurs cibles "
                f"({', '.join(b.cible for b in bons)}) — ambiguë")
    else:
        ordre = {"décalée": 0, "auto-citation": 1, "introuvable": 2,
                 "fichier-absent": 3}
        retenu = sorted(essais, key=lambda e: ordre.get(e.etat, 9))[0]
    if ancre.cible_heritee:
        retenu.notes.append(f"cible INFÉRÉE (`:NNN` nu) -> {retenu.cible}")
    return retenu


def _dans_zone_non_normative(nom_fichier: str, ligne: int) -> bool:
    for fichier, debut, fin, _ in ZONES_NON_NORMATIVES:
        if (pathlib.PurePath(fichier).name == nom_fichier
                and debut <= ligne <= fin):
            return True
    return False


def _est_declaree(ancre: Ancre) -> CitationDeclaree | None:
    for c in CITATIONS_NON_VERBATIM:
        if (ancre.source == c.source
                and abs(ancre.ligne_source - c.ligne_source) <= 2
                and pathlib.PurePath(ancre.cible).name
                == pathlib.PurePath(c.cible).name):
            return c
    return None


def appliquer_registre(ancre: Ancre) -> Ancre:
    declaree = _est_declaree(ancre)
    if declaree is not None and ancre.etat in ("introuvable", "auto-citation"):
        ancre.notes.append(f"DÉCLARÉE — {declaree.raison}")
        ancre.etat = "non-verbatim-déclarée"
        return ancre
    """Le troisième et le quatrième état — indétectables par le texte."""
    if not (ancre.etat.startswith("exacte") or ancre.etat == "décalée"):
        return ancre
    for fichier, debut, fin, raison in ZONES_NON_NORMATIVES:
        if (pathlib.PurePath(fichier).name
                == pathlib.PurePath(ancre.source).name
                and debut <= ancre.ligne_source <= fin):
            ancre.notes.append(f"ZONE NON NORMATIVE — {raison}")
            if ancre.etat in ("introuvable", "auto-citation", "décalée"):
                ancre.etat = "non-verbatim-déclarée"
            return ancre
    ligne = ancre.ligne_trouvee or ancre.debut
    cite = _normaliser(_sans_emphase(ancre.fragment or ""))
    for s in SUPERSESSIONS:
        if not (pathlib.PurePath(ancre.cible).name == pathlib.PurePath(
                s.fichier).name and s.ligne_debut <= ligne <= s.ligne_fin):
            continue
        if s.fragments and not any(
                _normaliser(_sans_emphase(f)) in cite for f in s.fragments):
            continue
        if True:
            ancre.notes.append(
                f"{s.etat.upper()} — {s.portee} | voir {s.superseant}")
            ancre.etat = s.etat
    return ancre


def garde_registre() -> list[str]:
    """Le registre lui-même est un texte : il peut pourrir. Chaque
    supersession déclare un superséant sous forme d'ancre — on la
    vérifie. Un registre faux est pire qu'un registre vide."""
    problemes = []
    for s in SUPERSESSIONS:
        for ancre in extraire_depuis_texte(s.superseant, "REGISTRE"):
            resultat = verifier(ancre)
            if not resultat.etat.startswith("exacte"):
                problemes.append(
                    f"registre : superséant {resultat.citation} "
                    f"-> {resultat.etat} ({resultat.detail})")
        cible = _resoudre_cible(s.fichier)
        if cible is None:
            problemes.append(f"registre : fichier superséé absent {s.fichier}")
            continue
        n_lignes = len(cible.read_text(encoding="utf-8").splitlines())
        if s.ligne_fin > n_lignes:
            problemes.append(
                f"registre : plage {s.fichier}:{s.ligne_debut}-{s.ligne_fin} "
                f"dépasse le fichier ({n_lignes} lignes)")
    return problemes


def extraire_depuis_texte(texte: str, source: str) -> list[Ancre]:
    ancres = []
    for m in MOTIF_ANCRE.finditer(texte):
        cible = m.group("cible") or JOURNAL
        cible = ALIAS.get(cible, cible)
        if MOTIF_SECTION.match(cible):
            cible = JOURNAL
        if not _cible_plausible(cible):
            continue
        ancres.append(Ancre(source=source, ligne_source=0, cible=cible,
                            debut=int(m.group("debut")),
                            fin=int(m.group("fin")) if m.group("fin") else None,
                            fragment=m.group("fragment")))
    return ancres


# --------------------------------------------------------------------

def collecter() -> list[pathlib.Path]:
    fichiers: list[pathlib.Path] = []
    for racine in (RACINE, RACINE_PHYSICATOR):
        if not racine.exists():
            continue
        for motif in MOTIFS_SOURCES:
            fichiers.extend(sorted(racine.glob(motif)))
    return [f for f in fichiers if ".venv" not in f.parts]


def main() -> int:
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--json", type=pathlib.Path)
    analyseur.add_argument("--nues", action="store_true",
                           help="lister aussi les ancres sans texte")
    options = analyseur.parse_args()

    problemes_registre = garde_registre()
    if problemes_registre:
        for p in problemes_registre:
            print(f"REGISTRE CASSÉ : {p}")
        return 2

    resultats: list[Ancre] = []
    for chemin in collecter():
        relatif = str(chemin.relative_to(
            RACINE if RACINE in chemin.parents else RACINE_PHYSICATOR))
        racine_source = ("physicator" if RACINE_PHYSICATOR in chemin.parents
                         or chemin.parent == RACINE_PHYSICATOR else "cascade")
        for ancre in extraire(chemin, relatif, racine_source):
            resultats.append(appliquer_registre(verifier(ancre)))

    comptes: dict[str, int] = {}
    for a in resultats:
        comptes[a.etat] = comptes.get(a.etat, 0) + 1

    print(f"{len(resultats)} ancres dans {len(collecter())} documents\n")
    for etat in ("introuvable", "auto-citation", "décalée",
                 "fichier-absent", "périmée", "contestée",
                 "exacte-typographie", "non-verbatim-déclarée",
                 "exacte-emphase", "exacte", "nue"):
        n = comptes.get(etat, 0)
        if n:
            print(f"  {etat:16s} {n:4d}")

    for etat in ("introuvable", "auto-citation", "fichier-absent",
                 "décalée", "exacte-typographie",
                 "non-verbatim-déclarée", "périmée", "contestée"):
        concernees = [a for a in resultats if a.etat == etat]
        if not concernees:
            continue
        print(f"\n=== {etat.upper()} ({len(concernees)}) ===")
        for a in concernees:
            print(f"  {a.source}:{a.ligne_source} -> `{a.citation}`")
            if a.detail:
                print(f"      {a.detail}")
            for note in a.notes:
                print(f"      {note}")
            if a.fragment:
                extrait = _normaliser(a.fragment)[:110]
                print(f"      « {extrait}{'…' if len(extrait) == 110 else ''} »")

    if options.nues:
        nues = [a for a in resultats if a.etat == "nue"]
        print(f"\n=== NUES ({len(nues)}) — invérifiables par construction ===")
        for a in nues:
            print(f"  {a.source}:{a.ligne_source} -> `{a.citation}`")

    if options.json:
        options.json.write_text(json.dumps(
            {"comptes": comptes,
             "ancres": [{"source": a.source, "ligne_source": a.ligne_source,
                         "citation": a.citation, "etat": a.etat,
                         "ligne_trouvee": a.ligne_trouvee,
                         "detail": a.detail, "notes": a.notes}
                        for a in resultats]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nartefact : {options.json}")

    fatales = (comptes.get("introuvable", 0)
               + comptes.get("fichier-absent", 0)
               + comptes.get("auto-citation", 0))
    if fatales:
        print(f"\nÉCHEC : {fatales} ancre(s) dont le texte a disparu — "
              "changement de fond, jamais une renumérotation silencieuse.")
    return 1 if fatales else 0


if __name__ == "__main__":
    sys.exit(main())
