"""Verrous du VÉRIFICATEUR D'ANCRES.

Le vérificateur est un instrument de chemin-verdict : c'est lui qui dira
si une entrée du journal cite juste. Un vérificateur faux est pire que pas
de vérificateur — il délivre un certificat de rigueur à un corpus qui
ment. Ces tests l'attaquent donc dans les deux sens :

  - il doit DIRE OUI quand l'ancre est bonne (ne pas fabriquer d'alarme,
    faute qui apprend à ignorer l'outil) ;
  - il doit DIRE NON quand l'ancre est fausse, y compris sous chacune des
    formes de fausseté qu'il a lui-même exhumées du corpus le 2026-08-03 :
    citation qui referme une parenthèse, mot supprimé sans marque
    d'élision, séparateur décimal « corrigé », alias non résolu, cible
    ambiguë entre deux dépôts.

Règle appliquée (§A53) : *un verrou numérique ne garde que ce que son état
allume.* Chaque cas est donc monté sur un corpus jetable qui contient
exactement le piège visé."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import verifier_ancres as va  # noqa: E402


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Un faux dépôt, cible et source, isolé du vrai."""
    monkeypatch.setattr(va, "RACINE", tmp_path)
    monkeypatch.setattr(va, "RACINE_PHYSICATOR", tmp_path / "autre")
    (tmp_path / "autre").mkdir()
    return tmp_path


def _etats(corpus, source: str, cible_nom: str, cible_texte: str):
    (corpus / cible_nom).write_text(cible_texte, encoding="utf-8")
    doc = corpus / "doc.md"
    doc.write_text(source, encoding="utf-8")
    return [va.appliquer_registre(va.verifier(a))
            for a in va.extraire(doc, "doc.md", "cascade")]


CIBLE = "cible.md"


# ----------------------------------------------------------------- OUI

def test_ancre_exacte(corpus):
    etats = _etats(corpus, "voir `cible.md:2` « le texte vrai »",
                   CIBLE, "ligne un\nle texte vrai\nligne trois\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_fragment_replie_sur_plusieurs_lignes(corpus):
    """Une citation du markdown se replie ; la cible non. L'aplatissement
    des blancs est ce qui rend la vérification possible du tout."""
    etats = _etats(corpus, "voir `cible.md:2` « le texte\n  vrai et long »",
                   CIBLE, "ligne un\nle texte vrai et long\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_fragment_a_travers_un_bloc_cite(corpus):
    """Le journal cite l'essentiel de sa matière dans des blocs `>`. Sans
    retirer le marqueur, un fragment replié sur deux lignes d'un bloc
    devient « … avant la > décision … » et ne se retrouve jamais : le
    vérificateur sous-détecte là où le corpus est le plus dense, et rend
    « introuvable » des ancres justes. C'est la pire des deux erreurs —
    celle qui apprend à ignorer l'outil."""
    etats = _etats(corpus, "voir `cible.md:2` « le moins cher qui échoue »",
                   CIBLE, "x\n> le moins cher\n> qui échoue d'abord\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_citation_a_elision(corpus):
    """Le corpus écrit « début […] fin ». Cherché tel quel, un fragment
    élidé ne se trouve JAMAIS : il devient une ancre nue déguisée."""
    etats = _etats(corpus, "voir `cible.md:2` « début […] fin »",
                   CIBLE, "x\ndébut du passage coupé puis la fin\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_elision_ne_tolere_pas_le_desordre(corpus):
    """Les morceaux doivent apparaître DANS L'ORDRE : sans quoi
    « A […] B » validerait un texte qui dit B puis A."""
    etats = _etats(corpus, "voir `cible.md:2` « zèbre […] abeille »",
                   CIBLE, "x\nabeille puis zèbre\n")
    assert [a.etat for a in etats] == ["introuvable"]


# ----------------------------------------------------------------- NON

def test_ancre_decalee(corpus):
    etats = _etats(corpus, "voir `cible.md:5` « le texte vrai »",
                   CIBLE, "le texte vrai\nautre\nautre\nautre\nautre\n")
    assert etats[0].etat == "décalée"
    assert etats[0].ligne_trouvee == 1


def test_la_tolerance_est_d_une_ligne_et_pas_de_deux(corpus):
    """Une tolérance existe — un fragment multi-lignes commence parfois
    sur la ligne d'avant. Elle vaut UNE ligne. Testée aux deux bords :
    une tolérance qu'on ne borne pas devient une absence de vérification."""
    cible = "un\ndeux\nle texte vrai\nquatre\ncinq\n"
    assert _etats(corpus, "voir `cible.md:4` « le texte vrai »",
                  CIBLE, cible)[0].etat == "exacte"
    assert _etats(corpus, "voir `cible.md:5` « le texte vrai »",
                  CIBLE, cible)[0].etat == "décalée"


def test_texte_disparu(corpus):
    etats = _etats(corpus, "voir `cible.md:2` « ce qui n'existe plus »",
                   CIBLE, "un\ndeux\n")
    assert [a.etat for a in etats] == ["introuvable"]


def test_citation_qui_referme_une_parenthese(corpus):
    """LE DÉFAUT TROUVÉ DANS LE CORPUS (deux occurrences, §A50 et le
    prereg de la tranche) : on tronque au milieu d'une incise et on ferme
    proprement. Le fragment est bien formé, a l'air verbatim, et n'a
    jamais existé. C'est la fausseté la plus difficile à voir à l'œil."""
    etats = _etats(corpus, "voir `cible.md:2` « COPIE float64 (anti-fuite) »",
                   CIBLE, "x\nCOPIE float64 (anti-fuite : la suite compte)\n")
    assert [a.etat for a in etats] == ["introuvable"]


def test_mot_supprime_sans_marque_d_elision(corpus):
    """Troisième variante trouvée dans le corpus (§A51 citant `:4221`) :
    la citation se resserre en silence."""
    etats = _etats(corpus, "voir `cible.md:2` « L1 EN RÉSERVE : cadence »",
                   CIBLE, "x\nL1 EN RÉSERVE pré-enregistrée : cadence k=2\n")
    assert [a.etat for a in etats] == ["introuvable"]


def test_separateur_decimal_corrige(corpus):
    """L'ancre est JUSTE, la citation ne l'est pas. On ne l'absout pas :
    on la nomme, dans un état à part."""
    etats = _etats(corpus, "voir `cible.md:2` « Porte 33,3 = REPLI »",
                   CIBLE, "x\nPorte 33.3 = REPLI PRÉ-NOMMÉ\n")
    assert [a.etat for a in etats] == ["exacte-typographie"]


def test_la_plage_prime_sur_la_lecture_stricte(corpus):
    """Une correspondance PERMISSIVE dans la plage citée vaut mieux qu'une
    correspondance STRICTE hors plage — sans quoi le vérificateur dirait
    « décalée » d'une ancre juste dont seule la typographie diffère.
    C'est le bug qui masquait les vraies découvertes à la première passe."""
    cible = "x\nPorte 33.3 = REPLI\n" + "\n" * 8 + "Porte 33,3 = REPLI\n"
    etats = _etats(corpus, "voir `cible.md:2` « Porte 33,3 = REPLI »",
                   CIBLE, cible)
    assert etats[0].etat == "exacte-typographie"
    assert etats[0].ligne_trouvee == 2


def test_ancre_nue_est_signalee_comme_inverifiable(corpus):
    etats = _etats(corpus, "voir `cible.md:2` sans texte",
                   CIBLE, "un\ndeux\n")
    assert [a.etat for a in etats] == ["nue"]
    assert "INVÉRIFIABLE" in etats[0].detail


def test_auto_citation_ne_valide_pas(corpus):
    """Quand la citation vit dans le fichier qu'elle cite, le fragment s'y
    trouve forcément — au site de citation. Le compter validerait toute
    ancre interne, fausse comprise."""
    doc = corpus / "doc.md"
    doc.write_text("ligne\nligne\nligne\nvoir `doc.md:1` « un texte absent »\n",
                   encoding="utf-8")
    etats = [va.appliquer_registre(va.verifier(a))
             for a in va.extraire(doc, "doc.md", "cascade")]
    assert [a.etat for a in etats] == ["auto-citation"]


# ------------------------------------------------------- alias et cibles

def test_alias_du_corpus_resolus(corpus, monkeypatch):
    """Sans table d'alias, `SPEC:431` était JETÉ EN SILENCE — 26 ancres
    invisibles sur 184. Une ancre ignorée ne se signale jamais : c'est
    pire qu'une ancre fausse."""
    monkeypatch.setitem(va.ALIAS, "SPEC", CIBLE)
    etats = _etats(corpus, "voir `SPEC:2` « quadruplet V2 »",
                   CIBLE, "x\nquadruplet V2 gravé\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_prefixe_de_section_designe_le_journal(corpus):
    etats = _etats(corpus, "voir `§A43:2` « une clause »",
                   va.JOURNAL, "x\nune clause du journal\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_horodatage_n_est_pas_une_ancre(corpus):
    etats = _etats(corpus, "il est `23:40` — horloge lue",
                   CIBLE, "rien\n")
    assert etats == []


# --------------------------------------------------- registre (§A51-7)

def test_supersession_mord_sur_le_fragment_declare(corpus, monkeypatch):
    monkeypatch.setattr(va, "SUPERSESSIONS", (
        va.Supersession(fichier=CIBLE, ligne_debut=1, ligne_fin=9,
                        portee="le quadruplet", superseant="J:1 « MORT V2 »",
                        fragments=("3 slots énergie",)),))
    etats = _etats(corpus, "voir `cible.md:2` « 3 slots énergie c=8 »",
                   CIBLE, "x\n3 slots énergie c=8\n")
    assert [a.etat for a in etats] == ["périmée"]


def test_supersession_ne_salit_pas_ses_voisines(corpus, monkeypatch):
    """LE FAUX POSITIF TROUVÉ À LA PREMIÈRE PASSE : déclarée par plage
    seule, l'entrée « modèle dense » marquait périmées la descente par
    l'énergie et le streaming, tous deux vivants. Un lecteur qui voit de
    faux périmés cesse de lire les vrais."""
    monkeypatch.setattr(va, "SUPERSESSIONS", (
        va.Supersession(fichier=CIBLE, ligne_debut=1, ligne_fin=9,
                        portee="le modèle dense", superseant="J:1 « cap »",
                        fragments=("γ₂·n_fov²·N_niv",)),))
    etats = _etats(corpus, "voir `cible.md:2` « l'énergie décide »",
                   CIBLE, "x\nl'énergie décide du raffinement\n")
    assert [a.etat for a in etats] == ["exacte"]


def test_le_registre_se_verifie_lui_meme():
    """Un registre faux est pire qu'un registre vide : il certifie des
    supersessions qui n'existent pas. Chaque superséant est une ancre —
    on la vérifie contre le vrai corpus."""
    assert va.garde_registre() == []


def test_le_corpus_reel_ne_porte_aucune_ancre_morte():
    """Le verrou d'intégration : aucune ancre `introuvable` non déclarée
    dans les deux dépôts. Un texte disparu est un changement de FOND
    (§A50) — jamais une renumérotation silencieuse."""
    morts = []
    for chemin in va.collecter():
        racine = ("physicator"
                  if va.RACINE_PHYSICATOR in chemin.parents
                  or chemin.parent == va.RACINE_PHYSICATOR else "cascade")
        relatif = str(chemin.relative_to(
            va.RACINE if va.RACINE in chemin.parents
            else va.RACINE_PHYSICATOR))
        for ancre in va.extraire(chemin, relatif, racine):
            resultat = va.appliquer_registre(va.verifier(ancre))
            if resultat.etat in ("introuvable", "auto-citation",
                                 "fichier-absent"):
                morts.append(f"{resultat.source}:{resultat.ligne_source} "
                             f"-> {resultat.citation} ({resultat.etat})")
    assert not morts, "ancres mortes non déclarées :\n" + "\n".join(morts)
