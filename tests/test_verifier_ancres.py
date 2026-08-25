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


# ------------------------------------------- renvois par NOM DE TEST
#
# Trou d'outillage remonté à la clôture de la boucle de rendu : le corpus
# cite des VERROUS par leur nom de fonction, et le vérificateur n'en voyait
# aucun. C'est la faute de §A62-bis-2 sous sa forme la plus discrète — un
# document qui cite un test supprimé promet une garde que rien ne tient,
# et la promesse a l'apparence de la preuve.
#
# Les cas ci-dessous sont montés sur des mini-dépôts jetables, jamais sur
# le corpus vivant : un verrou qui n'allume que ce qui se trouve déjà là
# ne garde rien le jour où ça change (§A53).

def _depot_de_test(corpus, doc: str, fichiers: dict[str, str]):
    """Un mini-dépôt : un document citant, et des sources Python."""
    for nom, contenu in fichiers.items():
        chemin = corpus / nom
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(contenu, encoding="utf-8")
    (corpus / "doc.md").write_text(doc, encoding="utf-8")
    index, echecs = va.indexer_tests()
    renvois = va.extraire_renvois_tests(corpus / "doc.md", "doc.md")
    return [va.verifier_renvoi_test(r, index) for r in renvois], echecs


UN_TEST = "def test_le_verrou_qui_existe():\n    assert True\n"


@pytest.fixture
def declarer(monkeypatch):
    """Pose un registre de renommages jetable, sans toucher au vrai."""
    def _poser(ligne_debut, ligne_fin, nom_cite="test_ancien_nom",
               nom_livre="test_le_verrou_qui_existe"):
        monkeypatch.setattr(va, "RENVOIS_TESTS_DECLARES", (
            va.RenvoiTestDeclare(
                source="doc.md", ligne_debut=ligne_debut,
                ligne_fin=ligne_fin, nom_cite=nom_cite,
                nom_livre=nom_livre, raison="montage de test"),))
    return _poser


def test_renvoi_vers_un_verrou_vivant(corpus):
    renvois, _ = _depot_de_test(
        corpus, "le verrou `test_le_verrou_qui_existe` tient.",
        {"tests/test_chose.py": UN_TEST})
    assert [r.etat for r in renvois] == ["test-vivant"]


def test_renvoi_vers_un_verrou_supprime_est_fatal(corpus):
    """LE CAS QUI JUSTIFIE L'OUTIL. Un document qui cite un test disparu
    affirme une garde absente — et jusqu'ici rien ne pouvait le voir."""
    renvois, _ = _depot_de_test(
        corpus, "le verrou `test_supprime_l_an_dernier` tient.",
        {"tests/test_chose.py": UN_TEST})
    assert [r.etat for r in renvois] == ["test-introuvable"]
    assert "garde promise" in renvois[0].detail


def test_le_renvoi_mort_fait_sortir_en_code_non_nul(corpus, monkeypatch):
    """La fatalité est le seul organe qui MORD. Sans elle l'outil rend un
    joli rapport que personne ne lit et que la CI laisse passer."""
    (corpus / "tests").mkdir()
    (corpus / "tests" / "test_chose.py").write_text(UN_TEST, encoding="utf-8")
    (corpus / "doc.md").write_text("verrou `test_disparu`\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verifier_ancres.py"])
    monkeypatch.setattr(va, "RENVOIS_TESTS_DECLARES", ())
    monkeypatch.setattr(va, "SUPERSESSIONS", ())
    assert va.main() == 1


def test_le_corpus_sain_sort_en_zero(corpus, monkeypatch):
    """Le contre-verrou : un outil qui échoue toujours n'est plus lu."""
    (corpus / "tests").mkdir()
    (corpus / "tests" / "test_chose.py").write_text(UN_TEST, encoding="utf-8")
    (corpus / "doc.md").write_text("verrou `test_le_verrou_qui_existe`\n",
                                   encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verifier_ancres.py"])
    monkeypatch.setattr(va, "RENVOIS_TESTS_DECLARES", ())
    monkeypatch.setattr(va, "SUPERSESSIONS", ())
    assert va.main() == 0


def test_renvoi_de_famille_resolu_par_prefixe(corpus):
    """`test_construire_halo_…` désigne une famille. Le chercher tel quel
    rendrait « introuvable » un renvoi juste — l'erreur qui apprend à
    ignorer l'outil, la pire des deux."""
    renvois, _ = _depot_de_test(
        corpus, "voir `test_construire_halo_…` pour l'assemblage.",
        {"tests/test_halo.py":
         "def test_construire_halo_interieur_fin():\n    assert True\n"})
    assert [r.etat for r in renvois] == ["test-famille"]
    assert renvois[0].resolu == "test_construire_halo_interieur_fin"


def test_famille_sans_aucun_membre_reste_fatale(corpus):
    """Le marqueur de famille n'est pas un laissez-passer : une famille
    entièrement supprimée est aussi morte qu'un nom seul."""
    renvois, _ = _depot_de_test(
        corpus, "voir `test_famille_eteinte_…`.",
        {"tests/test_halo.py": UN_TEST})
    assert [r.etat for r in renvois] == ["test-introuvable"]


def test_forme_abregee_resolue_dans_la_famille_de_son_appui(corpus):
    """`test_c_property_terrain_reel_plein` / `_front_wet_dry` — le corpus
    n'en porte qu'UNE, mais une ancre ignorée ne se signale jamais."""
    renvois, _ = _depot_de_test(
        corpus, "`test_c_property_plein` / `_front_wet_dry` : idem",
        {"tests/test_c.py":
         "def test_c_property_plein():\n    assert True\n\n"
         "def test_c_property_front_wet_dry():\n    assert True\n"})
    assert [r.etat for r in renvois] == ["test-vivant", "test-abrégé"]
    assert renvois[1].resolu == "test_c_property_front_wet_dry"


def test_l_abrege_n_attrape_pas_hors_de_sa_famille(corpus):
    """Sans la condition de préfixe commun, `_front_wet_dry` capterait
    n'importe quel test finissant pareil : une résolution qui a l'air
    d'en être une, donc pire que pas de résolution."""
    renvois, _ = _depot_de_test(
        corpus, "`test_c_property_plein` / `_front_wet_dry` : idem",
        {"tests/test_c.py":
         "def test_c_property_plein():\n    assert True\n\n"
         "def test_autre_chose_front_wet_dry():\n    assert True\n"})
    assert renvois[1].etat == "test-introuvable"


def test_un_jeton_souligne_sans_appui_n_est_pas_un_renvoi(corpus):
    """Le corpus porte 30 jetons backtickés en `_` : 29 sont des helpers
    et des constantes. Les lire comme des tests noierait le seul vrai."""
    renvois, _ = _depot_de_test(
        corpus, "la constante `_MODULES_ETAT` et le helper `_rhs_o2`.",
        {"tests/test_chose.py": UN_TEST})
    assert renvois == []


def test_un_nom_de_fichier_n_est_pas_un_renvoi_de_fonction(corpus):
    renvois, _ = _depot_de_test(
        corpus, "voir `tests/test_chose.py` et `test_chose.py`.",
        {"tests/test_chose.py": UN_TEST})
    assert renvois == []


def test_defini_mais_non_collecte_n_est_pas_vivant(corpus):
    """« Exister » ne veut pas dire « être défini ». Un `def test_x` que
    pytest ne ramasse pas ne tourne jamais : le dire vivant ferait de cet
    outil la faute qu'il traque, appliquée à lui-même."""
    renvois, _ = _depot_de_test(
        corpus, "le verrou `test_le_verrou_qui_existe` tient.",
        {"outils/aide.py": UN_TEST})
    assert [r.etat for r in renvois] == ["test-non-collecté"]


def test_source_python_illisible_est_remontee_pas_avalee(corpus):
    """FAIL-LOUD : un fichier de tests imparsable rend ses définitions
    invisibles. Avalée, la panne d'outil se traduirait en accusation
    FATALE contre des verrous bien vivants."""
    (corpus / "tests").mkdir()
    (corpus / "tests" / "test_casse.py").write_text(
        "def test_x(:\n", encoding="utf-8")
    _, echecs = va.indexer_tests()
    assert echecs and "illisible" in echecs[0]


def test_declaration_de_renommage_couvre_toute_sa_plage(corpus, declarer):
    """Le plan cite le nom mort DEUX fois à onze lignes d'écart — la note
    qui le désavoue et le bloc de code qu'elle désavoue. Déclarée au
    point, la seconde sortait en FATAL : l'outil reprochait au document
    exactement ce que le document consigne."""
    declarer(ligne_debut=1, ligne_fin=3)
    renvois, _ = _depot_de_test(
        corpus,
        "note : `test_ancien_nom` est devenu `test_le_verrou_qui_existe`\n"
        "\n"
        "def test_ancien_nom():\n",
        {"tests/test_chose.py": UN_TEST})
    assert [r.etat for r in renvois] == [
        "test-déclaré", "test-vivant", "test-déclaré"]


def test_declaration_dont_le_nom_livre_n_existe_pas_casse_le_registre(
        corpus, declarer):
    declarer(ligne_debut=1, ligne_fin=3, nom_livre="test_jamais_ecrit")
    (corpus / "tests").mkdir()
    (corpus / "tests" / "test_chose.py").write_text(UN_TEST, encoding="utf-8")
    (corpus / "doc.md").write_text("x\n", encoding="utf-8")
    index, echecs = va.indexer_tests()
    problemes = va.garde_renvois_tests(index, echecs)
    assert any("nom livré INEXISTANT" in p for p in problemes)


def test_declaration_perimee_casse_le_registre(corpus, declarer):
    """Un registre qui ne se nettoie pas finit par mentir : si le nom
    excusé ressuscite, la déclaration le couvre pour rien."""
    declarer(ligne_debut=1, ligne_fin=3,
             nom_cite="test_le_verrou_qui_existe")
    (corpus / "tests").mkdir()
    (corpus / "tests" / "test_chose.py").write_text(UN_TEST, encoding="utf-8")
    (corpus / "doc.md").write_text("x\n", encoding="utf-8")
    index, echecs = va.indexer_tests()
    problemes = va.garde_renvois_tests(index, echecs)
    assert any("PÉRIMÉE" in p for p in problemes)


def test_le_registre_des_renvois_se_verifie_lui_meme():
    """Sur le VRAI corpus : les deux renommages déclarés pointent vers des
    tests qui existent, et les noms qu'ils excusent n'existent plus."""
    index, echecs = va.indexer_tests()
    assert va.garde_renvois_tests(index, echecs) == []


def test_le_corpus_reel_ne_promet_aucun_verrou_absent():
    """Le verrou d'intégration du versant renvois."""
    index, _ = va.indexer_tests()
    morts = []
    for chemin in va.collecter():
        relatif = str(chemin.relative_to(
            va.RACINE if va.RACINE in chemin.parents
            else va.RACINE_PHYSICATOR))
        for renvoi in va.extraire_renvois_tests(chemin, relatif):
            resultat = va.verifier_renvoi_test(renvoi, index)
            if resultat.etat == "test-introuvable":
                morts.append(f"{resultat.source}:{resultat.ligne_source} "
                             f"-> {resultat.nom}")
    assert not morts, ("renvois vers un verrou inexistant :\n"
                       + "\n".join(morts))


def test_le_repli_module_ne_regarde_pas_dans_venv(corpus):
    """UN FATAL MASQUÉ PAR UNE DÉPENDANCE. Le repli « ce n'est pas une
    fonction, c'est un FICHIER de tests » cherchait sur tout le disque du
    dépôt : un `site-packages/…/test_utils.py` aurait excusé en NON FATAL
    un renvoi mort vers `test_utils`. Jamais déclenché à ce jour — un
    verrou se juge sur ce qu'il attrape le jour où ça arrive."""
    (corpus / ".venv").mkdir()
    (corpus / ".venv" / "test_utils.py").write_text(
        "def test_dune_dependance():\n    assert True\n", encoding="utf-8")
    renvois, _ = _depot_de_test(corpus, "le verrou `test_utils` tient.",
                                {"tests/test_chose.py": UN_TEST})
    assert [r.etat for r in renvois] == ["test-introuvable"]


def test_le_repli_module_reconnait_un_vrai_fichier_de_tests(corpus):
    """Le contre-verrou : le repli doit continuer de servir, sans quoi on
    aurait fabriqué un faux fatal en corrigeant un fatal masqué."""
    renvois, _ = _depot_de_test(corpus, "voir `test_chose` pour le reste.",
                                {"tests/test_chose.py": UN_TEST})
    assert [r.etat for r in renvois] == ["test-module"]
