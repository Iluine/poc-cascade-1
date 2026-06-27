"""Fenêtrage de Harten — granularité de routage native (§1, §6).

Analyse multirésolution (MRA) : moyenne de cellule + coefficient de détail. Le détail
mesure la variation sous-grille par fenêtre grossière — petit en zone lisse, grand aux
fronts (cisaillement, vortex, interface). C'est l'indicateur natif du routeur :
- détail petit + halo grossier déjà vu  → réutiliser l'opérateur mémoïsé ;
- détail modéré                          → invoquer l'expert de régime ;
- détail grand (structure sous-grille)   → descendre d'un niveau (fallback).

Le routeur ne voit QUE le halo grossier + le détail (jamais le champ fin) → décision bon marché.
"""

from cascade.harten.windowing import (  # noqa: F401
    coarsen,
    upsample,
    detail,
    detail_magnitude,
    decompose,
    reconstruct,
    window_features,
)
