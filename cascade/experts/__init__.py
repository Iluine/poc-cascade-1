"""Experts de régime — surrogates POD+DMD par régime (fluide), POD modes de déformation (solide).

ORDRE (§10) : experts d'abord, C1 les teste isolés, puis le routeur (Phase 1).

Anti-circularité (feedback critique) : les régimes sont définis par la PHYSIQUE
(vorticité/Q + marqueur de structure), PAS par le détail de Harten — sinon le routeur nul
(seuil de détail) serait un classificateur de régime parfait par construction et C2 (appris
vs nul) deviendrait invérifiable. Voir :mod:`cascade.experts.regimes`.

Sortie = flux par face (forme conservative = le bus). POD construite sur les flux directement.
"""

from cascade.experts.regimes import (  # noqa: F401
    vorticity,
    q_criterion,
    structure_mask,
    label_regimes,
    calibrate_omega_threshold,
    CONVECTIVE,
    SHEAR,
    INTERFACE,
)
from cascade.experts.pod_dmd import PODDMD, fit, pod_energy  # noqa: F401
