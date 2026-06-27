"""Harnais perceptuel (§8) — TOUT en spectral, sur la fenêtre stationnaire.

L2 point-à-point INTERDITE comme critère (trajectoire VIV chaotique ; la L2
punirait le bon comportement). Métriques :

- M1 : spectre fluide — pente plage inertielle (K41) + puissance par bande.
- M2 : lâcher — Strouhal (routé vs oracle).
- M3 : structure — fréquence dominante + enveloppe d'amplitude.
- M4 : lock-in — cohérence de phase sillage↔structure (l'invariant couplé clé).

Analyse hors-graphe (numpy/scipy) : ne fait pas partie du graphe JAX différentiable.
"""

from cascade.metrics.spectral import (  # noqa: F401
    psd_welch,
    dominant_frequency,
    strouhal,
    amplitude_envelope,
    energy_spectrum_2d,
    inertial_slope,
    phase_coherence,
)
