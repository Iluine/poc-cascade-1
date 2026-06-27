"""cascade2phys — routage deux-physiques sur interface FSI (Test T1).

Carte des sous-modules (la contribution CONSTRUITE, cf. PREREGISTRATION.md §1) :

- ``cascade.fluid``      wrapper XLB     — oracle fluide + champ d'entrée des experts
- ``cascade.solid``      wrapper jax-fem — oracle solide
- ``cascade.coupling``   IB Peskin       — bus = flux par face (forme conservative)
- ``cascade.harten``     fenêtrage 2–3 niveaux (moyenne cellule + coefficient de détail)
- ``cascade.router``     routeur 3 sorties : nul (Harten codé dur) | appris
- ``cascade.experts``    POD+DMD fluide par régime ; POD solide (modes de déformation)
- ``cascade.metrics``    M1–M4 perceptuel (spectral ; JAMAIS L2 comme critère)
- ``cascade.accounting`` FLOPs + wall-clock

Invariant gravé : G0/G0' valident l'INSTRUMENT, C1–C4 valident l'ARCHITECTURE.
"""

__version__ = "0.0.1"
