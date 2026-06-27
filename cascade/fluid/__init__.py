"""Oracle fluide — wrapper du backend JAX différentiable de XLB (repli de AegirJAX).

LBM D2Q9 BGK. Sillage entretenu (§2) : entrée vitesse prescrite, sortie
non-réfléchissante (extrapolation), parois haut/bas configurables.
"""

from cascade.fluid.lbm import FluidConfig, Fluid, build_fluid  # noqa: F401
