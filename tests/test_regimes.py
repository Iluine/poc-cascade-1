"""Définition physique des régimes (anti-circularité). Le label vient de la vorticité/Q,
pas du détail de Harten — ce que le test `test_regime_diverges_from_harten_detail` prouve."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cascade.experts import regimes as R
from cascade.harten import windowing as H


def _uniform(nx=32, ny=32, U=0.1):
    return jnp.stack([jnp.full((nx, ny), U), jnp.zeros((nx, ny))])


def _solid_rotation(n=32, Om=0.05, cx=None, cy=None):
    cx = (n - 1) / 2 if cx is None else cx
    cy = (n - 1) / 2 if cy is None else cy
    xs = jnp.arange(n, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    return jnp.stack([-Om * (Y - cy), Om * (X - cx)])  # ω = 2·Om partout


def test_uniform_flow_is_convective():
    u = _uniform()
    labels = R.label_regimes(u, center=(100, 100), radius=5, omega_threshold=0.01)  # interface hors domaine
    assert np.all(np.asarray(labels) == R.CONVECTIVE)


def test_solid_rotation_is_shear():
    u = _solid_rotation(n=32, Om=0.05)  # ω = 0.1 partout
    labels = R.label_regimes(u, center=(-10, -10), radius=1, omega_threshold=0.05, use_q=False)
    assert (np.asarray(labels) == R.SHEAR).mean() > 0.8


def test_interface_near_body():
    u = _uniform()
    labels = np.asarray(R.label_regimes(u, center=(16, 16), radius=4, omega_threshold=0.01, margin=2))
    assert labels[16, 16] == R.INTERFACE          # au centre du corps
    assert labels[16, 28] == R.CONVECTIVE          # loin, vorticité nulle


def test_q_criterion_positive_in_vortex():
    u = _solid_rotation(n=32, Om=0.05)
    q = np.asarray(R.q_criterion(u))
    assert q[16, 16] > 0    # rotation solide : Q>0 (pas de cisaillement)


def test_calibrate_threshold_positive():
    u = _solid_rotation(n=40, Om=0.05)
    thr = R.calibrate_omega_threshold(u, center=(-10, -10), radius=1, q=0.85)
    assert thr > 0 and np.isfinite(thr)


def test_regime_diverges_from_harten_detail():
    # CŒUR DE L'ANTI-CIRCULARITÉ : rotation solide LISSE -> vorticité haute (régime SHEAR)
    # MAIS détail de Harten BAS (champ lisse). Le détail ne définit donc PAS le régime ;
    # c'est exactement la zone où le routeur appris peut battre le seuil de détail nu.
    u = _solid_rotation(n=32, Om=0.05)             # ω=0.1 partout -> SHEAR
    labels = R.label_regimes(u, center=(-10, -10), radius=1, omega_threshold=0.05, use_q=False)
    frac_shear = (np.asarray(labels) == R.SHEAR).mean()
    det_mag = float(H.detail_magnitude(H.detail(u)[1]).max())
    # vorticité du champ (échelle du régime)
    omega_max = float(jnp.abs(R.vorticity(u)).max())
    assert frac_shear > 0.8                         # régime = SHEAR (par vorticité)
    assert det_mag < 0.5 * omega_max                # détail Harten faible malgré SHEAR
