"""POD+DMD : doit récupérer les fréquences d'un signal multi-modes et rouler sans dériver."""

import numpy as np
from cascade.experts import pod_dmd


def _synthetic(F=60, n=400, freqs=(0.02, 0.05), dt=1.0, seed=0):
    """Champ = somme de modes spatiaux oscillant à des fréquences connues + champ moyen."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    mean = rng.normal(size=F)
    S = np.tile(mean[:, None], (1, n))
    for f in freqs:
        shape = rng.normal(size=F)
        S = S + np.outer(shape, np.cos(2 * np.pi * f * t)) + np.outer(rng.normal(size=F), np.sin(2 * np.pi * f * t))
    return S, mean, freqs


def test_recovers_frequencies():
    S, _, freqs = _synthetic(freqs=(0.02, 0.05))
    rom = pod_dmd.fit(S, rank=4)
    dmd_f = np.sort(np.unique(np.round(rom.dmd_frequencies(), 4)))
    dmd_f = dmd_f[dmd_f > 1e-6]
    for f in freqs:
        assert np.min(np.abs(dmd_f - f)) < 0.003, f"freq {f} non retrouvée dans {dmd_f}"


def test_pod_energy_captures_modes():
    S, _, _ = _synthetic(freqs=(0.02, 0.05))   # 2 modes -> 4 dims (cos+sin) dominent
    assert pod_dmd.pod_energy(S, rank=4) > 0.99


def test_rollout_stable_for_oscillatory():
    # signal purement oscillatoire (cycle limite) -> rayon spectral ~1, rollout borné
    S, _, _ = _synthetic(freqs=(0.02, 0.05), n=600)
    rom = pod_dmd.fit(S, rank=4)
    assert rom.spectral_radius() < 1.02            # pas de croissance
    roll = rom.rollout(S[:, 0], 2000)              # long rollout
    assert np.all(np.isfinite(roll))
    assert roll.std() < 5 * S.std()                # amplitude bornée (pas de dérive)


def test_advance_matches_data_one_step():
    S, _, _ = _synthetic(freqs=(0.03,), n=300)
    rom = pod_dmd.fit(S, rank=2)
    pred = rom.advance(S[:, 100])
    # erreur de prédiction un-pas petite devant la variation du signal
    err = np.linalg.norm(pred - S[:, 101])
    scale = np.linalg.norm(S[:, 101] - S[:, 100])
    assert err < 0.1 * scale
