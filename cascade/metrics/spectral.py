"""Outils spectraux pour M1–M4 (§8). numpy/scipy, hors graphe JAX."""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def _as1d(x):
    return np.asarray(x, dtype=np.float64).ravel()


def psd_welch(x, dt=1.0, nperseg=None):
    """Densité spectrale de puissance unilatérale (Welch, fenêtre de Hann, détrendée)."""
    x = _as1d(x)
    if nperseg is None:
        nperseg = min(len(x), max(256, len(x) // 8))
    f, p = sps.welch(x, fs=1.0 / dt, nperseg=nperseg, detrend="linear", window="hann")
    return f, p


def dominant_frequency(x, dt=1.0, band=None, pad=8):
    """Fréquence dominante avec précision sub-bin (périodogramme zero-paddé + interpolation
    parabolique). Adapté à un signal périodique stationnaire (VIV) où Welch arrondirait
    trop grossièrement la basse fréquence du lâcher.

    Renvoie (f_peak, (f, p)) où (f, p) est le spectre Welch (pour tracé/diagnostic).
    """
    x = _as1d(x)
    x = x - np.mean(x)
    n = len(x)
    w = np.hanning(n)
    xw = x * w
    nfft = int(2 ** np.ceil(np.log2(n * pad)))
    X = np.fft.rfft(xw, n=nfft)
    f = np.fft.rfftfreq(nfft, d=dt)
    mag = np.abs(X)
    mask = np.ones_like(f, dtype=bool); mask[0] = False
    if band is not None:
        mask &= (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return np.nan, psd_welch(x, dt=dt)
    idx_in = np.where(mask)[0]
    k0 = idx_in[np.argmax(mag[idx_in])]
    # interpolation parabolique sur le log-magnitude (précision sous-bin)
    if 0 < k0 < len(mag) - 1:
        y0, y1, y2 = np.log(mag[k0 - 1] + 1e-300), np.log(mag[k0] + 1e-300), np.log(mag[k0 + 1] + 1e-300)
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = np.clip(delta, -0.5, 0.5)
    else:
        delta = 0.0
    df = f[1] - f[0]
    fpk = (k0 + delta) * df
    return float(fpk), psd_welch(x, dt=dt)


def strouhal(x, D, U, dt=1.0, band=None):
    """St = f·D/U à partir de la fréquence dominante du signal (p.ex. portance)."""
    f, _ = dominant_frequency(x, dt=dt, band=band)
    if not np.isfinite(f):
        return np.nan, np.nan
    return float(f * D / U), float(f)


def amplitude_envelope(x):
    """Enveloppe d'amplitude (Hilbert) — M3 : amplitude, pas position instantanée.

    Renvoie (enveloppe_moyenne, enveloppe_array) sur le signal détrendé.
    """
    x = _as1d(x)
    x = x - np.mean(x)
    env = np.abs(sps.hilbert(x))
    return float(np.mean(env)), env


def energy_spectrum_2d(u, kbins=None):
    """Spectre d'énergie radial E(k) d'un champ de vitesse 2D ``u`` de forme (2, nx, ny).

    M1 : pente de plage inertielle. (À Re modéré le sillage est laminaire : la plage
    inertielle peut être absente — on rapporte la pente mesurée, sans la sur-interpréter.)
    """
    u = np.asarray(u, dtype=np.float64)
    _, nx, ny = u.shape
    uh = np.fft.fft2(u[0]); vh = np.fft.fft2(u[1])
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (nx * ny) ** 2
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    if kbins is None:
        kbins = np.arange(0.5, min(nx, ny) // 2)
    Ek = np.zeros(len(kbins) - 1)
    kcen = 0.5 * (kbins[1:] + kbins[:-1])
    for i in range(len(kbins) - 1):
        m = (kmag >= kbins[i]) & (kmag < kbins[i + 1])
        Ek[i] = e[m].sum()
    return kcen, Ek


def inertial_slope(k, Ek, krange):
    """Pente log-log de E(k) sur ``krange=(kmin,kmax)`` (référence K41 = −5/3)."""
    k = np.asarray(k); Ek = np.asarray(Ek)
    m = (k >= krange[0]) & (k <= krange[1]) & (Ek > 0)
    if m.sum() < 3:
        return np.nan
    coef = np.polyfit(np.log(k[m]), np.log(Ek[m]), 1)
    return float(coef[0])


def phase_coherence(x, y, dt=1.0, band=None, nperseg=None):
    """M4 — lock-in : cohérence quadratique MAXIMALE entre sillage ``x`` et structure
    ``y`` sur ``band`` (max, PAS moyenne — la docstring disait l'inverse du code,
    review 28/07). Le max est GÉNÉREUX : un seul bin cohérent suffit à scorer haut.
    Depuis M11/M14, M4 est SURFACÉ en diagnostic, il ne juge plus aucun verdict."""
    x = _as1d(x); y = _as1d(y)
    n = min(len(x), len(y))
    if nperseg is None:
        nperseg = min(n, max(256, n // 8))
    f, cxy = sps.coherence(x[:n], y[:n], fs=1.0 / dt, nperseg=nperseg, window="hann")
    mask = np.ones_like(f, dtype=bool); mask[0] = False
    if band is not None:
        mask &= (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return np.nan
    return float(np.max(cxy[mask]))
