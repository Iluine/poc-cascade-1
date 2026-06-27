"""Surrogate de régime POD+DMD sur le bus minimal (ρ, ρu) — hypothèse nulle.

Pas de flux conservatif, pas de moments, pas de ghosts (sur-ingénierie retirée, feedback) :
on apprend l'avance temporelle de l'état macroscopique directement, et on laisse C4 dire si
le bus pauvre casse à l'interface (enrichissement chirurgical seulement alors).

POD (SVD des snapshots) → base spatiale bas-rang ; DMD (opérateur linéaire dans les coords
réduites) → avance temporelle. Fitting hors-graphe (numpy) : Phase 1 est non-différentiable
(§7). L'application (advance/rollout) est de simples matmuls → portable JAX si besoin Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PODDMD:
    mean: np.ndarray     # (F,)   moyenne temporelle (le champ moyen, ex. sillage moyen)
    modes: np.ndarray    # (F, r) modes POD (base spatiale)
    A: np.ndarray        # (r, r) opérateur DMD réduit (avance d'un pas)
    dt: float = 1.0

    @property
    def rank(self) -> int:
        return self.modes.shape[1]

    @property
    def eigs(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def reduce(self, q: np.ndarray) -> np.ndarray:
        """État (F,) ou (F,n) -> coords réduites (r,) ou (r,n)."""
        q = np.asarray(q)
        if q.ndim == 1:
            return self.modes.T @ (q - self.mean)
        return self.modes.T @ (q - self.mean[:, None])

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            return self.mean + self.modes @ a
        return self.mean[:, None] + self.modes @ a

    def advance(self, q: np.ndarray) -> np.ndarray:
        """Un pas : projette, applique A, reconstruit."""
        return self.reconstruct(self.A @ self.reduce(q))

    def rollout(self, q0: np.ndarray, n: int) -> np.ndarray:
        """Rollout autonome de n pas depuis q0. Renvoie (F, n)."""
        a = self.reduce(q0)
        out = np.empty((self.mean.shape[0], n))
        for k in range(n):
            a = self.A @ a
            out[:, k] = self.reconstruct(a)
        return out

    def dmd_frequencies(self) -> np.ndarray:
        """Fréquences DMD (Im(log λ)/(2π dt)) — doivent contenir la fréquence du régime."""
        lam = self.eigs
        return np.abs(np.angle(lam)) / (2.0 * np.pi * self.dt)

    def spectral_radius(self) -> float:
        """max|λ| : >1 => le rollout diverge (dérive) ; ~1 => cycle limite stable."""
        return float(np.max(np.abs(self.eigs)))


def fit(snapshots: np.ndarray, rank: int, dt: float = 1.0,
        stabilize: bool = False) -> PODDMD:
    """Ajuste un POD+DMD. snapshots (F, n_snap), colonnes = états successifs (Δt = dt·stride).

    stabilize=False (hypothèse nulle) : DMD brut. Si le rollout dérive (rayon spectral >1),
    C1 l'attrapera — on n'enrichit (stabilize) que si nécessaire (discipline null-first).
    """
    S = np.asarray(snapshots, dtype=np.float64)
    mean = S.mean(axis=1)
    S0 = S - mean[:, None]
    U, s, _ = np.linalg.svd(S0, full_matrices=False)
    Ur = U[:, :rank]
    coords = Ur.T @ S0                      # (r, n_snap)
    X, Y = coords[:, :-1], coords[:, 1:]
    A = Y @ np.linalg.pinv(X)               # opérateur DMD réduit
    if stabilize:
        # projeter les valeurs propres sur/dans le cercle unité (enrichissement, pas défaut)
        w, V = np.linalg.eig(A)
        w = w / np.maximum(np.abs(w), 1.0)
        A = np.real(V @ np.diag(w) @ np.linalg.inv(V))
    return PODDMD(mean=mean, modes=Ur, A=A, dt=dt)


def pod_energy(snapshots: np.ndarray, rank: int) -> float:
    """Fraction d'énergie capturée par les ``rank`` premiers modes POD (choix du rang)."""
    S = np.asarray(snapshots, dtype=np.float64)
    S0 = S - S.mean(axis=1)[:, None]
    s = np.linalg.svd(S0, compute_uv=False)
    return float(np.sum(s[:rank] ** 2) / np.sum(s ** 2))
