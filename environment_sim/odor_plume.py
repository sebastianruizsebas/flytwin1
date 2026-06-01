"""
Environment Simulation — Gaussian-puff turbulent odor plume (Phase 1b)
Implements:
    c(x,t) = sum_k A_k * exp(-||x - mu_k(t)||^2 / (2 sigma_k^2))
Puffs advect with mean wind w_t plus Gaussian noise (turbulent jitter).
Exposes bilateral antennal concentration sensors and wind vector.

Biological basis: Drosophila detect odor plumes as intermittent Gaussian
packets (puffs) advected by airflow. Left/right antennal asymmetry drives
casting behavior; mean wind direction encodes upwind surging cues.

GPU/performance note
--------------------
Puff state is stored as three contiguous numpy arrays (_pos, _amp, _sig)
instead of a Python list of Puff dataclass objects.  This removes the
O(K) Python loop from ``step()`` and enables vectorised advection via a
single numpy broadcast operation.  JAX JIT is not used here because the
number of puffs changes every step (spawn/cull), which requires dynamic
array sizes incompatible with JAX tracing.  Plain numpy suffices since
the advection broadcast is already near-native speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OdorPlume:
    """Turbulent plume composed of discrete Gaussian puffs.

    Puff state is stored internally as three arrays:
      _pos : (K, 3) float64  world positions (metres)
      _amp : (K,)  float64  peak amplitude per puff
      _sig : (K,)  float64  spatial spread sigma per puff (metres)
    """
    wind_mean:       np.ndarray = field(default_factory=lambda: np.array([0.3, 0.0, 0.0]))
    wind_noise_std:  float = 0.05
    puff_rate:       float = 10.0     # new puffs per second
    source_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _rng:            np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    # Maximum arena radius; puffs beyond this are culled
    _max_radius:     float = field(default=2.0, repr=False)

    # Vectorised puff state (no Puff dataclass objects)
    _pos: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float), repr=False)
    _amp: np.ndarray = field(default_factory=lambda: np.empty(0,     dtype=float), repr=False)
    _sig: np.ndarray = field(default_factory=lambda: np.empty(0,     dtype=float), repr=False)

    def step(self, dt: float) -> None:
        """
        Advance plume state by dt seconds.

        Fully vectorised — no Python loop over puffs:
          1. Advect all K puffs in one numpy broadcast (K, 3) + (K, 3).
          2. Cull distant puffs with boolean indexing.
          3. Spawn new puffs and concatenate.
        """
        K = len(self._pos)
        if K > 0:
            noise = self._rng.normal(0.0, self.wind_noise_std, (K, 3))
            self._pos = self._pos + (self.wind_mean + noise) * dt  # (K,3) — no loop

            # Cull puffs outside arena
            dists = np.linalg.norm(self._pos - self.source_position[None, :], axis=1)
            keep = dists < self._max_radius
            self._pos = self._pos[keep]
            self._amp = self._amp[keep]
            self._sig = self._sig[keep]

        # Spawn new puffs (Poisson)
        n_new = int(self._rng.poisson(self.puff_rate * dt))
        if n_new > 0:
            new_pos = self.source_position[None, :] + self._rng.normal(0.0, 0.005, (n_new, 3))
            new_amp = np.ones(n_new, dtype=float)
            new_sig = np.full(n_new, 0.02, dtype=float)

            self._pos = np.concatenate([self._pos, new_pos], axis=0)
            self._amp = np.concatenate([self._amp, new_amp])
            self._sig = np.concatenate([self._sig, new_sig])

    def concentration_at(self, position: np.ndarray) -> float:
        """
        Evaluate c(x,t) at a 3D position by summing Gaussian puff contributions.
        Returns raw (un-clipped) concentration summed over all active puffs.
        """
        if self._pos.size == 0:
            return 0.0
        sq_dist = np.sum((self._pos - position[None, :]) ** 2, axis=1)  # (K,)
        contributions = self._amp * np.exp(-sq_dist / (2.0 * self._sig ** 2))
        return float(np.sum(contributions))

    def get_antennal_obs(
        self,
        antenna_left: np.ndarray,
        antenna_right: np.ndarray,
    ) -> dict:
        """
        Return u_sens = {c_left, c_right, wind_vector} for injection into Level 1.
        Applies sigmoidal gain to bilateral concentrations to produce [0-1] drive.

        The wind_vector is the planar (x, y) component of wind_mean, which encodes
        upwind direction cues used by the SURGE mode selector at Level 3.
        """
        c_left_raw  = self.concentration_at(antenna_left)
        c_right_raw = self.concentration_at(antenna_right)
        return {
            "c_left":       sigmoid_gain(c_left_raw),
            "c_right":      sigmoid_gain(c_right_raw),
            "wind_vector":  self.wind_mean[:2].copy(),  # planar wind (x, y)
        }

    @property
    def n_puffs(self) -> int:
        """Current number of active puffs (for diagnostics / logging)."""
        return len(self._pos)


def sigmoid_gain(c: float, gain: float = 10.0, threshold: float = 0.1) -> float:
    """Map odor concentration to depolarizing current in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-gain * (c - threshold)))
