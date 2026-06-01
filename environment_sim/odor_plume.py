"""
Environment Simulation — Gaussian-puff turbulent odor plume (Phase 1b)
Implements:
    c(x,t) = sum_k A_k * exp(-||x - mu_k(t)||^2 / (2 sigma_k^2))
Puffs advect with mean wind w_t plus Gaussian noise (turbulent jitter).
Exposes bilateral antennal concentration sensors and wind vector.

Biological basis: Drosophila detect odor plumes as intermittent Gaussian
packets (puffs) advected by airflow. Left/right antennal asymmetry drives
casting behavior; mean wind direction encodes upwind surging cues.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Puff:
    position: np.ndarray       # (3,) world coordinates, metres
    amplitude: float = 1.0
    sigma: float = 0.02        # spatial spread, metres


@dataclass
class OdorPlume:
    """Turbulent plume composed of discrete Gaussian puffs."""
    wind_mean: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.0, 0.0]))
    wind_noise_std: float = 0.05
    puff_rate: float = 10.0    # new puffs per second
    source_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _puffs: list[Puff] = field(default_factory=list, repr=False)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    # Maximum arena radius; puffs beyond this are culled
    _max_radius: float = field(default=2.0, repr=False)

    def step(self, dt: float) -> None:
        """
        Advance plume state by dt seconds.
        Each puff advects with mean wind plus Gaussian turbulent noise.
        New puffs are spawned at the source with Poisson rate puff_rate.
        Puffs that drift beyond _max_radius are culled.
        """
        # Advect existing puffs
        noise = self._rng.normal(0.0, self.wind_noise_std, (len(self._puffs), 3))
        for i, puff in enumerate(self._puffs):
            puff.position = puff.position + (self.wind_mean + noise[i]) * dt

        # Cull puffs that have travelled too far from source
        self._puffs = [
            p for p in self._puffs
            if np.linalg.norm(p.position - self.source_position) < self._max_radius
        ]

        # Spawn new puffs: expected number follows Poisson(puff_rate * dt)
        n_new = self._rng.poisson(self.puff_rate * dt)
        for _ in range(n_new):
            # Small jitter around source position on spawn
            spawn_pos = self.source_position + self._rng.normal(0.0, 0.005, 3)
            self._puffs.append(Puff(position=spawn_pos.copy()))

    def concentration_at(self, position: np.ndarray) -> float:
        """
        Evaluate c(x,t) at a 3D position by summing Gaussian puff contributions.
        Returns raw (un-clipped) concentration summed over all active puffs.
        """
        if not self._puffs:
            return 0.0
        positions = np.array([p.position for p in self._puffs])   # (K, 3)
        amplitudes = np.array([p.amplitude for p in self._puffs])  # (K,)
        sigmas = np.array([p.sigma for p in self._puffs])          # (K,)
        sq_dist = np.sum((positions - position) ** 2, axis=1)      # (K,)
        contributions = amplitudes * np.exp(-sq_dist / (2.0 * sigmas ** 2))
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
        c_left_raw = self.concentration_at(antenna_left)
        c_right_raw = self.concentration_at(antenna_right)
        return {
            "c_left": sigmoid_gain(c_left_raw),
            "c_right": sigmoid_gain(c_right_raw),
            "wind_vector": self.wind_mean[:2].copy(),  # planar wind (x, y)
        }


def sigmoid_gain(c: float, gain: float = 10.0, threshold: float = 0.1) -> float:
    """Map odor concentration to depolarizing current in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-gain * (c - threshold)))
