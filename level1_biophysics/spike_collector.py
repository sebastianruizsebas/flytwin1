"""
Level 1 — Spike Collector (Phase 1c)
Runs HH neurons across a grid of g_KA scaling values and saves spike trains
to HDF5 under data/spikes/.

Trial protocol:
- For each g_KA in [0.5, 2.0] (default 20 log-spaced steps):
    inject u_sens (sensory drive) + 1/f noise, run T=5000 ms at 1 ms resolution,
    record binary spike trains.
- Biological motivation: g_KA controls transient responsiveness and spike
    timing in Drosophila projection neurons; sweeping it produces systematic
    changes in the PP-GLM coefficients that can be linked back to behavior.

Implementation note:
The actual integration and spike extraction are delegated to the Brian2-backed
HHNeuron batch simulator, which uses SpikeMonitor under the hood.
"""
from __future__ import annotations

import os

import h5py
import numpy as np

from level1_biophysics.hh_neuron import HHNeuron


DT_MS = 1.0


def pink_noise(n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate 1/f (pink) noise via spectral shaping of white noise.
    Models correlated biological variability in sensory drive.
    """
    rng = rng or np.random.default_rng()
    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    return np.fft.irfft(spectrum, n=n_samples)


def pink_noise_batch(
    n_trials: int,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate independent pink-noise traces for a batch of trials."""
    rng = rng or np.random.default_rng()
    return np.vstack([pink_noise(n_samples, rng=rng) for _ in range(n_trials)])


def collect_spikes(
    swc_path: str,
    u_sens: np.ndarray,
    g_KA_values: np.ndarray,
    n_trials: int = 5000,
    out_dir: str = "data/spikes",
) -> None:
    """
    Run spike collection loop and save to HDF5.

    For each g_KA scaling value, n_trials independent runs of T=len(u_sens) ms
    are executed. Each trial adds independent pink noise to u_sens, producing
    a (n_trials, T) binary spike matrix saved to HDF5.

    Parameters
    ----------
    swc_path : str
        Path to neuron morphology SWC file (unused in reduced model but kept
        for interface compatibility with future NEURON extension).
    u_sens : np.ndarray
        (T,) array of mean sensory current drive in µA/cm², one value per ms.
    g_KA_values : np.ndarray
        Array of g_KA multiplier values to sweep (e.g. np.linspace(0.5, 2.0, 20)).
    n_trials : int
        Number of independent noise trials per g_KA condition.
    out_dir : str
        Output directory for HDF5 files.
    """
    os.makedirs(out_dir, exist_ok=True)
    mean_drive = np.asarray(u_sens, dtype=float)
    T = len(mean_drive)
    rng = np.random.default_rng(seed=42)

    # Noise amplitude: a fraction of mean drive amplitude
    noise_scale = float(np.std(mean_drive)) * 0.3 if np.std(mean_drive) > 0 else 0.5

    out_path = os.path.join(out_dir, "spikes_gKA_sweep.h5")
    with h5py.File(out_path, "w") as f:
        f.attrs["n_trials"] = n_trials
        f.attrs["T_ms"] = T
        f.attrs["dt_ms"] = DT_MS
        f.attrs["g_KA_values"] = g_KA_values

        for i, gka in enumerate(g_KA_values):
            noise = pink_noise_batch(n_trials, T, rng=rng) * noise_scale
            spike_mat = HHNeuron.simulate_spike_trials(
                u_sens=mean_drive,
                drive_noise=noise,
                g_ka_scale=float(gka),
                dt_ms=DT_MS,
            )

            grp = f.create_group(f"gKA_{i:03d}")
            grp.attrs["g_KA"] = float(gka)
            grp.create_dataset("spikes", data=spike_mat, compression="gzip")

    print(f"Saved spike data to {out_path}")
