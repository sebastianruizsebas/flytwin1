"""
Level 1 — Biophysics
Hodgkin-Huxley single-compartment neuron (Phase 1c).

This version uses Brian2 for numerical integration and spike detection while
preserving the reduced-model interface expected by the rest of the codebase.
Full compartmental NEURON morphology can still be swapped in later by
overriding build() and step() while keeping the same interface.

Interface: inject depolarizing current I_t (µA/cm²), step 1 ms, return spike bool.

Conductances (mS/cm²):
    g_Na   — fast sodium, spike initiation
    g_K    — delayed rectifier, repolarization
    g_KA   — A-type potassium, transient responsiveness / timing
    g_CaL  — L-type calcium, slower depolarising dynamics
    g_h    — hyperpolarization-activated (Ih), pacemaker / rebound

Reference defaults approximate Drosophila PN literature values
(Wilson & Laurent 2005; Nagel & Wilson 2011).
"""
from __future__ import annotations

import os

import numpy as np
from brian2 import Network, NeuronGroup, SpikeMonitor, TimedArray, defaultclock, ms, mV, uA, cm, prefs

from utils.accelerator import BRIAN2_TARGET

prefs.codegen.target = BRIAN2_TARGET


_INTEGRATION_DT_MS = 0.05


# ── Literature-derived base conductances (mS/cm²) ──────────────────────────
_BASE_G = {
    "g_Na":  120.0,
    "g_K":    36.0,
    "g_KA":    5.0,
    "g_CaL":   2.0,
    "g_h":     1.5,
}
# Reversal potentials (mV)
_E_Na =  50.0
_E_K  = -77.0
_E_Ca =  80.0
_E_h  = -43.0
_E_L  = -54.3   # leak
_G_L  =   0.3   # mS/cm²
_C_m  =   1.0   # µF/cm²
_V_REST = -65.0  # mV


_MODEL_EQS = """
dv/dt = (I_input - I_Na - I_K - I_KA - I_CaL - I_h - I_L) / C_m / ms : 1
dm/dt = alpha_m * (1.0 - m) - beta_m * m : 1
dh/dt = alpha_h * (1.0 - h) - beta_h * h : 1
dn/dt = alpha_n * (1.0 - n) - beta_n * n : 1
da/dt = (a_inf - a) / tau_a : 1
db/dt = (b_inf - b) / tau_b : 1
dd_gate/dt = (d_inf - d_gate) / tau_d : 1
dq/dt = (q_inf - q) / tau_q : 1
I_drive : 1
I_input = __INPUT_EXPR__ : 1
gNa : 1
gK : 1
gKA : 1
gCaL : 1
gh : 1
E_Na : 1
E_K : 1
E_Ca : 1
E_h : 1
E_L : 1
G_L : 1
C_m : 1
spike_threshold : 1
I_Na = gNa * m**3 * h * (v - E_Na) : 1
I_K = gK * n**4 * (v - E_K) : 1
I_KA = gKA * a**3 * b * (v - E_K) : 1
I_CaL = gCaL * d_gate * (v - E_Ca) : 1
I_h = gh * q * (v - E_h) : 1
I_L = G_L * (v - E_L) : 1
alpha_m = (1.0 / exprel(-(v + 40.0) / 10.0)) / ms : Hz
beta_m = (4.0 * exp(-(v + 65.0) / 18.0)) / ms : Hz
alpha_h = (0.07 * exp(-(v + 65.0) / 20.0)) / ms : Hz
beta_h = (1.0 / (1.0 + exp(-(v + 35.0) / 10.0))) / ms : Hz
alpha_n = (0.1 / exprel(-(v + 55.0) / 10.0)) / ms : Hz
beta_n = (0.125 * exp(-(v + 65.0) / 80.0)) / ms : Hz
a_inf = (0.0761 * exp((v + 94.22) / 31.84) / (1.0 + exp((v + 1.17) / 28.93))) ** (1.0 / 3.0) : 1
tau_a = (0.3632 + 1.158 / (1.0 + exp((v + 55.96) / 20.12))) * ms : second
b_inf = (1.0 / (1.0 + exp((v + 53.3) / 14.54))) ** 4.0 : 1
tau_b = (1.24 + 2.678 / (1.0 + exp((v + 50.0) / 16.027))) * ms : second
d_inf = 1.0 / (1.0 + exp(-(v + 10.0) / 6.0)) : 1
tau_d = (5.0 + 20.0 * exp(-((v + 25.0) / 30.0) ** 2)) * ms : second
q_inf = 1.0 / (1.0 + exp((v + 75.0) / 5.5)) : 1
tau_q = (50.0 + 750.0 / (1.0 + exp(-(v + 80.0) / 15.0))) * ms : second
"""


def _alpha_m(v): return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0)) if abs(v + 40.0) > 1e-7 else 1.0
def _beta_m(v):  return 4.0 * np.exp(-(v + 65.0) / 18.0)
def _alpha_h(v): return 0.07 * np.exp(-(v + 65.0) / 20.0)
def _beta_h(v):  return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
def _alpha_n(v): return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0)) if abs(v + 55.0) > 1e-7 else 0.1
def _beta_n(v):  return 0.125 * np.exp(-(v + 65.0) / 80.0)

# A-type K channel (Connor-Stevens simplified)
def _a_inf(v):   return (0.0761 * np.exp((v + 94.22) / 31.84) / (1.0 + np.exp((v + 1.17) / 28.93))) ** (1.0 / 3.0)
def _tau_a(v):   return 0.3632 + 1.158 / (1.0 + np.exp((v + 55.96) / 20.12))
def _b_inf(v):   return (1.0 / (1.0 + np.exp((v + 53.3) / 14.54))) ** 4.0
def _tau_b(v):   return 1.24 + 2.678 / (1.0 + np.exp((v + 50.0) / 16.027))

# L-type Ca (simplified)
def _d_inf(v):   return 1.0 / (1.0 + np.exp(-(v + 10.0) / 6.0))
def _tau_d(v):   return 5.0 + 20.0 * np.exp(-((v + 25.0) / 30.0) ** 2)

# Ih (HCN simplified)
def _q_inf(v):   return 1.0 / (1.0 + np.exp((v + 75.0) / 5.5))
def _tau_q(v):   return 50.0 + 750.0 / (1.0 + np.exp(-(v + 80.0) / 15.0))


class HHNeuron:
    """
    Single-compartment Hodgkin-Huxley neuron with five conductances.
    Serves as the reduced first-pass Level 1 unit.  Full compartmental
    NEURON morphology is a later extension.
    """

    SPIKE_THRESHOLD_MV = 0.0  # voltage crossing used for spike detection

    def __init__(self, swc_path: str = "", conductances: dict | None = None):
        self.swc_path = swc_path
        # Scaling factors on top of base conductances
        self.conductances: dict[str, float] = {k: 1.0 for k in _BASE_G}
        if conductances:
            self.conductances.update(conductances)

        self._cell = None  # legacy placeholder
        self._group: NeuronGroup | None = None  # NeuronGroup OR SpatialNeuron
        self._network: Network | None = None
        self._spike_monitor: SpikeMonitor | None = None
        # Compartmental model state — populated in build() when swc_path is valid
        self._is_compartmental: bool = False
        self._soma_section = None   # SpatialSubgroup for soma (set when compartmental)
        self._spike_times: list[float] = []
        self._t_ms: float = 0.0

        # State variables — initialised to resting values
        self._v:    float = _V_REST
        self._m:    float = self._gate_inf(_alpha_m, _beta_m, _V_REST)
        self._h:    float = self._gate_inf(_alpha_h, _beta_h, _V_REST)
        self._n:    float = self._gate_inf(_alpha_n, _beta_n, _V_REST)
        self._a:    float = _a_inf(_V_REST)
        self._b:    float = _b_inf(_V_REST)
        self._d:    float = _d_inf(_V_REST)
        self._q:    float = _q_inf(_V_REST)
        self._above_thresh: bool = False  # for threshold crossing detection

    # ── public interface ────────────────────────────────────────────────────

    def build(self) -> None:
        """
        Build the Brian2 network for this neuron.

        When self.swc_path points to an existing SWC skeleton file (produced by
        import_connectome.py), a multi-compartment Brian2 SpatialNeuron is
        constructed from the morphology.  HH conductances are placed at the soma;
        axon and dendrite compartments are passive cable.

        When no SWC path is given (or the file is absent), a single-compartment
        NeuronGroup is used as a fast fallback.  This keeps the PP-GLM fitting
        workflow functional before connectome skeletons are imported.
        """
        if self._network is not None:
            return

        if not self.swc_path:
            raise FileNotFoundError(
                "HHNeuron.build() requires a valid SWC morphology file.\n"
                "No swc_path was provided.  Run import_connectome.py with\n"
                "--export-skeletons to download skeleton files into\n"
                "data/connectome/skeletons/, then pass the path to HHNeuron."
            )
        if not os.path.exists(str(self.swc_path)):
            raise FileNotFoundError(
                f"HHNeuron.build() could not find the SWC file:\n"
                f"  {self.swc_path}\n"
                f"Run import_connectome.py with --export-skeletons to download\n"
                f"skeleton files into data/connectome/skeletons/."
            )

        # ── Compartmental path: SWC → Brian2 SpatialNeuron ───────────────────
        from level1_biophysics.compartmental_neuron import (
            build_compartmental_network,
            soma_section,
        )
        self._group, self._spike_monitor, self._network = \
            build_compartmental_network(self.swc_path, self.conductances)
        self._is_compartmental = True
        self._soma_section = soma_section(self._group)

        self._network.store("resting")
        self._sync_from_group()

    def step(self, current_uA: float, dt_ms: float = 1.0) -> bool:
        """
        Advance simulation by dt_ms using Brian2 Euler integration.
        Returns True if a spike (threshold crossing) occurred this step.
        current_uA: applied current in µA/cm².
        """
        if self._network is None or self._group is None or self._spike_monitor is None:
            self.build()

        prev_spike_count = len(self._spike_monitor.t)
        if self._is_compartmental:
            # Inject current density at soma only.
            # µA/cm² → Brian2 unit: uA/cm**2
            self._soma_section.I_drive = float(current_uA) * uA / cm**2
        else:
            self._group.I_drive = float(current_uA)
        self._network.run(float(dt_ms) * ms)

        new_spike_times = self._spike_monitor.t[prev_spike_count:] / ms
        if len(new_spike_times) > 0:
            self._spike_times.extend(np.asarray(new_spike_times, dtype=float).tolist())

        self._t_ms = float(self._network.t / ms)
        self._sync_from_group()
        return len(new_spike_times) > 0

    def reset(self) -> None:
        """Reset membrane potential and gating variables to resting state."""
        if self._network is None or self._group is None:
            self._v = _V_REST
            self._m = self._gate_inf(_alpha_m, _beta_m, _V_REST)
            self._h = self._gate_inf(_alpha_h, _beta_h, _V_REST)
            self._n = self._gate_inf(_alpha_n, _beta_n, _V_REST)
            self._a = _a_inf(_V_REST)
            self._b = _b_inf(_V_REST)
            self._d = _d_inf(_V_REST)
            self._q = _q_inf(_V_REST)
            self._above_thresh = False
            self._t_ms = 0.0
            self._spike_times = []
            return

        self._network.restore("resting")
        self._sync_from_group()
        self._above_thresh = False
        self._t_ms = 0.0
        self._spike_times = []

    @property
    def spike_times(self) -> list[float]:
        return list(self._spike_times)

    @classmethod
    def simulate_spike_trials(
        cls,
        u_sens: np.ndarray,
        drive_noise: np.ndarray,
        g_ka_scale: float,
        dt_ms: float = 1.0,
    ) -> np.ndarray:
        """
        Run a batch of independent trials in Brian2 and return a binary spike matrix.

        Parameters
        ----------
        u_sens : np.ndarray
            Mean sensory drive over time, shape (T,).
        drive_noise : np.ndarray
            Additive drive noise, shape (n_trials, T).
        g_ka_scale : float
            Multiplicative scale on the A-type potassium conductance.
        dt_ms : float
            Simulation time step in ms.
        """
        mean_drive = np.asarray(u_sens, dtype=float)
        noise = np.asarray(drive_noise, dtype=float)
        if noise.ndim != 2:
            raise ValueError("drive_noise must have shape (n_trials, T)")
        if mean_drive.ndim != 1:
            raise ValueError("u_sens must have shape (T,)")
        if noise.shape[1] != mean_drive.shape[0]:
            raise ValueError("drive_noise and u_sens must share the same time axis")

        n_trials, n_steps = noise.shape
        stimulus_values = (mean_drive[np.newaxis, :] + noise).T
        stimulus = TimedArray(stimulus_values, dt=float(dt_ms) * ms)

        defaultclock.dt = _INTEGRATION_DT_MS * ms
        group = cls._create_neuron_group(
            n_neurons=n_trials,
            input_expr="stimulus(t, i)",
            namespace={"stimulus": stimulus},
        )
        group.gKA = float(g_ka_scale) * _BASE_G["g_KA"]
        cls._assign_initial_state(group)

        spike_monitor = SpikeMonitor(group)
        network = Network(group, spike_monitor)
        network.run(n_steps * float(dt_ms) * ms)

        spike_matrix = np.zeros((n_trials, n_steps), dtype=np.uint8)
        for neuron_idx, spike_times in spike_monitor.spike_trains().items():
            spike_bins = np.floor(np.asarray(spike_times / ms, dtype=float) / float(dt_ms)).astype(int)
            spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < n_steps)]
            spike_matrix[int(neuron_idx), spike_bins] = 1

        return spike_matrix

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _gate_inf(alpha_fn, beta_fn, v: float) -> float:
        a = alpha_fn(v)
        b = beta_fn(v)
        return a / (a + b)

    @classmethod
    def _create_neuron_group(
        cls,
        n_neurons: int,
        input_expr: str = "I_drive",
        namespace: dict | None = None,
    ) -> NeuronGroup:
        model = _MODEL_EQS.replace("__INPUT_EXPR__", input_expr)
        group = NeuronGroup(
            n_neurons,
            model=model,
            threshold="v >= spike_threshold",
            refractory="v >= spike_threshold",
            method="rk4",
            namespace=namespace,
        )
        group.gNa = _BASE_G["g_Na"]
        group.gK = _BASE_G["g_K"]
        group.gKA = _BASE_G["g_KA"]
        group.gCaL = _BASE_G["g_CaL"]
        group.gh = _BASE_G["g_h"]
        group.E_Na = _E_Na
        group.E_K = _E_K
        group.E_Ca = _E_Ca
        group.E_h = _E_h
        group.E_L = _E_L
        group.G_L = _G_L
        group.C_m = _C_m
        group.spike_threshold = cls.SPIKE_THRESHOLD_MV
        return group

    def _apply_conductance_scaling(self, group: NeuronGroup) -> None:
        group.gNa = float(self.conductances["g_Na"]) * _BASE_G["g_Na"]
        group.gK = float(self.conductances["g_K"]) * _BASE_G["g_K"]
        group.gKA = float(self.conductances["g_KA"]) * _BASE_G["g_KA"]
        group.gCaL = float(self.conductances["g_CaL"]) * _BASE_G["g_CaL"]
        group.gh = float(self.conductances["g_h"]) * _BASE_G["g_h"]

    @classmethod
    def _assign_initial_state(cls, group: NeuronGroup) -> None:
        group.v = _V_REST
        group.m = cls._gate_inf(_alpha_m, _beta_m, _V_REST)
        group.h = cls._gate_inf(_alpha_h, _beta_h, _V_REST)
        group.n = cls._gate_inf(_alpha_n, _beta_n, _V_REST)
        group.a = _a_inf(_V_REST)
        group.b = _b_inf(_V_REST)
        group.d_gate = _d_inf(_V_REST)
        group.q = _q_inf(_V_REST)
        if hasattr(group, "I_drive"):
            group.I_drive = 0.0

    def _sync_from_group(self) -> None:
        if self._group is None:
            return
        if self._is_compartmental:
            # SpatialNeuron: v is in Brian2 Volts; divide by mV for the
            # dimensionless mV value expected by the rest of the interface.
            src = self._soma_section
            self._v = float(src.v[0] / mV)
            self._m = float(src.m[0])
            self._h = float(src.h[0])
            self._n = float(src.n[0])
            self._a = float(src.a[0])
            self._b = float(src.b[0])
            self._d = float(src.d_gate[0])
            self._q = float(src.q[0])
        else:
            # NeuronGroup: v is dimensionless (stored as mV numerics)
            self._v = float(self._group.v[0])
            self._m = float(self._group.m[0])
            self._h = float(self._group.h[0])
            self._n = float(self._group.n[0])
            self._a = float(self._group.a[0])
            self._b = float(self._group.b[0])
            self._d = float(self._group.d_gate[0])
            self._q = float(self._group.q[0])
        self._above_thresh = self._v >= self.SPIKE_THRESHOLD_MV
