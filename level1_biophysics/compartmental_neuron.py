"""
Level 1 — Compartmental HH neuron builder from SWC morphology.

Loaded automatically by HHNeuron.build() when a valid swc_path is provided,
pointing to a skeleton SWC file exported by import_connectome.py.

Architecture
============
* Soma compartments   (SWC type 1): full 5-conductance HH model
* Axon compartments   (SWC type 2): passive cable (leak only)
* Dendrite compartments (SWC type 3/4): passive cable (leak only)
* Drive current is injected as a current density at soma only

Biophysical cable parameters
==============================
* Cm = 1 µF/cm²     (standard HH value)
* Ri = 35.4 Ω·cm    (Drosophila antenna lobe PN; Gouwens & Wilson 2009)

Biological rationale
====================
Drosophila projection neurons have compact, electrotonically short dendrites.
Voltage-gated Na⁺/K⁺ channels are concentrated at the soma and axon initial
segment; dendritic integration is largely passive (Gouwens & Wilson 2009;
Nagel & Wilson 2011).  The active-soma / passive-cable model is therefore the
appropriate first-order compartmental approximation for fly CNS neurons.

Note on batch simulation
=========================
HHNeuron.simulate_spike_trials() still uses the fast NeuronGroup (point-neuron)
batch path, which runs n_trials neurons in parallel inside a single Brian2
NeuronGroup.  The compartmental path is used only for the per-step online
interface (build / step / reset), where a single morphologically correct
SpatialNeuron is simulated.

SWC coordinate convention
==========================
neuPrint's fetch_skeleton(format='swc') exports coordinates in micrometres (µm),
which matches Brian2's Morphology.from_file() default expectation.  No coordinate
rescaling is needed.
"""
from __future__ import annotations

import numpy as np
from brian2 import (
    Network,
    Morphology,
    SpatialNeuron,
    SpikeMonitor,
    defaultclock,
    ms,
    mV,
    uA,
    cm,
    ohm,
    siemens,
    metre,
    amp,
    uF,
    prefs,
)

from utils.accelerator import BRIAN2_TARGET
prefs.codegen.target = BRIAN2_TARGET

# ── import shared biophysical constants (single source of truth) ──────────────
from .hh_neuron import (
    _BASE_G,
    _E_Na, _E_K, _E_Ca, _E_h, _E_L,
    _G_L,
    _C_m,
    _V_REST,
    _alpha_m, _beta_m,
    _alpha_h, _beta_h,
    _alpha_n, _beta_n,
    _a_inf, _b_inf, _d_inf, _q_inf,
    _INTEGRATION_DT_MS,
)

# ── Drosophila cable constants ────────────────────────────────────────────────
_RI_OHM_CM = 35.4   # axial resistivity (Ω·cm), Gouwens & Wilson 2009

# SWC structure type codes (standard NeuroMorpho.Org / neuPrint convention)
_SWC_SOMA   = 1
_SWC_AXON   = 2
_SWC_DEND   = 3
_SWC_APICAL = 4

# ── Brian2 dimensional model equations for SpatialNeuron ─────────────────────
#
# v is in Brian2 Volts.  Use v/mV to obtain the numeric mV value needed in
# the Hodgkin-Huxley alpha/beta expressions.
#
# Im must be in amp/metre**2 (current density per membrane area).
#
# Conductances are per-compartment (S/m²); reversal potentials are shared.
# I_drive is per-compartment so the caller can inject current at soma only.
#
# Unit conversion recap:
#   mS/cm² → S/m²  :  ×10    (1 mS/cm² = 10⁻³ S / 10⁻⁴ m² = 10 S/m²)
#   µF/cm² → F/m²  :  ×10⁻²  (1 µF/cm² = 10⁻⁶ F / 10⁻⁴ m² = 0.01 F/m²)
#   µA/cm² → A/m²  :  ×10⁻²  (same area factor)

_SPATIAL_EQS = """
Im = (  gNa  * m**3 * h      * (v - E_Na)
      + gK   * n**4           * (v - E_K)
      + gKA  * a**3 * b       * (v - E_K)
      + gCaL * d_gate          * (v - E_Ca)
      + gh   * q              * (v - E_h)
      + G_L                   * (v - E_L)
      + I_drive
    ) : amp/metre**2

dm/dt = ((1.0 / exprel(-(v/mV + 40.0) / 10.0)) / ms * (1.0 - m)
         - (4.0 / ms) * exp(-(v/mV + 65.0) / 18.0) * m) : 1

dh/dt = ((0.07 / ms) * exp(-(v/mV + 65.0) / 20.0) * (1.0 - h)
         - (1.0 / ms) / (1.0 + exp(-(v/mV + 35.0) / 10.0)) * h) : 1

dn/dt = ((0.1 / exprel(-(v/mV + 55.0) / 10.0)) / ms * (1.0 - n)
         - (0.125 / ms) * exp(-(v/mV + 65.0) / 80.0) * n) : 1

da/dt = (a_inf - a) / tau_a : 1
db/dt = (b_inf - b) / tau_b : 1
dd_gate/dt = (d_inf - d_gate) / tau_d : 1
dq/dt  = (q_inf - q) / tau_q : 1

a_inf = (0.0761 * exp((v/mV + 94.22) / 31.84)
         / (1.0 + exp((v/mV + 1.17) / 28.93))) ** (1.0 / 3.0) : 1
tau_a = (0.3632 + 1.158 / (1.0 + exp((v/mV + 55.96) / 20.12))) * ms : second

b_inf = (1.0 / (1.0 + exp((v/mV + 53.3) / 14.54))) ** 4.0 : 1
tau_b = (1.24 + 2.678 / (1.0 + exp((v/mV + 50.0) / 16.027))) * ms : second

d_inf = 1.0 / (1.0 + exp(-(v/mV + 10.0) / 6.0)) : 1
tau_d = (5.0 + 20.0 * exp(-((v/mV + 25.0) / 30.0) ** 2)) * ms : second

q_inf = 1.0 / (1.0 + exp((v/mV + 75.0) / 5.5)) : 1
tau_q = (50.0 + 750.0 / (1.0 + exp(-(v/mV + 80.0) / 15.0))) * ms : second

gNa   : siemens/metre**2
gK    : siemens/metre**2
gKA   : siemens/metre**2
gCaL  : siemens/metre**2
gh    : siemens/metre**2
G_L   : siemens/metre**2
E_Na  : volt (shared)
E_K   : volt (shared)
E_Ca  : volt (shared)
E_h   : volt (shared)
E_L   : volt (shared)
I_drive : amp/metre**2
"""


# ── Public factory ─────────────────────────────────────────────────────────────

def build_compartmental_network(
    swc_path: str,
    conductances: dict,
) -> tuple[SpatialNeuron, SpikeMonitor, Network]:
    """
    Build a Brian2 SpatialNeuron from an SWC morphology file.

    The morphology is read with Brian2's Morphology.from_file(), which expects
    coordinates in micrometres (µm) — matching neuPrint's export convention.

    Parameters
    ----------
    swc_path : str
        Path to a neuPrint skeleton SWC file produced by import_connectome.py.
    conductances : dict
        Scaling factors keyed on _BASE_G names (e.g. {"g_KA": 2.0} doubles
        g_KA at the soma).

    Returns
    -------
    (neuron, spike_monitor, network)
    """
    defaultclock.dt = _INTEGRATION_DT_MS * ms

    morph = Morphology.from_file(swc_path)

    neuron = SpatialNeuron(
        morphology=morph,
        model=_SPATIAL_EQS,
        Cm=_C_m * uF / cm**2,           # 1 µF/cm² in SI
        Ri=_RI_OHM_CM * ohm * cm,        # 35.4 Ω·cm in SI
        threshold="v/mV > 0.0",          # spike detection threshold (mV)
        refractory="v/mV > 0.0",
        method="exponential_euler",
    )

    # ── reversal potentials (shared — same at all compartments) ──────────────
    neuron.E_Na = _E_Na * mV
    neuron.E_K  = _E_K  * mV
    neuron.E_Ca = _E_Ca * mV
    neuron.E_h  = _E_h  * mV
    neuron.E_L  = _E_L  * mV

    # ── start fully passive; activate HH at soma below ──────────────────────
    # mS/cm² → S/m²: multiply by 10
    _passive_S_per_m2 = 0.0 * siemens / metre**2
    neuron.gNa   = _passive_S_per_m2
    neuron.gK    = _passive_S_per_m2
    neuron.gKA   = _passive_S_per_m2
    neuron.gCaL  = _passive_S_per_m2
    neuron.gh    = _passive_S_per_m2
    neuron.G_L   = _G_L * 10.0 * siemens / metre**2
    neuron.I_drive = 0.0 * amp / metre**2

    # ── place active HH conductances at soma only ────────────────────────────
    _activate_soma_conductances(neuron, conductances)

    # ── all compartments to resting state ────────────────────────────────────
    _assign_initial_state_spatial(neuron)

    spike_monitor = SpikeMonitor(neuron)
    network = Network(neuron, spike_monitor)

    return neuron, spike_monitor, network


# ── Helpers ───────────────────────────────────────────────────────────────────

def soma_section(neuron: SpatialNeuron):
    """
    Return the soma subgroup of a SpatialNeuron.

    Brian2 names the soma section 'soma' for SWC files with type-1 nodes.
    Falls back to neuron[0:1] (first compartment) if no 'soma' attribute exists,
    which can happen with unusual SWC topologies or older Brian2 versions.
    """
    try:
        return neuron.soma
    except (AttributeError, KeyError):
        return neuron[0:1]


def _activate_soma_conductances(
    neuron: SpatialNeuron,
    conductances: dict,
) -> None:
    """
    Set HH conductances at soma; leave axon and dendrite compartments passive.

    Biological rationale: in Drosophila, voltage-gated Na⁺/K⁺ channels are
    concentrated at the soma and axon initial segment.  Dendrites are largely
    passive.  (Gouwens & Wilson 2009; Nagel & Wilson 2011)
    """
    soma = soma_section(neuron)
    # Scale factors (default 1.0 for any key absent from the dict)
    sc = {k: float(conductances.get(k, 1.0)) for k in _BASE_G}
    # mS/cm² → S/m²: multiply by 10
    soma.gNa   = sc["g_Na"]  * _BASE_G["g_Na"]  * 10.0 * siemens / metre**2
    soma.gK    = sc["g_K"]   * _BASE_G["g_K"]   * 10.0 * siemens / metre**2
    soma.gKA   = sc["g_KA"]  * _BASE_G["g_KA"]  * 10.0 * siemens / metre**2
    soma.gCaL  = sc["g_CaL"] * _BASE_G["g_CaL"] * 10.0 * siemens / metre**2
    soma.gh    = sc["g_h"]   * _BASE_G["g_h"]   * 10.0 * siemens / metre**2


def _gate_inf_val(alpha_fn, beta_fn, v: float) -> float:
    a = alpha_fn(v)
    b = beta_fn(v)
    return a / (a + b)


def _assign_initial_state_spatial(neuron: SpatialNeuron) -> None:
    """Set all compartments to resting membrane potential and gating steady state."""
    neuron.v = _V_REST * mV
    v_r = _V_REST  # numeric mV for gate computations
    neuron.m       = _gate_inf_val(_alpha_m, _beta_m, v_r)
    neuron.h       = _gate_inf_val(_alpha_h, _beta_h, v_r)
    neuron.n       = _gate_inf_val(_alpha_n, _beta_n, v_r)
    neuron.a       = _a_inf(v_r)
    neuron.b       = _b_inf(v_r)
    neuron.d_gate  = _d_inf(v_r)
    neuron.q       = _q_inf(v_r)
    neuron.I_drive = 0.0 * amp / metre**2
