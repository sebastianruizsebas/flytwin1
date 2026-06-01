# Implementation Log — 2026-05-31

## Session Summary

All MVP components stubbed in the initial scaffold were implemented in one session following the incremental-component-builder workflow. Each component was implemented one at a time with biological rationale documented before writing code.

---

## Components Implemented

### Phase 1b — `environment_sim/odor_plume.py`

**What changed:** Implemented three stubbed methods on `OdorPlume`.

| Method | Implementation |
|--------|----------------|
| `step(dt)` | Advects all active puffs with `wind_mean + Gaussian_noise * dt`; spawns Poisson-rate new puffs at source; culls puffs beyond `_max_radius` |
| `concentration_at(position)` | Vectorised sum of Gaussian puff contributions over all active puffs |
| `get_antennal_obs(ant_l, ant_r)` | Calls `concentration_at` for each antenna, applies `sigmoid_gain`, returns `{c_left, c_right, wind_vector}` |

**Biological rationale:** Intermittent Gaussian-puff model matches laboratory odor plume statistics (Murlis et al. 1992). Bilateral antennal asymmetry drives casting vs. surging decisions at the controller level.

---

### Phase 1c — `level1_biophysics/hh_neuron.py`

**What changed:** The reduced single-compartment HH implementation was rewritten to use Brian2 for integration and spike detection while preserving the existing wrapper interface.

**Key design choice:** Brian2 now owns the numerical integration, threshold detection, and spike bookkeeping through a one-neuron `NeuronGroup` plus `SpikeMonitor`. The wrapper still exposes `build()`, `step()`, `reset()`, and `spike_times`, so the closed loop remains decoupled from the simulation backend. A smaller internal integration step (0.05 ms, RK4) is used to keep the stiff conductance dynamics numerically stable while the external interface remains 1 ms.

**Five conductances implemented:**
- `g_Na` — Hodgkin-Huxley fast sodium (spike initiation)
- `g_K` — delayed rectifier (repolarisation)
- `g_KA` — A-type potassium (Connor-Stevens simplified; transient responsiveness / timing)
- `g_CaL` — L-type calcium (simplified d-gate)
- `g_h` — hyperpolarisation-activated (HCN simplified q-gate)

**Base conductance values** approximate Drosophila projection neuron literature (Wilson & Laurent 2005; Nagel & Wilson 2011).

---

### Phase 1c — `level1_biophysics/spike_collector.py`

**What changed:** `collect_spikes()` now delegates trial simulation to the Brian2-backed HH batch path and writes the resulting `SpikeMonitor`-derived spike matrices to HDF5.

- Loops over `g_KA_values`; for each value runs `n_trials` independent trials in parallel in Brian2.
- Builds one pink-noise batch per conductance condition and adds it to the shared sensory drive.
- Saves a `(n_trials, T)` uint8 spike matrix per condition to a single HDF5 file.
- Output: `data/spikes/spikes_gKA_sweep.h5`.

---

### Phase 2a — `level2_bridge/design_matrix.py`

**What changed:** Implemented `build_design_row()` and `build_design_matrix()`.

**Design matrix structure (24 columns):**
| Columns | Feature | Rationale |
|---------|---------|-----------|
| 0 | Baseline constant | Intercept / mean firing rate |
| 1–10 | Stimulus filter on u_sens | Captures odor-driven PN response timing |
| 11–20 | Spike-history filter | Refractoriness and burst adaptation |
| 21 | Heading θ | Couples neural activity to locomotor context |
| 22 | Δc = c_left − c_right | Turn-relevant plume asymmetry |
| 23 | Wind angle relative to heading | Upwind/downwind encoding |

Bell basis functions use non-linearly spaced lags (denser near lag 0) via `geomspace`.

---

### Phase 2b/2c — `level2_bridge/ppglm.py`

**What changed:** Implemented `fit_joint()`, `cross_validate_lambda()`, and added new `evaluate_online()`.

- `fit_joint()`: L-BFGS-B with smooth-L1 (Huber) approximation to the trend-filter penalty for gradient availability. Returns `(M, 24)` beta array.
- `cross_validate_lambda()`: Standard k-fold CV looping over a lambda grid.
- `evaluate_online()`: Evaluates Bernoulli log-likelihood on a short spike window using a pre-fitted beta. Called every 10 ms in the closed loop.

**Note:** Online SBI/NPE (`sbi_trainer.py`) remains a later extension as specified in `roadmap_changes.md`.

---

### Phase 3a — `level3_controller/active_inference.py`

**What changed:** Full rewrite replacing the 3-state stub with the roadmap-compliant architecture.

**Key changes vs. previous stub:**

| Aspect | Old stub | New implementation |
|--------|----------|-------------------|
| State type | Separate `position`, `heading`, etc. fields | Unified `mu`/`sigma` arrays (10D Gaussian) |
| State dimensions | ~5D, no food/obstacle | 10D: x, y, θ, c_left, c_right, Δc, w_x, w_y, d_obs, d_food |
| Task modes | 3 (near_source, lost_plume, obstructed) | 4 (SURGE, CAST, AVOID, STOP) |
| Belief update | `NotImplementedError` stub | Full predict-correct Kalman-style filter |
| NPE dependency | Required | Removed — PP-GLM log-likelihood used directly |

**Predict step:** inflates `sigma` with process noise.
**Correct (body):** Kalman gain fusion for directly observed dims.
**Correct (odor):** Kalman fusion with PP-GLM log-likelihood modulating precision.
**Task state update:** softmax over mode-specific heuristic scores derived from mu/sigma.

---

### Phase 3b/3c — `level3_controller/policy.py`

**What changed:** Full rewrite replacing flight-oriented primitives with walking commands.

**Key changes vs. previous stub:**

| Aspect | Old stub | New implementation |
|--------|----------|-------------------|
| Action enum | `SURGE_FORWARD, RETREAT, CAST_LEFT, CAST_RIGHT, ASCEND, HOVER` | `BehavioralMode.SURGE/CAST/AVOID/STOP` |
| Motor commands | `(wing_amplitude_scale, yaw_torque, vertical_thrust)` | `{forward_speed, yaw_rate, sidestep, active}` |
| EFE | `NotImplementedError` | Pragmatic KL + epistemic uncertainty reduction |
| STOP mode | Absent | Full feeder-stop logic with hard-override |

**EFE decomposition:**
- Pragmatic term: weighted squared deviation from preferred state (high odor, low d_food, safe d_obs).
- Epistemic term: negative total uncertainty on odor dimensions; CAST gets a bonus for reducing plume-direction uncertainty.
- Hard safety overrides: AVOID priority when `d_obs < 0.075 m`; STOP priority when `d_food < 0.05 m` and odor is present.

---

### Phase 4 — `run_closed_loop.py`

**What changed:** Full implementation of the simulation loop with all subsystems wired together.

**Loop structure implemented:**
1. `plume.step(dt)` every 1 ms
2. Bilateral antenna readout → `u_sens`
3. `HHNeuron.step()` → spike flag
4. Build design row; accumulate spike + row buffers
5. Every 10 ms: `evaluate_online()` → `last_log_lik`
6. Every 20 ms: `controller.update_beliefs()` → `select_action()` → `mode_to_motor_command()`
7. Kinematic plant step (MuJoCo/flybody placeholder)
8. Optional HDF5 logging; stop-at-food early termination

**flybody integration:** `_kinematic_plant_step()` is a minimal placeholder that propagates (x, y, θ) from motor commands. Replace with `env.step()` when flybody is available.

**CLI:** `python run_closed_loop.py --duration 30000 --log data/run_log.h5 --beta data/beta.npy`

---

## Roadmap Alignment Changes

All changes in `level3_controller/` align with the alignment notes in `roadmap.md`:

| File | Roadmap requirement | Status |
|------|---------------------|--------|
| `active_inference.py` | 10D walking state, 4 modes SURGE/CAST/AVOID/STOP | Done |
| `policy.py` | Walking primitives, explicit STOP mode | Done |
| `run_closed_loop.py` | PP-GLM only (no separate NPE step) | Done |
| `sbi_trainer.py` | Reclassified as later extension | Unchanged (stubs preserved) |

---

## Deferred Items (Later Extensions)

- Full NEURON compartmental morphology in `hh_neuron.py` (SWC import, multi-compartment)
- neuprint-python connectome import (`data/connectome/`)
- Full flybody / MuJoCo arena integration in `run_closed_loop.py`
- SBI/NPE training in `sbi_trainer.py`
- `sum_of_slopes.py` feature importance for conductance sweep analysis
- Arena obstacle geometry and proper d_obs computation from MuJoCo contacts
