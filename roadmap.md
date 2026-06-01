# Digital Twin of *Drosophila* — Implementation Roadmap
**FSU / Janelia Cech Fellowship, Summer 2026**

---

## Project Goal

Construct a conductance-based neuroAI agent of a fruit fly that uses the **full connectome** as its neural substrate and **whole-body fly mechanics** as its plant, then navigates a turbulent odor plume through obstacles, reaches a food source, and stops at the food. The novelty of the project is not a toy odor-tracking controller by itself; it is the coupling of full-brain wiring, biophysical neural dynamics, and whole-body mechanics in one closed loop.

---

## Three-Level Architecture

| Level | Name | Timescale | State representation | Model |
|-------|------|-----------|---------------------|-------|
| 3 | Abstract Task | ~100 ms – s | **Discrete** — 4 behavioral modes | EFE policy selection over $\pi \in \{\texttt{SURGE, CAST, AVOID, STOP}\}$ |
| 2 | Body–Environment | ~10 – 50 ms | **Continuous** — 10D walking-task state vector $\mathbf{s}^{(2)} \in \mathbb{R}^{10}$ | PP-GLM likelihood bridge + Gaussian belief |
| 1 | Biophysics | ~1 – 5 ms | **Continuous** — conductance-based neural dynamics on the connectome | Hodgkin-Huxley / reduced conductance models, exposed upward **only** via PP-GLM likelihood |

For the first implementation, the bridge is a **PP-GLM likelihood model**. Online conductance inference with SBI/NPE is optional later; it is not required for the first food-seeking closed-loop agent.

### State Space Specification

**Level 3 — four discrete behavioral modes** (`level3_controller/policy.py : BehavioralMode`)

| Mode | Trigger condition | EFE character |
|------|-------------------|---------------|
| `SURGE` | Odor present and the agent is confidently oriented along the plume | Pragmatic-dominant: move toward the food source |
| `CAST` | Plume contact is weak or lost and source direction is uncertain | Epistemic-dominant: lateral search to re-acquire plume |
| `AVOID` | Obstacle proximity $d_\text{obs} < d_\text{thresh}$ | Pragmatic: minimize collision risk |
| `STOP` | Food is reached: $d_\text{food} < d_\text{stop}$ and odor remains high | Pragmatic: terminate locomotion at the goal |

EFE is evaluated independently for each of the 4 modes every 20–50 ms. $\pi^* = \arg\min G(\pi)$.

**Level 2 — 10-dimensional continuous walking-task state vector** (`level3_controller/active_inference.py : BodyEnvState`)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | $x$ | Body position — forward axis | m |
| 1 | $y$ | Body position — lateral axis | m |
| 2 | $\theta$ | Heading angle (yaw) | rad |
| 3 | $c_\text{left}$ | Left-antenna odor concentration | a.u. |
| 4 | $c_\text{right}$ | Right-antenna odor concentration | a.u. |
| 5 | $\Delta c$ | Bilateral odor gradient $c_\text{left} - c_\text{right}$ | a.u. |
| 6 | $w_x$ | Wind velocity — forward component | m/s |
| 7 | $w_y$ | Wind velocity — lateral component | m/s |
| 8 | $d_\text{obs}$ | Distance to nearest obstacle | m |
| 9 | $d_\text{food}$ | Distance to the food target / feeder | m |

The high-level task is a **walking** navigation problem, so vertical position is not part of the Level 2 belief state even though the whole-body plant remains 3D. The belief $q(\mathbf{s}^{(2)})$ is maintained as a Gaussian $\mathcal{N}(\boldsymbol{\mu}^{(2)}, \boldsymbol{\Sigma}^{(2)})$ with diagonal covariance, updated by a predict-correct cycle each 20 ms.

**Level 1 — conductance-based connectome dynamics, PP-GLM interface only**

Level 1 remains the neural substrate of novelty: the full connectome provides the wiring scaffold, and conductance-based dynamics are assigned first to task-relevant neuron classes embedded within that scaffold. Higher levels never read membrane voltages or gating variables directly. The sole exported quantity from Level 1 is the PP-GLM likelihood summary derived from the spike buffer.

---

## Key Software Stack

| Package | Role |
|---------|------|
| Python 3.10 | Primary language (`conda env: flytwin`) |
| NEURON | Conductance-based simulation of connectome neurons and readout populations |
| neuprint-python | Whole-connectome queries and morphology import |
| MuJoCo + dm_control | Whole-body physical simulation and odor arena |
| flybody (TuragaLab) | Whole-body fly mechanics with contact-rich locomotion |
| numpy / scipy | GLM fitting, design matrix construction, filtering, analysis |
| statsmodels | GLM baselines and likelihood checks |
| h5py | Spike, state, and evaluation logging |

> `sbi` remains a later extension for conductance inversion or adaptive physiology, but it is not part of the minimum viable closed-loop agent.

---

## Phase 0: Environment and Repository Setup

### 0a) Conda Environment

```bash
conda env create -f environment.yml
conda activate flytwin
pip install neuron
pip install neuprint-python
```

Obtain a neuPrint API token from <https://neuprint.janelia.org> and store it in a local `.env` file. **Never commit tokens or credentials to version control.**

### 0b) Repository Structure *(incremental, MVP-first)*

```
firstrepoMay31/
├── environment.yml
├── data/
│   ├── connectome/          # full-connectome adjacency, neuron tables, morphologies
│   └── spikes/              # simulated spike-train HDF5 files
├── level1_biophysics/
│   ├── hh_neuron.py         # conductance-based neuron wrapper
│   └── spike_collector.py
├── level2_bridge/
│   ├── design_matrix.py     # PP-GLM feature engineering
│   ├── ppglm.py             # PP-GLM fitting and likelihood evaluation
│   ├── sum_of_slopes.py     # optional later analysis for conductance sweeps
│   └── sbi_trainer.py       # optional later extension, not MVP-critical
├── level3_controller/
│   ├── active_inference.py  # Gaussian Level 2 belief + discrete Level 3 belief
│   └── policy.py
├── environment_sim/
│   ├── odor_plume.py        # plume, feeder, obstacle arena
│   └── arena.xml            # MuJoCo scene descriptor
└── run_closed_loop.py       # main simulation entry point
```

---

## Phase 1: Full Connectome + Whole-Body Plant

### 1a) Full Connectome and Morphological Import

- **Whole connectome is a non-negotiable part of the project novelty.** Start from the most complete female connectome compatible with the body model and tooling. Do not reduce the roadmap to a hand-picked microcircuit plus toy controller.
- Import the full adjacency matrix $W$, neuron metadata, and available morphologies. Persist the full graph under `data/connectome/`.
- For the first closed-loop agent, define a **task interface layer** on top of the full connectome:
  - olfactory sensory input populations
  - obstacle-related sensory populations or readouts
  - descending / motor readout populations
- The interface layer is a readout of the full substrate, not a replacement for it.

```python
from neuprint import Client, fetch_neurons, fetch_adjacencies
import os

c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1',
           token=os.environ['NEUPRINT_TOKEN'])
# Fetch the full graph and task-interface neuron sets.
# Save adjacency W and morphology assets under data/connectome/
```

- **Validate:** imported neuron tables must preserve body IDs, class labels, and the interface-population definitions used later by the bridge and controller.

### 1b) Whole-Body Odor Arena in MuJoCo / flybody

- Use the whole-body `flybody` mechanics as the plant. The first task is **walking** to food, not flight; prune flight-specific planning until the walking plume-tracking loop works end to end.
- Build a ground-plane arena containing:
  - a turbulent odor plume
  - rigid obstacles
  - a food target / feeder with a stop radius or contact zone
- Expose the minimal observation channels needed by the controller:
  - bilateral antennal odor concentration
  - wind vector
  - obstacle proximity
  - food-target distance or contact signal

Implement plume concentration in `environment_sim/odor_plume.py`:

$$c(\mathbf{x}, t) = \sum_k A_k \exp\!\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}_k(t)\|^2}{2\sigma_k^2}\right)$$

Puffs $\boldsymbol{\mu}_k$ advect with mean wind vector $\mathbf{w}_t$ plus Gaussian noise. Antennal readout remains:

$$\mathbf{u}_t^{\text{sens}} = [c_{\text{left}}(t),\ c_{\text{right}}(t),\ \mathbf{w}_t]$$

### 1c) Conductance-Based Neural Simulation on the Connectome Substrate

- The project is **conductance-based**, but the first implementation should avoid the unnecessary requirement of full compartmental Hodgkin-Huxley detail for every neuron in the connectome.
- Keep the **full connectome wiring scaffold**, but assign detailed conductance-based models first to the task-critical neuron classes embedded in that scaffold:
  - sensory input populations
  - key intermediate populations for plume tracking / obstacle response
  - descending readout populations
- The rest of the graph may initially use reduced conductance units or fixed relay dynamics while preserving the connectome topology. This keeps the novelty while keeping the first agent tractable.

Key conductances to parameterize:

| Conductance | Role |
|-------------|------|
| $g_{Na}$ | Fast sodium — spike initiation |
| $g_K$ | Delayed rectifier — repolarization |
| $g_{KA}$ | A-type potassium — transient responsiveness / timing |
| $g_{CaL}$ | L-type calcium — slower depolarizing dynamics |
| $g_h$ | Hyperpolarization-activated current |

**Trial generation protocol:**
- For each conductance condition $g_i$ (for example $g_{KA} \in [0.5, 2.0]$), inject odor-driven input plus $1/f$ noise to mimic biological variability.
- Record binary spike trains $o_t^{\text{neural}}$ at 1 ms resolution.
- Save to HDF5 under `data/spikes/`.

---

## Phase 2: PP-GLM Likelihood Bridge (MVP)

### 2a) Design Matrix Construction — `level2_bridge/design_matrix.py`

The design matrix $x_t$ determines what Level 1 information is exposed to the controller. For the first agent, keep it compact and directly tied to plume-tracking behavior.

| Column(s) | Feature | Biological / control motivation |
|-----------|---------|--------------------------------|
| 1 | Baseline constant | Baseline firing rate |
| 2–11 | Stimulus filter on $\mathbf{u}_t^{\text{sens}}$ | Encodes odor-driven sensory timing |
| 12–21 | Spike-history filter on $h_t$ | Encodes refractoriness and recent neural history |
| 22 | Heading $\theta_t$ | Couples neural activity to locomotor context |
| 23 | Bilateral odor gradient $\Delta c_t$ | Turn-relevant plume asymmetry |
| 24 | Food-target distance or plume-confidence scalar | Makes stopping at the feeder identifiable at the controller level |

This yields $\boldsymbol{\beta} \in \mathbb{R}^{24}$ per neuron or readout channel.

### 2b) PP-GLM Fitting with Optional Trend Filtering — `level2_bridge/ppglm.py`

Spike probability:

$$p(\text{spike at } t) = \sigma\!\left(\mathbf{x}_t^\top \boldsymbol{\beta}\right)$$

If multiple conductance conditions are fit jointly, use the trend-filter penalty:

$$\mathcal{L}(\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_M) = -\sum_i \log P(\text{data}_i \mid \boldsymbol{\beta}_i) + \lambda \sum_i \|\boldsymbol{\beta}_{i+1} - \boldsymbol{\beta}_i\|_1$$

Trend filtering is useful when comparing conductance sweeps, but it is not the central novelty of the first agent. The must-have outcome is a stable PP-GLM that produces a usable likelihood interface from spikes to controller.

### 2c) Online Likelihood Interface — `level2_bridge/ppglm.py`

At runtime, the bridge evaluates the recent spike buffer and returns a log-likelihood update for the odor- and goal-relevant dimensions of $q(\mathbf{s}^{(2)})$.

- Every 10 ms, evaluate the PP-GLM on the recent spike window.
- Convert the result into a likelihood or log-likelihood term.
- Use that term directly in the Level 2 belief update.

The first agent does **not** need online conductance inversion. Conductance changes can be introduced offline and compared behaviorally.

### 2d) Optional Later Extension — SBI / NPE

`sum_of_slopes.py` and `sbi_trainer.py` remain useful if the project later expands to infer conductances from behavior or spike summaries. That is a valid extension, but it is not required to demonstrate the core novelty of a full-connectome, whole-body, conductance-based odor-guided agent.

---

## Phase 3: Active Inference Controller

### 3a) Belief State Representation — `level3_controller/active_inference.py`

The controller maintains factored beliefs at two levels:

- $q(\mathbf{s}^{(2)})$ — **continuous** 10D Gaussian belief $\mathcal{N}(\boldsymbol{\mu}^{(2)}, \boldsymbol{\Sigma}^{(2)})$ over $[x, y, \theta, c_\text{left}, c_\text{right}, \Delta c, w_x, w_y, d_\text{obs}, d_\text{food}]$.
- $q(\mathbf{s}^{(3)})$ — **discrete** categorical distribution over 4 behavioral modes: `SURGE`, `CAST`, `AVOID`, `STOP`.

Level 1 biophysical variables are **not** inferred online and are **never** read directly by Levels 2–3. The PP-GLM bridge converts the neural spike buffer into a likelihood update for the odor- and goal-relevant dimensions of $q(\mathbf{s}^{(2)})$.

**Belief update rule (per 20 ms cycle):**
1. **Predict:** propagate $\boldsymbol{\mu}^{(2)}$ with a walking kinematic model using the last motor command.
2. **Correct (body/environment):** fuse MuJoCo observation $[x, y, \theta, d_\text{obs}, d_\text{food}]$.
3. **Correct (neural bridge):** use the PP-GLM likelihood to update the odor-related dimensions $(c_\text{left}, c_\text{right}, \Delta c)$.
4. **Update Level 3:** re-compute the mode probabilities from the current mean and uncertainty, with `STOP` rising when $d_\text{food}$ is small and odor remains high.

### 3b) Expected Free Energy (EFE) Minimization — `level3_controller/policy.py`

Policy selection minimizes Expected Free Energy over the 4 discrete behavioral modes:

$$G(\pi) = \underbrace{\mathrm{KL}\!\left[q(\mathbf{s}^{(2)}\mid\pi) \,\|\, p(\mathbf{s}^{(2)})\right]}_{\text{pragmatic}} - \underbrace{\mathbb{E}_q\!\left[\log p(o^\text{neural} \mid \mathbf{s}^{(2)}, \pi)\right]}_{\text{epistemic}}$$

- **Pragmatic term:** prefer states that approach food, stay collision-free, and eventually stop at the feeder.
- **Epistemic term:** prefer actions that reduce uncertainty about plume direction when odor contact is weak.

| Mode | Role | Trigger / interpretation |
|------|------|--------------------------|
| `SURGE` | Move toward source | Strong odor and low heading uncertainty |
| `CAST` | Search across plume | Weak odor and high plume-direction uncertainty |
| `AVOID` | Prevent collision | Obstacle distance too small |
| `STOP` | Terminate locomotion at food | Food reached or feeder contact established |

### 3c) Action Mapping to flybody — `level3_controller/policy.py`

The first agent is a **walking** agent. Replace flight-oriented primitives with walking primitives and whole-body leg actuation.

| Level 3 mode | Level 2 primitive(s) | Whole-body command |
|--------------|----------------------|--------------------|
| `SURGE` | `walk_forward` | Increase forward stepping with symmetric leg drive |
| `CAST` | `turn_left` / `turn_right` | Alternate yaw-biased stepping to sweep laterally |
| `AVOID` | `sidestep` + `turn_away` | Re-route around obstacles while preserving plume contact |
| `STOP` | `stop` | Halt stepping and maintain stable body posture at feeder |

The whole-body `flybody` model remains critical: the controller sets high-level walking commands, while the plant handles contact-rich leg mechanics, posture, and collisions.

---

## Phase 4: Closed-Loop Integration and Validation

### 4a) Temporal Scheduling — `run_closed_loop.py`

| Subsystem | Update interval | Module |
|-----------|----------------|--------|
| Level 1 conductance dynamics | 1 – 5 ms | `level1_biophysics/hh_neuron.py` |
| PP-GLM likelihood bridge | 10 ms | `level2_bridge/ppglm.py` |
| Controller (belief + EFE) | 20 ms | `level3_controller/active_inference.py` |
| MuJoCo / flybody physics | 1 – 5 ms internal, exposed at controller cadence | `environment_sim/odor_plume.py` |

Master clock remains 1 ms. The controller only needs the bridge and body observations; no separate NPE step is required for the first agent.

### 4b) The Full Closed Loop

Each iteration of `run_closed_loop.py` executes in order:

1. Advance `flybody` in MuJoCo.
2. Read odor, wind, obstacle, and food-target observations.
3. Inject sensory drive into the conductance-based neural substrate and collect spikes.
4. Every 10 ms: evaluate the PP-GLM on the spike buffer and obtain a likelihood update.
5. Every 20 ms: update the Level 2 Gaussian belief and Level 3 mode probabilities.
6. Evaluate EFE for `SURGE`, `CAST`, `AVOID`, and `STOP`.
7. Map the selected mode to walking primitives and apply them to the whole-body plant.
8. Log position, heading, odor, obstacle distance, food distance, spikes, PP-GLM outputs, selected mode, and success / failure flags.

### 4c) Validation and Success Criteria

**Level 1 (conductance-based neural substrate):**
- Spike statistics remain stable and biologically plausible across the chosen conductance range.
- Conductance perturbations produce measurable changes in neural response timing or gain.

**Bridge (PP-GLM):**
- Held-out log-likelihood is better than a baseline intercept-only model.
- Online PP-GLM evaluation completes comfortably within the 10 ms bridge budget.
- The bridge output changes systematically with conductance perturbations and plume conditions.

**Closed-loop whole-agent behavior:**
- The agent follows the odor plume to the feeder in the presence of obstacles.
- The agent avoids collisions or keeps collision rate below a defined threshold.
- The agent enters the feeder zone and **stops at food** for a sustained dwell period.
- Conductance perturbations shift navigation performance in interpretable ways (time to food, path length, cast frequency, stop latency).

---

## Key Design Decisions and Trade-offs

### 1. Full connectome and whole-body mechanics are the core novelty
Do not prune the project into a small hand-crafted circuit plus point-mass controller. The connectome and the whole-body plant are the distinctive scientific contribution and must remain central.

### 2. Walking before flight
For the first odor-to-food agent, walking is the right task. It preserves the whole-body novelty while pruning unnecessary flight-control complexity.

### 3. PP-GLM first, SBI later
The first closed-loop agent only needs a reliable PP-GLM likelihood interface from Level 1 spikes to the controller. Online SBI / NPE is optional and should not block the first demonstration.

### 4. Conductance-based detail where it matters first
The full connectome remains the scaffold, but biophysical detail should be assigned first to task-relevant populations rather than every neuron at once.

### 5. Female connectome consistency
Use a female connectome and the female `flybody` reconstruction to avoid anatomy mismatches between neural substrate and mechanical plant.

### 6. Decoupling Level 1 from Levels 2–3
Higher levels consume only the bridge likelihood summary, not raw NEURON state variables. This keeps the controller tractable and modular.

### 7. Discrete Level 3 / Continuous Level 2 split
This remains the correct split for the MVP: discrete behavioral modes at Level 3 and a compact continuous Gaussian belief at Level 2.

---

## Code Alignment Notes

The roadmap above implies the following pruning and alignment changes to the current stub structure:

| File | Current state | Required roadmap-aligned change |
|------|--------------|----------------------------------|
| `level3_controller/active_inference.py` | 3-state task belief and 3D position-centric body state | Replace with walking-task 10D belief over `[x, y, theta, c_left, c_right, delta_c, w_x, w_y, d_obs, d_food]` and 4 modes `SURGE`, `CAST`, `AVOID`, `STOP` |
| `level3_controller/policy.py` | Flight-oriented primitives like `ascend` / `hover` | Replace with walking primitives such as `walk_forward`, `turn_left`, `turn_right`, `sidestep`, `stop` |
| `level3_controller/policy.py` | No explicit food-stop mode | Add `STOP` mode and feeder-stop logic |
| `level2_bridge/sbi_trainer.py` | Implies NPE as part of the main loop | Reclassify as optional later extension, not first-pass dependency |
| `run_closed_loop.py` | Contains a separate NPE step in the closed loop | Simplify the first loop to PP-GLM likelihood update only |
| `environment_sim/odor_plume.py` | Arena described generically | Ensure the first arena includes obstacles, feeder target, and explicit stop-at-food condition |

---

## Conceptual Overview

This project should be framed as a **conductance-based whole-animal neuroAI system**, not as an inference benchmark with a fly wrapper around it. The full connectome is the neural substrate, the whole-body `flybody` mechanics are the plant, and the first behavioral target is a walking fly that follows odor plumes, avoids obstacles, reaches food, and stops at the feeder.

The roadmap therefore keeps the two scientifically distinctive components — full-brain wiring and whole-body mechanics — and prunes away first-pass complexity that is not needed to prove the core idea. The bridge is a PP-GLM likelihood model, not an online conductance-inference engine. The controller reasons over a compact walking-task state, not a full latent biophysical posterior. Conductance perturbations still matter, but they are used first to test how neural physiology shifts behavior, not to force an unnecessary online inversion problem.

If the first implementation succeeds, the result is already scientifically meaningful: a fruit fly agent whose plume-tracking and obstacle-avoidance behavior arises from the interaction of a connectome-based conductance substrate, a whole-body mechanical plant, and an active inference controller. Later additions such as SBI, richer physiology, or flight can be layered on top of that core system instead of blocking it.
