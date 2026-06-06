# Digital Twin of *Drosophila* — Run Guide

Conductance-based neuroAI agent using the full CNS connectome (`male-cns:v0.9`)
as the neural substrate and whole-body fly mechanics as the plant.  The fly
navigates a turbulent odor plume, avoids obstacles, reaches food, and stops at
the feeder.

---

## Prerequisites

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- A neuPrint account with API access at <https://neuprint.janelia.org>
- Your neuPrint API token (do **not** commit it — store it as an environment variable)

---

## Step 1 — Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate flytwin
```

If you have a CUDA 12 GPU and want GPU-accelerated JAX, the `environment.yml`
already lists `jax[cuda12]`.  For a CPU-only machine replace that line with
`jax[cpu]` before running the command above, or install afterwards:

```bash
pip install "jax[cpu]"
```

For GPU-accelerated Brian2 simulations (optional, requires a C++ toolchain and
CUDA 12 toolkit):

```bash
pip install brian2cuda
```

---

## Step 2 — Store your neuPrint token

Set the token as an environment variable so scripts can find it automatically.

**Linux / macOS:**
```bash
export NEUPRINT_TOKEN="your_token_here"
```

**Windows (PowerShell):**
```powershell
$env:NEUPRINT_TOKEN = "your_token_here"
```

To make it permanent, add the export line to your shell profile (`~/.bashrc`,
`~/.zshrc`, or the Windows user environment variables).

Verify the token works by listing available datasets:

```bash
python import_connectome.py --list-datasets
```

---

## Step 3 — Import the full connectome

Downloads the full `male-cns:v0.9` neuron table, adjacency matrix, and
task-interface metadata into `data/connectome/`.  No `--dataset` argument is
required; `male-cns:v0.9` is the default.

```bash
python import_connectome.py
```

This may take several minutes depending on your connection.  The following files
are written:

| File | Contents |
|------|----------|
| `data/connectome/neurons.csv.gz` | Full neuron metadata table |
| `data/connectome/body_ids.npy` | Array of all body IDs |
| `data/connectome/adjacency.npz` | Sparse synaptic weight matrix |
| `data/connectome/connections.csv.gz` | Per-synapse connection table |
| `data/connectome/interface_neurons.csv.gz` | Task-interface subset (if patterns given) |

---

## Step 4 — Export SWC morphology skeletons

`run_closed_loop.py` requires a SWC skeleton file for the compartmental HH
neuron.  Export skeletons for the task-interface populations with:

```bash
python import_connectome.py --export-interface-skeletons
```

Skeleton files are written to `data/connectome/skeletons/<bodyId>.swc`.

To restrict the export to specific neuron types or instances (regex patterns),
use `--interface-type` and/or `--interface-instance`:

```bash
python import_connectome.py \
    --export-interface-skeletons \
    --interface-type "MBON.*" \
    --interface-type "ORN.*" \
    --skeleton-limit 50
```

Pick a body ID from the exported files for the next step — for example:

```bash
ls data/connectome/skeletons/
```

---

## Step 5 — Generate training data (parameter sweep)

`generate_training_data.py` sweeps across fly starting positions and conductance
conditions to produce spike trains and design matrices, then jointly fits the
PP-GLM beta vector with the trend-filter penalty.

**Default sweep** (8 positions × 45 conductance conditions = 360 runs, ~30 trials each):

```bash
python generate_training_data.py
```

**Quick smoke-test** (fewer conditions, shorter trials):

```bash
python generate_training_data.py \
    --n-positions 2 \
    --n-trials    5 \
    --trial-ms    1000 \
    --g-ka 0.5 1.0 2.0 \
    --g-na 1.0 \
    --g-cal 1.0
```

**Full research sweep** (finer conductance grid, longer trials, biologically constrained):

```bash
python generate_training_data.py \
    --n-positions 12 \
    --n-trials    50 \
    --trial-ms    10000 \
    --g-ka  0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
    --g-na  0.75 0.875 1.0 1.125 1.25 \
    --g-cal 0.5 0.75 1.0 1.5 2.0 \
    --lam   0.5
```

**Biological justification for conductance bounds** (Drosophila antennal lobe projection neurons,
Nagel & Wilson 2011; Wilson & Laurent 2005):

| Parameter | Nominal | Min scale | Max scale | Rationale |
|-----------|---------|-----------|-----------|-----------|
| `g_KA` (A-type K⁺, Shal/Kv4) | ~5 mS/cm² | **0.25×** | **2.0×** | Below 0.25× the transient repolarisation is lost and spike trains become tonic/pathological. Above 2× matches the highest measured Shal expression in identified PNs. Removing the 0.1× value from the old sweep avoids a near-complete knock-out that is not seen in healthy flies. |
| `g_Na` (fast Na⁺, para/NaV1) | ~120 mS/cm² | **0.75×** | **1.25×** | Below 0.75× (90 mS/cm²) spike initiation fails intermittently under the plume-drive currents used here (~6–10 µA/cm²). Above 1.25× (150 mS/cm²) generates high-frequency burst artefacts not observed in whole-cell PN recordings. ±25% covers the neuron-to-neuron variability reported in Drosophila slice data. |
| `g_CaL` (L-type Ca²⁺, Dmca1D/Cav1) | ~2 mS/cm² | **0.5×** | **2.0×** | Below 0.5× (1 mS/cm²) the slow calcium-dependent plateau is essentially absent. The 0.25× value in the old sweep removed nearly all L-type calcium, which is inconsistent with the ubiquitous expression of Dmca1D in Drosophila PNs. |

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--out-dir PATH` | `data/spikes` | Output directory |
| `--n-positions N` | `8` | Number of fly starting positions |
| `--n-trials N` | `30` | Noise realisations per (position, condition) |
| `--trial-ms N` | `5000` | Trial duration in ms |
| `--g-ka SCALES…` | `0.25 0.5 1.0 1.5 2.0` | g_KA scale values (min 0.25× — near-absent Shal is not biologically realistic) |
| `--g-na SCALES…` | `0.8 1.0 1.2` | g_Na scale values (keep within 0.75–1.25× for reliable spiking) |
| `--g-cal SCALES…` | `0.5 1.0 2.0` | g_CaL scale values (min 0.5× — Dmca1D is constitutively expressed in PNs) |
| `--lam FLOAT` | `1.0` | Trend-filter penalty strength |
| `--seed INT` | `42` | Global RNG seed |
| `--quiet` | off | Suppress per-condition progress |

**Output files written to `data/spikes/`:**

| File | Contents |
|------|----------|
| `beta.npy` | **(24,) nominal beta** — pass to `run_closed_loop.py --beta` |
| `betas_all.npy` | (M, 24) beta matrix across all M conductance conditions |
| `cond_grid.npy` | (M, 3) table of `[g_KA, g_Na, g_CaL]` scale values |
| `training_data.h5` | Full spike + design-matrix archive per condition (requires h5py) |

**What the sweep varies and why:**

- **Fly position** (upwind distance and lateral offset) — samples the full dynamic
  range of odor drive including plume-edge crossings that load the bilateral
  gradient column (Δc) in the design matrix.
- **g_KA** (A-type potassium, Shal/Kv4 family) — controls transient repolarisation, spike
  timing, and adaptation. Sweeps from 0.25–2.0× the nominal 5 mS/cm², covering the range of
  Shal expression measured across identified *Drosophila* PNs. The lower bound (0.25×)
  preserves a small but measurable transient K⁺ current; removing it entirely (0.1×) produces
  non-biological runaway firing.
- **g_Na** (fast sodium, para/NaV1) — sets spike threshold and gain. Sweeps 0.75–1.25× the
  nominal 120 mS/cm². Below 0.75× (90 mS/cm²) spike initiation becomes unreliable under
  physiological drive currents; above 1.25× (150 mS/cm²) generates high-frequency burst
  artefacts not seen in whole-cell *Drosophila* PN recordings (Nagel & Wilson 2011).
- **g_CaL** (L-type calcium, Dmca1D/Cav1) — adds slower depolarising dynamics and shapes
  burst statistics. Sweeps 0.5–2.0× the nominal 2 mS/cm². The 0.5× lower bound preserves
  the calcium-dependent plateau that is functionally important for SURGE/CAST mode transitions;
  0.25× (the previous minimum) was sub-physiological given the ubiquitous Dmca1D expression
  in *Drosophila* projection neurons.

---

### How the PP-GLM is fit from sensory stimuli

The script implements this pipeline for every (position, conductance) combination:

```
OdorPlume.step()              → c_left(t), c_right(t)   [antenna concentrations]
    ↓  mean + 1/f noise drive
HHNeuron.simulate_spike_trials() → spikes(n_trials, T)  [Level 1 output]
    ↓  build_design_matrix()
X (T × 24)                       [design matrix per trial]
    ↓  fit_joint([X_cond1, X_cond2, …], [y_cond1, …], lam)
beta (24,)                       [PP-GLM filter coefficients]
```

**The 24 design matrix columns and what sensory stimulus each encodes:**

| Column(s) | Name | Sensory signal captured |
|-----------|------|--------------------------|
| 0 | Intercept | Baseline firing rate (no stimulus) |
| 1–10 | Stimulus filter | Past odor drive `u_sens(t-τ)` at 10 log-spaced lags τ ∈ [0, 200 ms]. The fitted weights reveal the integration window: a narrow peak near τ=0 means fast transduction; a broader peak means temporal summation. |
| 11–20 | Spike-history filter | Past spikes at 10 linearly spaced lags τ ∈ [1, 100 ms]. Negative weights at short lags = refractoriness; positive at longer lags = burst facilitation. |
| 21 | Heading θ | Body orientation — allows spike rate to vary with direction of travel relative to the plume. |
| 22 | Bilateral gradient Δc | `c_left − c_right` — non-zero only when the fly is at a plume edge. Lateral starting positions in the sweep ensure this column is exercised. |
| 23 | Wind angle | Direction of mean wind relative to heading — encodes upwind/downwind cues. |

**Why position diversity matters for stimulus filter quality:**

If the fly always starts on the plume centreline, the stimulus filter only sees
high-concentration, symmetric encounters.  The stimulus filter columns (1–10)
will over-fit to sustained drive and miss the onset/offset dynamics.  Starting
positions at varied upwind distances and lateral offsets force the filter to
learn from both strong pulses (near source, on-axis) and sparse whiffs
(far upwind, off-axis), giving a filter that generalises across the full
plume encounter statistics the closed-loop agent will experience.

**How `fit_joint` ties the conductance sweep together:**

`fit_joint` fits one beta per conductance condition simultaneously.  The
trend-filter penalty (`--lam`) penalises abrupt changes between adjacent
conditions in the conductance grid:

$$\mathcal{L}(\beta_1,\ldots,\beta_M) = -\sum_i \log P(\text{data}_i \mid \beta_i) + \lambda \sum_i \|\beta_{i+1} - \beta_i\|_1$$

This means the resulting `betas_all.npy` matrix smoothly interpolates the
PP-GLM filter shape across the conductance landscape — so you can later
extract a beta for any (g_KA, g_Na, g_CaL) combination without re-fitting.
The nominal beta saved to `beta.npy` corresponds to biologically standard
conductances (all scale factors = 1.0).

---

## Step 6 — Run the closed-loop simulation

Pass the path to any exported SWC skeleton via `--swc`.  Substitute the body ID
with one from `data/connectome/skeletons/`.

**Minimal run (60 seconds simulated, no logging):**

```bash
python run_closed_loop.py \
    --swc data/connectome/skeletons/<bodyId>.swc
```

**With HDF5 logging, a fitted beta vector, and data-grounded A matrix:**

```bash
python run_closed_loop.py \
    --swc  data/connectome/skeletons/<bodyId>.swc \
    --beta data/spikes/beta.npy \
    --h5   data/spikes/training_data.h5 \
    --log  data/spikes/run_001.h5 \
    --duration 120000
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--swc PATH` | *(required)* | SWC skeleton file for the compartmental HH neuron |
| `--duration MS` | `60000` | Simulation duration in milliseconds |
| `--beta PATH` | `None` | Pre-fitted `.npy` PP-GLM beta vector (24-dim) |
| `--h5 PATH` | `None` | `training_data.h5` — calibrates pymdp A matrices from data via empirical Bayes.  Recommended: `data/spikes/training_data.h5` |
| `--log PATH` | `None` | HDF5 output file for full state/spike log |
| `--quiet` | off | Suppress per-ms progress output |

The simulation prints mode selections (`SURGE`, `CAST`, `AVOID`, `STOP`) and
position updates every 1000 ms.  A summary of final position, food distance, and
mode history is printed at completion.

---

## Step 7 — Inspect results

**Closed-loop run log:**

```python
import h5py, numpy as np, matplotlib.pyplot as plt

with h5py.File("data/spikes/run_001.h5") as f:
    positions = f["positions"][:]
    modes     = f["modes"][:]
    log_liks  = f["log_liks"][:]

plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.title("Agent trajectory")
plt.show()
```

**Training data archive (per-condition spike statistics):**

```python
import h5py, numpy as np

with h5py.File("data/spikes/training_data.h5") as f:
    # List all position / condition groups
    for pos_key in list(f.keys()):
        if pos_key == "fit":
            continue
        for cond_key in f[pos_key]:
            grp = f[pos_key][cond_key]
            rate = grp.attrs["spike_rate_Hz"]
            print(f"{pos_key}/{cond_key}  {rate:.1f} Hz")

    # Load the fitted beta vectors across all conductance conditions
    betas = f["fit/betas_all"][:]   # (M, 24)
    conds = f["fit/cond_grid"][:]   # (M, 3) — [g_KA, g_Na, g_CaL]

# Plot how the stimulus filter norm changes with g_KA
g_ka_vals  = conds[:, 0]
stim_norms = np.linalg.norm(betas[:, 1:11], axis=1)
plt.scatter(g_ka_vals, stim_norms)
plt.xlabel("g_KA scale"); plt.ylabel("stimulus filter norm")
plt.title("How A-type potassium shifts the odor filter")
plt.show()
```

---

## Architecture quick reference

```
Level 1 — Biophysics (1 ms)
    HH neuron on SWC compartmental morphology → spike trains
        ↓  PP-GLM likelihood bridge (10 ms)
Level 2 — Body–Environment
    10D Gaussian belief [x, y, θ, c_L, c_R, Δc, w_x, w_y, d_obs, d_food]
        ↓  EFE policy selection (20 ms)
Level 3 — Task
    Discrete mode: SURGE | CAST | AVOID | STOP → motor commands → flybody
```

The neural substrate is the `male-cns:v0.9` full connectome.  Higher levels
**never** read membrane voltages directly — only the PP-GLM likelihood summary
crosses the Level 1 → Level 2 boundary.

---

## Accelerator detection

At startup, `utils/accelerator.py` auto-detects the best available backend and
prints a one-line summary:

| Condition | `DEVICE` | `BRIAN2_TARGET` | Effect |
|-----------|----------|-----------------|--------|
| CUDA GPU + `jax[cuda12]` + `brian2cuda` | `gpu` | `cuda_standalone` | Full GPU path |
| CUDA GPU + `jax[cuda12]`, no brian2cuda | `gpu` | `cython` (if g++ found) else `numpy` | JAX on GPU, Brian2 on CPU JIT |
| No GPU, JAX installed | `cpu-jax` | `cython` / `numpy` | JAX JIT on CPU |
| No JAX | `cpu` | `cython` / `numpy` | Pure numpy |

No manual configuration is needed — the correct backend is chosen automatically.
