# Full Active Inference Implementation Steps

## Purpose

This document describes the methodology to fully implement the four parts of an
active inference controller in this repository while preserving the project's
core architecture:

1. Likelihood $P(o \mid s)$
2. Prior over hidden state $P(s)$
3. Transition model $P(s' \mid s, a)$
4. Policy prior / posterior $P(\pi)$ and $q(\pi)$

The implementation should remain consistent with the current research framing:

- the full connectome remains the long-term neural substrate
- the whole-body fly / MuJoCo plant remains the plant
- the PP-GLM remains the first bridge from Level 1 spikes to higher-level state inference
- active inference upgrades the controller logic, not the scientific novelty of the project

The target outcome is a controller that performs approximate active inference
over the existing 10D walking-task hidden state and 4 discrete behavioral
policies: `SURGE`, `CAST`, `AVOID`, `STOP`.

---

## Current Status in the Repo

The current code already contains some of the required pieces, but they are not
yet assembled into a full generative active inference loop.

### Already present

- `level2_bridge/ppglm.py`
  - spike-based likelihood machinery for odor-related latent variables
  - `infer_odor_posterior(...)` already converts recent spikes into a posterior
    over `[c_left, c_right, delta_c]`
- `level3_controller/active_inference.py`
  - Gaussian belief state over the 10D Level 2 state
  - correction step for body observations and spike-derived odor posterior
- `level3_controller/policy.py`
  - four discrete behavioral modes
  - approximate expected free energy scoring per mode
- `run_closed_loop.py`
  - control loop timing and data flow between body, plume, spikes, and policy

### Not yet fully implemented

- a single explicit observation model $P(o \mid s)$ that combines body and neural observations
- a proper action-conditioned transition model inside the belief update
- a clean distinction between filtering priors over hidden state and preferred outcomes
- an explicit policy prior and posterior over policies
- policy selection driven by $q(\pi)$ rather than a direct greedy argmin

---

## Implementation Philosophy

The correct first implementation is **approximate active inference**, not a full
symbolic or message-passing treatment.

For this repository, the practical choice is:

- **continuous hidden state**: diagonal-Gaussian approximation over the 10D Level 2 state
- **discrete policy set**: four policies / modes
- **neural likelihood**: PP-GLM-based likelihood from recent spike buffers
- **body likelihood**: Gaussian observation model over kinematic and task observations
- **state prediction**: action-conditioned kinematic transition with process noise
- **policy inference**: softmax posterior over policies using expected free energy

This keeps the controller mathematically aligned with active inference while
remaining computationally lightweight enough for the 20 ms control loop.

---

## Python Imports and Why They Are Needed

The implementation does not require a large new dependency stack. Most of what
is needed already exists in the environment.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp, softmax

from level2_bridge.ppglm import OdorPosterior, infer_odor_posterior
from level3_controller.active_inference import BehavioralMode
```

### Import rationale

- `dataclasses`
  - for structured model objects such as `StateBelief`, `ObservationModel`,
    `TransitionModel`, and `PolicyPosterior`
- `enum.IntEnum`
  - to keep the discrete policy set explicit and stable across the controller
- `typing`
  - to make model interfaces readable and safe when passing observation and action dictionaries
- `numpy`
  - state vectors, diagonal covariances, log-likelihood evaluation, and policy scoring
- `scipy.special.logsumexp`
  - numerically stable normalization for state-grid or policy posterior computations
- `scipy.special.softmax`
  - policy posterior normalization from log-probabilities or negative EFE values
- `level2_bridge.ppglm.OdorPosterior`
  - existing structured output of the spike bridge
- `level2_bridge.ppglm.infer_odor_posterior`
  - existing PP-GLM-based neural likelihood interface

### Optional imports

```python
import h5py
import jax.numpy as jnp
from utils.accelerator import HAS_JAX, jit
```

- `h5py`
  - optional for logging belief trajectories, policy posteriors, and diagnostic terms
- `jax.numpy` / `jit`
  - optional if multi-step policy rollouts or gradient-based parameter tuning become expensive
  - not required for the first full active inference implementation

No new mandatory package is required beyond `numpy` and `scipy` for the core controller update.

---

## Hidden State and Observation Definitions

### Hidden state

Retain the existing 10D Level 2 hidden state:

$$
\mathbf{s}_t = [x, y, \theta, c_{left}, c_{right}, \Delta c, w_x, w_y, d_{obs}, d_{food}]^\top
$$

### Observation vector

Use a factored observation model with two observation groups:

1. **Body / environment observations**

$$
o_t^{body} = [x, y, \theta, w_x, w_y, d_{obs}, d_{food}, c_{left}^{body}, c_{right}^{body}]
$$

2. **Neural observations**

$$
o_t^{neural} = \text{recent spike buffer over the Level 1 readout neuron(s)}
$$

The full likelihood is then:

$$
p(o_t \mid s_t) = p(o_t^{body} \mid s_t) \, p(o_t^{neural} \mid s_t)
$$

This keeps the PP-GLM bridge as the odor-related neural likelihood term while
the body sensors provide the remaining state constraints.

---

## Step 1: Implement an Explicit Likelihood Model $P(o \mid s)$

## Goal

Replace the current split correction logic with an explicit observation model
whose log-likelihood can be evaluated and debugged directly.

## Methodology

Define two explicit likelihood functions.

### 1A. Body likelihood

Use a diagonal Gaussian observation model for measurements that come directly
from the plant or sensor layer:

$$
\log p(o_t^{body} \mid s_t) = -\frac{1}{2}\sum_j \left[
\frac{(o_{t,j}^{body} - h_j(s_t))^2}{\sigma_{obs,j}^2}
+ \log(2\pi \sigma_{obs,j}^2)
\right]
$$

where $h_j(s_t)$ is the deterministic observation map. For the first pass,
$h_j(s_t)$ is mostly identity on observable state coordinates.

### 1B. Neural likelihood

Keep the existing PP-GLM as the neural observation model. The current
`infer_odor_posterior(...)` function already approximates the spike likelihood
over the odor subspace by evaluating candidate odor states against the spike window.

For full active inference, expose a lower-level scoring function such as:

```python
def neural_log_likelihood(
    spike_window: np.ndarray,
    beta: np.ndarray,
    spike_history_window: np.ndarray,
    heading_window: np.ndarray,
    wind_angle_window: np.ndarray,
    candidate_odor_state: np.ndarray,
) -> float:
    ...
```

Then use it inside the Gaussian belief update or local odor-subspace update.

### 1C. Combined observation model

Add a lightweight container in a new file such as `level3_controller/generative_model.py`:

```python
@dataclass
class ObservationModel:
    body_obs_sigma: np.ndarray

    def body_log_likelihood(self, obs: Mapping[str, float], state_mu: np.ndarray) -> float:
        ...

    def neural_log_likelihood(self, ... ) -> float:
        ...
```

## Biological rationale

- body observations reflect the whole-animal loop through the plant and environment
- the PP-GLM likelihood preserves the project's core bridge: spikes inform odor-related hidden state without exposing raw membrane variables upward

## Files to modify

- `level2_bridge/ppglm.py`
  - expose a reusable neural log-likelihood helper if needed
- `level3_controller/active_inference.py`
  - replace implicit correction-only logic with calls to explicit observation-model functions
- `level3_controller/generative_model.py`
  - new file for explicit observation and transition model objects

## Verification

- confirm that the combined log-likelihood increases when the hidden state is closer to the true observed state
- confirm that odor posterior shifts in the expected direction when spike rates rise or fall

---

## Step 2: Implement a Proper Prior Over Hidden State $P(s)$

## Goal

Separate three concepts that are currently partially mixed:

1. **initial prior** over hidden state at episode start
2. **predictive prior** over hidden state before assimilating new observations
3. **prior preferences** over desirable outcomes used in policy evaluation

## Methodology

### 2A. Initial prior

Initialize the belief with domain-informed means and variances, not all zeros.

For example:

- `x, y, theta`: from the starting pose in the arena
- `c_left, c_right, delta_c`: low concentration prior with broad variance if plume contact is uncertain
- `w_x, w_y`: prior centered on known plume wind settings
- `d_obs, d_food`: inferred from initial arena geometry or coarse sensor readings

### 2B. Predictive prior

At each control cycle, compute:

$$
p(s_t \mid o_{1:t-1}, a_{1:t-1})
$$

by propagating the previous posterior through the transition model.

This predictive prior becomes the input to the observation update.

### 2C. Prior preferences

Keep preferred outcomes separate from the filtering prior. These correspond to
task-level preferences such as:

- low `d_food`
- high odor contact near the target
- safe obstacle distance
- stable feeder stop once the food is reached

These should live in the policy / EFE layer, not inside the filtering prior.

## Recommended data structures

```python
@dataclass
class StateBelief:
    mu: np.ndarray
    sigma: np.ndarray

@dataclass
class PreferredOutcomeModel:
    preferred_mu: np.ndarray
    preferred_weight: np.ndarray
```

## Biological rationale

This separation matters because the fly's current belief about the world is not
the same thing as the controller's preferred future outcomes. Mixing them makes
policy inference unstable and obscures whether behavior is driven by evidence or preference.

## Files to modify

- `level3_controller/active_inference.py`
  - initialize meaningful priors
  - separate filtering prior from preference prior
- `level3_controller/policy.py`
  - move task preferences into a clearly named prior-preference structure

## Verification

- confirm that priors move forward across time even before an observation update
- confirm that preferred outcomes do not overwrite the hidden-state estimate directly

---

## Step 3: Implement the Transition Model $P(s' \mid s, a)$

## Goal

Make the hidden-state prediction step genuinely action-conditioned rather than
just inflating uncertainty.

## Methodology

Define a transition function:

$$
s_{t+1} = f(s_t, a_t) + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, Q)
$$

For the first implementation, use the existing walking kinematics as the deterministic part:

- forward speed updates `x, y`
- yaw rate updates `theta`
- sidestep updates lateral position
- odor and wind states evolve from the environment observations and short-horizon persistence assumptions
- `d_food` and `d_obs` update from geometry or predicted motion

### 3A. Reuse the current plant-side logic

The current kinematic fallback in `run_closed_loop.py` already contains a basic
action-to-motion map. The controller should reuse the same kinematic assumptions
for prediction so that the belief model and plant are consistent.

### 3B. Predict both mean and uncertainty

Use:

$$
\mu_{t|t-1} = f(\mu_{t-1|t-1}, a_{t-1})
$$

$$
\Sigma_{t|t-1} = \Sigma_{t-1|t-1} + Q
$$

For the first pass, a diagonal covariance update is acceptable.

### 3C. Put transition logic inside the controller

Do not leave the action-conditioned prediction only in `policy.py` or only in
the plant step. The controller's `_predict(...)` step must use the last action
or last selected policy.

## Recommended interface

```python
@dataclass
class TransitionModel:
    process_noise: np.ndarray

    def predict(self, belief: StateBelief, action: Mapping[str, float]) -> StateBelief:
        ...
```

## Biological rationale

The transition model is the controller's internal expectation of how actions
change sensory and task-relevant state. Without it, the system is doing reactive
state correction, not active inference.

## Files to modify

- `level3_controller/active_inference.py`
  - change `_predict()` into an action-conditioned prediction step
- `run_closed_loop.py`
  - feed the last applied motor command back into the controller
- `level3_controller/generative_model.py`
  - optional new home for the transition model object

## Verification

- confirm that predicted `x, y, theta` change before correction when the agent moves
- confirm that predicted `d_food` decreases during forward motion toward the feeder
- confirm that posterior uncertainty grows under prediction-only rollouts

---

## Step 4: Implement a Policy Prior and Policy Posterior

## Goal

Turn the existing mode set into a real inferred policy distribution.

## Methodology

### 4A. Policy prior $P(\pi)$

Define an explicit prior over the four policies:

$$
P(\pi) = P(\texttt{SURGE}), P(\texttt{CAST}), P(\texttt{AVOID}), P(\texttt{STOP})
$$

The first implementation can use:

- a uniform prior when no behavioral context is strongly favored
- optional habit persistence by giving extra mass to the previously selected mode
- optional safety bias by increasing prior mass on `AVOID` near obstacles

This prior should be explicit and inspectable, not implicit inside hard-coded thresholds.

### 4B. Expected free energy $G(\pi)$

For each policy, compute a one-step or short-horizon approximation of:

$$
G(\pi) = \text{pragmatic cost}(\pi) - \text{epistemic value}(\pi)
$$

The current `policy.py` already contains a good first-pass decomposition:

- pragmatic term: closeness to food, safety from obstacles, odor retention near target
- epistemic term: uncertainty reduction for odor-related variables

The difference is that these scores should feed a posterior over policies rather than a direct argmin.

### 4C. Policy posterior $q(\pi)$

Compute:

$$
q(\pi) \propto P(\pi) \exp[-\gamma G(\pi)]
$$

where $\gamma$ is a precision or inverse-temperature parameter.

In log-space:

$$
\log q(\pi) = \log P(\pi) - \gamma G(\pi) - \log Z
$$

This is where `softmax` and `logsumexp` are useful.

### 4D. Use the policy posterior for action selection

Replace the current direct `min(G)` selector with one of:

- `argmax q(pi)` for deterministic control
- sampling from `q(pi)` for exploratory control

Store the posterior in `TaskState.probs` and make it the single source of truth for policy selection.

## Recommended interface

```python
@dataclass
class PolicyPosterior:
    log_prior: np.ndarray
    g_value: np.ndarray
    probs: np.ndarray

def infer_policy_posterior(
    controller: ActiveInferenceController,
    precision: float = 1.0,
) -> PolicyPosterior:
    ...
```

## Biological rationale

This keeps the controller in the correct active inference form: it does not
choose actions only by thresholding state variables, but by inferring which
policy best trades off preference satisfaction and uncertainty reduction.

## Files to modify

- `level3_controller/policy.py`
  - add explicit policy prior
  - return posterior over policies, not only a chosen mode
- `level3_controller/active_inference.py`
  - let `TaskState.probs` store the inferred policy posterior
- `run_closed_loop.py`
  - choose commands from the inferred posterior

## Verification

- confirm policy probabilities sum to 1
- confirm `CAST` gains posterior mass when odor uncertainty rises
- confirm `AVOID` gains posterior mass as obstacle distance shrinks
- confirm `STOP` dominates when `d_food` is very small and odor remains high

---

## Step 5: Integrate the Full Active Inference Update Cycle

The per-cycle controller update should become:

```python
# 1. Predict hidden state from previous posterior and previous action
belief_prior = transition_model.predict(prev_belief, last_action)

# 2. Evaluate observation likelihoods
body_logp = observation_model.body_log_likelihood(body_obs, belief_prior.mu)
odor_post = infer_odor_posterior(..., prior_mean=belief_prior.mu[odor_idx], prior_sigma=belief_prior.sigma[odor_idx])

# 3. Form posterior over hidden state
belief_post = fuse_body_and_neural_evidence(belief_prior, body_obs, odor_post)

# 4. Evaluate policies
policy_post = infer_policy_posterior(controller, precision=gamma)

# 5. Select or sample policy
mode = BehavioralMode(int(np.argmax(policy_post.probs)))

# 6. Map policy to motor command
command = mode_to_motor_command(mode, controller)
```

This update can still run on the current cadence:

- Level 1 spikes: 1 ms
- PP-GLM neural likelihood: 10 ms
- full active inference state and policy update: 20 ms

---

## Step 6: Recommended File Layout

### Minimal-change option

- keep `level3_controller/active_inference.py` as the orchestrator
- keep `level3_controller/policy.py` for EFE and policy inference
- extend `level2_bridge/ppglm.py` only where neural likelihood needs to be exposed more directly

### Cleaner option

Add:

- `level3_controller/generative_model.py`
  - `StateBelief`
  - `ObservationModel`
  - `TransitionModel`
  - `PreferredOutcomeModel`
  - `PolicyPosterior`

This keeps `active_inference.py` focused on orchestration rather than storing every mathematical detail.

---

## Step 7: Validation Checklist

The implementation should not be considered complete until all four checks pass.

### Likelihood

- PP-GLM neural likelihood improves held-out spike prediction relative to an intercept-only baseline
- body observation likelihood is maximized near the true hidden state

### Prior / filtering

- predictive priors move correctly under action even before correction
- posterior variance decreases after informative observations

### Transition model

- action-conditioned prediction matches the plant-side kinematic update closely enough for short horizons
- predicted state degrades gracefully under missing observations

### Policy posterior

- `TaskState.probs` is the actual policy posterior
- actions are selected from that posterior
- policy mass shifts appropriately with uncertainty, obstacle risk, and feeder proximity

---

## Step 8: Order of Implementation

Implement in this order to minimize breakage:

1. formalize the observation model
2. formalize the transition model inside the controller
3. separate filtering prior from preferred outcomes
4. implement explicit policy prior and posterior
5. switch action selection to `q(pi)`
6. add logging for priors, posteriors, and EFE terms

This order preserves the current PP-GLM bridge and closed-loop runner while upgrading the controller incrementally.

---

## Summary

The methodology is to upgrade the current controller into a **diagonal-Gaussian,
discrete-policy active inference controller** rather than to replace the existing
architecture. The PP-GLM remains the neural likelihood bridge from Level 1 to
Level 2. The whole-body plant remains the source of body observations and the
recipient of motor commands. The key missing work is not a new neural model; it
is making the controller's generative model explicit:

- explicit observation likelihood
- explicit action-conditioned transition prior
- clear separation between hidden-state prior and prior preferences
- explicit posterior over policies

That is the shortest path to a true active inference implementation consistent
with the broader connectome-based, conductance-based, whole-animal roadmap.