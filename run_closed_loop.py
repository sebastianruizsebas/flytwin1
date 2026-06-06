"""
Closed-Loop Simulation Entry Point (Phase 4)
Master 1 ms clock; subsystems trigger on sub-sample counters.

Execution order per iteration (Phase 4b):
  1. Advance OdorPlume state (1 ms).
  2. Read bilateral antennal odor sensors; compute u_sens.
  3. Inject u_sens into HH neuron(s); advance 1 ms; collect spikes.
    4. Every 10 ms: infer a spike-conditioned odor posterior with the PP-GLM bridge.
  5. Every 20 ms: update Level 2 Gaussian belief and Level 3 mode probabilities.
  6. Evaluate EFE for SURGE, CAST, AVOID, STOP; select mode.
  7. Map mode to walking motor command; apply (or log for flybody).
  8. Log state to HDF5 if log_path is provided.

Note on MuJoCo / flybody integration:
  Full flybody integration requires the flybody package and an mjx/dm_control
  environment built from arena.xml.  In this first-pass implementation the
  physical plant step is stubbed with a simple kinematic update so the
  closed-loop logic can be exercised without the full flybody install.
  Replace _kinematic_plant_step() with the flybody step when available.
"""
from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

from environment_sim.odor_plume import OdorPlume
from level1_biophysics.hh_neuron import HHNeuron
from level1_biophysics.connectome_rnn import ConnectomeRNN
from level2_bridge.design_matrix import MAX_HIST_LAG_MS
from level2_bridge.ppglm import OdorPosterior, infer_odor_posterior
from level2_bridge.motor_readout import MotorReadout
from level2_bridge.sbi_trainer import NPEInferenceEngine
from level3_controller.active_inference import (
    ActiveInferenceController, IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C,
)
from level3_controller.policy import select_action, mode_to_motor_command


DT_MS       = 1.0    # master clock step, ms
DT_PHYSICS_MS = 2.0  # intended MuJoCo physics step (used for kinematic fallback)
DT_GLM_MS   = 10.0   # PP-GLM / bridge evaluation interval, ms
DT_CTRL_MS  = 20.0   # EFE / belief update interval, ms


# ── Antenna offsets from body centre (metres) ───────────────────────────────
_ANT_OFFSET_Y = 0.003   # 3 mm bilateral separation


def _kinematic_plant_step(state: dict, cmd: dict, dt_s: float) -> dict:
    """
    Minimal kinematic placeholder for the full flybody plant.
    Updates (x, y, theta) from motor commands; leaves obstacle/food
    distances unchanged (they would come from MuJoCo contact/proximity sensors).

    state keys: x, y, theta, d_obs, d_food, w_x, w_y, c_left, c_right
    cmd keys: forward_speed, yaw_rate, sidestep, active

    Walking speed scale: forward_speed=1 → ~0.01 m/s (typical Drosophila walking).
    """
    if not cmd.get("active", True):
        return state  # STOP — no movement

    WALK_SCALE = 0.01  # m/s per unit forward_speed
    theta = state["theta"]
    dx = cmd.get("forward_speed", 0.0) * WALK_SCALE * np.cos(theta) * dt_s
    dy = cmd.get("forward_speed", 0.0) * WALK_SCALE * np.sin(theta) * dt_s
    dy += cmd.get("sidestep", 0.0) * WALK_SCALE * 0.5 * dt_s
    dtheta = cmd.get("yaw_rate", 0.0) * dt_s

    new_state = dict(state)
    new_state["x"]     = state["x"]     + dx
    new_state["y"]     = state["y"]     + dy
    new_state["theta"] = state["theta"] + dtheta
    # Food distance decreases as fly moves toward origin (food at [0,0])
    new_state["d_food"] = float(np.hypot(new_state["x"], new_state["y"]))
    # Obstacle distance: placeholder — not updated without MuJoCo geometry
    return new_state


def run(
    duration_ms: float = 60_000.0,
    arena_xml: str = "environment_sim/arena.xml",
    beta_path: str | None = None,
    h5_path: str | None = None,
    log_path: str | None = None,
    swc_path: str = "",
    connectome_dir: str | None = None,
    motor_readout: MotorReadout | None = None,
    npe_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Main simulation loop.

    Parameters
    ----------
    duration_ms : float
        Total simulation duration in ms.
    arena_xml : str
        Path to MuJoCo scene XML (used when flybody is available).
    beta_path : str or None
        Path to a numpy .npy file containing a pre-fitted (24,) beta vector.
        If None, a zero vector is used (produces flat PP-GLM likelihood).
    h5_path : str or None
        Path to training_data.h5.  When provided the Level 3 pymdp A matrices
        are calibrated from that file via empirical Bayes, making the agent's
        sensory-mode predictions data-grounded rather than hand-coded.
    log_path : str or None
        Path to HDF5 log file.  If None, logging is skipped.
    swc_path : str
        Path to a neuron morphology SWC file exported by import_connectome.py
        (e.g. data/connectome/skeletons/<bodyId>.swc).  Required: HHNeuron
        will raise FileNotFoundError if the file is absent.
    connectome_dir : str or None
        Path to data/connectome/ directory.  When provided, a ConnectomeRNN
        (Brian2 LIF population with connectome weights) is used instead of a
        single HHNeuron.  The full 165k-neuron network is instantiated; use
        --max-synapses for a faster smoke test (e.g. 500000).
    verbose : bool
        Print progress every 1000 ms.
    npe_path : str or None
        Path to a pickled NPEInferenceEngine posterior (data/npe_posterior.pkl).
        When provided, the engine is loaded at startup and called every 20 ms
        to infer biophysical conductances [g_KA, g_Na, g_CaL] from the current
        beta vector, updating the HHNeuron live.  Requires prior offline
        training via ``python -c "from level2_bridge.sbi_trainer import
        train_npe_from_data; train_npe_from_data()"``.

    motor_readout : MotorReadout or None
        Optional motor readout instance built from imported connectome assets
        (see level2_bridge.motor_readout.load_motor_readout).  When provided,
        motor commands are derived from motoneuron pool activations rather than
        the heuristic mode_to_motor_command() path.  Requires spike_window to
        be a (W, N) population array; the single-neuron fallback is used when
        motor_readout is None.

    Returns
    -------
    dict with keys 'positions', 'modes', 'log_liks' — numpy arrays of logged
    quantities for post-hoc analysis.
    """
    # ── Initialise subsystems ──────────────────────────────────────────────
    plume = OdorPlume(
        wind_mean=np.array([-0.2, 0.0, 0.0]),  # blows toward fly at negative x
        wind_noise_std=0.04,
        puff_rate=15.0,
        puff_sigma=0.05,
        source_position=np.zeros(3),
    )

    # ── Optional SBI amortized inference engine ─────────────────────────
    npe_engine: NPEInferenceEngine | None = None
    if npe_path is not None:
        try:
            npe_engine = NPEInferenceEngine(npe_path)
            npe_engine.load()
            if verbose:
                print(f"NPE posterior loaded from {npe_path}")
        except FileNotFoundError as exc:
            import warnings
            warnings.warn(str(exc))
            npe_engine = None

    # ── Initialise Level 1 neuron / population ────────────────────────────
    use_population = connectome_dir is not None
    if use_population:
        print(f"Using ConnectomeRNN (Brian2 LIF) from {connectome_dir}")
        population = ConnectomeRNN(
            connectome_dir=connectome_dir,
            input_body_ids=None,   # broadcast sensory drive to all neurons
        )
        population.build()         # ~2-4 min for full 25M-synapse network
        neuron = None
    else:
        # Single HH neuron representing a readout projection neuron
        neuron = HHNeuron(swc_path=swc_path, conductances={"g_KA": 1.0})
        neuron.build()
        population = None

    # Simulated body state (kinematic fallback)
    body_state: dict = {
        "x": -0.5, "y": 0.0, "theta": 0.0,
        "d_obs": 1.0, "d_food": 0.5,
        "w_x": 0.2, "w_y": 0.0,
        "c_left": 0.0, "c_right": 0.0,
    }

    # Seed the filtering prior p(s_0) from the known start pose and wind rather
    # than from an uninformative zero vector. This keeps the controller's prior
    # belief distinct from the policy-level preferred outcomes used later.
    controller = ActiveInferenceController(
        initial_obs=body_state,
        h5_path=h5_path,
    )

    # Load pre-fitted PP-GLM beta (24-dim); fall back to zeros
    if beta_path is not None:
        beta = np.load(beta_path)
    else:
        beta = np.zeros(24)

    # Running histories for the spike bridge construction
    s_hist: List[int]   = [0]   * MAX_HIST_LAG_MS
    spike_buffer_window: List[float] = []  # accumulates over DT_GLM_MS window
    heading_buffer_window: List[float] = []
    wind_angle_buffer_window: List[float] = []
    spike_history_buffer_window: List[np.ndarray] = []

    # Counters and bookkeeping
    t_ms        = 0.0
    glm_counter = 0.0
    ctrl_counter = 0.0
    last_log_lik = 0.0
    last_spike_posterior: OdorPosterior | None = None
    last_mode    = None
    last_cmd: dict = {"forward_speed": 0.0, "yaw_rate": 0.0, "sidestep": 0.0, "active": True}

    # Pending NPE inference future (submitted in parallel with population.step())
    _npe_future: Optional[Future] = None
    _pending_beta_for_npe: Optional[np.ndarray] = None

    # Thread pool: 1 worker suffices — NPE inference and Brian2 run() both release
    # the GIL, so they genuinely overlap on separate OS threads.
    _executor = ThreadPoolExecutor(max_workers=1)

    # Logs
    log_positions: List[np.ndarray] = []
    log_modes:     List[int]        = []
    log_log_liks:  List[float]      = []

    # Optional HDF5 logging
    h5_file = None
    if log_path is not None:
        try:
            import h5py
            h5_file = h5py.File(log_path, "w")
        except ImportError:
            print("h5py not available — skipping HDF5 logging")

    if verbose:
        print(f"Starting closed-loop simulation: {duration_ms:.0f} ms")

    t_start = time.time()

    while t_ms < duration_ms:
        dt_s = DT_MS * 1e-3

        # ── 1. Advance plume ──────────────────────────────────────────────
        plume.step(dt_s)

        # ── 2. Read antennal odor sensors ─────────────────────────────────
        theta = body_state["theta"]
        body_xy = np.array([body_state["x"], body_state["y"], 0.0])
        perp = np.array([-np.sin(theta), np.cos(theta), 0.0])
        ant_left  = body_xy + perp * _ANT_OFFSET_Y
        ant_right = body_xy - perp * _ANT_OFFSET_Y

        u_sens_dict = plume.get_antennal_obs(ant_left, ant_right)
        c_left  = u_sens_dict["c_left"]
        c_right = u_sens_dict["c_right"]
        wind_vec = u_sens_dict["wind_vector"]
        # Clamp input current to HH rheobase range to prevent numerical overflow
        # at high odor concentration (sigmoid max = 1.0 → 15 µA/cm²; safe up to ~10)
        u_val = float(np.clip(0.5 * (c_left + c_right) * 10.0, 0.0, 9.5))

        # Wind angle relative to heading
        wind_angle = float(np.arctan2(wind_vec[1], wind_vec[0]) - theta)

        # ── 3. Inject into Level 1, collect spike(s) ─────────────────────
        # When running the connectome population, submit any pending NPE inference
        # to the thread pool NOW so it runs concurrently with Brian2's network.run().
        # Brian2 Cython/C++ code and PyTorch both release the GIL during their
        # compute-intensive inner loops, so the two genuinely overlap on CPU.
        if use_population and npe_engine is not None and _pending_beta_for_npe is not None:
            _beta_snap = _pending_beta_for_npe.copy()
            _pending_beta_for_npe = None
            _npe_future = _executor.submit(npe_engine.infer, _beta_snap, 50)

        if use_population:
            # Population path: ConnectomeRNN → (N,) spikes (GIL released here)
            pop_spikes_t = population.step(I_ext=u_val, dt_ms=DT_MS)  # (N,)
            # Single representative spike for PP-GLM bridge: mean of all neurons
            spiked = int(pop_spikes_t.mean() > 0.5 / population.N)
        else:
            # Single HH neuron path (also submits NPE inline since no concurrency benefit)
            if npe_engine is not None and _pending_beta_for_npe is not None:
                _beta_snap = _pending_beta_for_npe.copy()
                _pending_beta_for_npe = None
                _npe_future = _executor.submit(npe_engine.infer, _beta_snap, 50)
            spiked = neuron.step(current_uA=u_val, dt_ms=DT_MS)

        # Rolling spike history snapshot (before updating s_hist)
        spike_history_snapshot = np.array(s_hist, dtype=float)
        s_hist.pop(0); s_hist.append(int(spiked))

        # Accumulate bridge buffers
        spike_buffer_window.append(float(spiked))
        heading_buffer_window.append(theta)
        wind_angle_buffer_window.append(wind_angle)
        spike_history_buffer_window.append(spike_history_snapshot)

        # Update body state with updated odor
        body_state["c_left"]  = c_left
        body_state["c_right"] = c_right
        body_state["w_x"]     = float(wind_vec[0])
        body_state["w_y"]     = float(wind_vec[1])

        glm_counter  += DT_MS
        ctrl_counter += DT_MS

        # ── 4. Spike-to-odor bridge every 10 ms ──────────────────────────
        if glm_counter >= DT_GLM_MS:
            glm_counter = 0.0
            if len(spike_buffer_window) > 0:
                spk_arr = np.array(spike_buffer_window)
                last_spike_posterior = infer_odor_posterior(
                    spike_window=spk_arr,
                    beta=beta,
                    spike_history_window=np.stack(spike_history_buffer_window),
                    heading_window=np.array(heading_buffer_window, dtype=float),
                    wind_angle_window=np.array(wind_angle_buffer_window, dtype=float),
                    prior_mean=controller.body_state.mu[[IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C]],
                    prior_sigma=controller.body_state.sigma[[IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C]],
                )
                last_log_lik = last_spike_posterior.log_evidence
            spike_buffer_window.clear()
            heading_buffer_window.clear()
            wind_angle_buffer_window.clear()
            spike_history_buffer_window.clear()

        # ── 5–7. Belief update + EFE + motor command every 20 ms ─────────
        if ctrl_counter >= DT_CTRL_MS:
            ctrl_counter = 0.0

            # Feed the previously applied command back into the controller as an
            # efference-copy style action input. The predictive prior should be
            # conditioned on what the body just did, not only on the last state.
            controller.set_last_motor(last_cmd)
            controller.update_beliefs(
                mujoco_obs=body_state,
                spike_posterior=last_spike_posterior,
            )

            # ── SBI online conductance update ─────────────────────────────
            # Collect the previously submitted NPE future (it ran concurrently
            # with the last population.step()), then queue the next one for the
            # coming 20 ms window.
            if npe_engine is not None:
                if _npe_future is not None:
                    try:
                        theta_samples = _npe_future.result(timeout=0.05)  # non-blocking collect
                        g_ka, g_na, g_cal = theta_samples.mean(axis=0)
                        if neuron is not None:
                            neuron.set_conductances({
                                "g_KA": float(g_ka),
                                "g_Na": float(g_na),
                                "g_CaL": float(g_cal),
                            })
                    except Exception:
                        pass
                    _npe_future = None
                # Queue the next inference to overlap with the next population.step()
                _pending_beta_for_npe = beta.copy()

            if use_population and motor_readout is not None:
                # Full population → motor pool path
                pop_spikes_arr = pop_spikes_t[:, np.newaxis] if pop_spikes_t.ndim == 1 else pop_spikes_t
                motor_state = motor_readout.step(pop_spikes_arr, DT_MS)
                last_cmd = motor_state.command
                last_mode = int(motor_state.mode)
            elif motor_readout is not None and len(spike_buffer_window) > 0:
                # Motor readout path: spikes → pool rates → z_t → command.
                # spike_buffer_window holds (W,) scalars from the single neuron
                # stub; replace with a (W, N) population array when Level 1 is
                # expanded to a full motor population.
                pop_spikes = np.array(spike_buffer_window, dtype=float)[:, None]
                motor_state = motor_readout.step(pop_spikes, DT_MS)
                last_cmd = motor_state.command
                last_mode = int(motor_state.mode)
            else:
                # Fallback: heuristic mode → command via policy.py
                mode = select_action(controller)
                last_cmd = mode_to_motor_command(mode, controller)
                last_mode = int(mode)

            # Apply kinematic fallback step (replace with flybody.step())
            prev_body = dict(body_state)
            body_state = _kinematic_plant_step(body_state, last_cmd, DT_CTRL_MS * 1e-3)
            controller.set_last_motor(last_cmd)

            # ── Error-driven B-matrix adaptation ─────────────────────────
            # Compute discrepancy between commanded and actual movement;
            # use to adapt the motor readout's locomotor basis online.
            if motor_readout is not None:
                fwd_cmd   = float(last_cmd.get("forward_speed", 0.0))
                yaw_cmd   = float(last_cmd.get("yaw_rate", 0.0)) / 5.0  # normalise
                side_cmd  = float(last_cmd.get("sidestep", 0.0))
                stop_cmd  = 0.0 if last_cmd.get("active", True) else 1.0
                dx_actual = body_state["x"] - prev_body["x"]
                dyaw_actual = body_state["theta"] - prev_body["theta"]
                fwd_actual  = dx_actual / max(DT_CTRL_MS * 1e-3, 1e-9) / 0.01
                position_error = np.array([
                    fwd_cmd  - fwd_actual,
                    yaw_cmd  - dyaw_actual,
                    side_cmd - 0.0,        # no sidestep sensor yet
                    stop_cmd - 0.0,
                ], dtype=float)
                motor_readout.adapt_from_error(position_error)

            log_positions.append(np.array([body_state["x"], body_state["y"], body_state["theta"]]))
            log_modes.append(last_mode)
            log_log_liks.append(last_log_lik)

            # Check stop condition
            inferred_c_avg = 0.5 * (
                controller.body_state.c_left + controller.body_state.c_right
            )
            if body_state["d_food"] < 0.02 and inferred_c_avg > 0.3:
                if verbose:
                    print(f"  ** Food reached at t={t_ms:.0f} ms, d_food={body_state['d_food']:.3f} m")
                break

        t_ms += DT_MS

        if verbose and t_ms % 1000 < DT_MS:
            elapsed = time.time() - t_start
            mode_name = BehavioralModeStr(last_mode) if last_mode is not None else "—"
            print(f"  t={t_ms:.0f} ms | mode={mode_name} | "
                  f"d_food={body_state['d_food']:.3f} | log_lik={last_log_lik:.2f} "
                  f"[{elapsed:.1f}s elapsed]")

    if verbose:
        print("Simulation complete.")

    # Drain any in-flight NPE future before closing
    if _npe_future is not None:
        try:
            _npe_future.result(timeout=1.0)
        except Exception:
            pass
    _executor.shutdown(wait=False)

    if h5_file is not None:
        if log_positions:
            h5_file.create_dataset("positions",  data=np.array(log_positions))
            h5_file.create_dataset("modes",      data=np.array(log_modes))
            h5_file.create_dataset("log_liks",   data=np.array(log_log_liks))
        h5_file.close()

    return {
        "positions": np.array(log_positions) if log_positions else np.empty((0, 3)),
        "modes":     np.array(log_modes,    dtype=int),
        "log_liks":  np.array(log_log_liks, dtype=float),
    }


def BehavioralModeStr(mode_int: int | None) -> str:
    names = {0: "SURGE", 1: "CAST", 2: "AVOID", 3: "STOP"}
    return names.get(mode_int, "?")


def save_trajectory_video(
    log_path: str,
    video_path: str,
    fps: int = 30,
    trail_ms: int = 2000,
) -> None:
    """
    Render a top-down 2D animation of the agent trajectory from a run log.

    Uses only matplotlib + mediapy -- no MuJoCo required.
    Each frame shows:
      - Full past trajectory (faded grey trail)
      - Last `trail_ms` ms coloured by behavioral mode
      - Current heading arrow
      - Odor source / food marker and obstacle cylinders
      - Wind direction arrow, arena walls
      - Mode label and time readout

    Parameters
    ----------
    log_path    : path to HDF5 log written by run_closed_loop.run()
    video_path  : output mp4 / gif path
    fps         : playback frame rate; lower values show each step longer
                  (default 5 = one control-step visible for 200 ms wall-clock)
    trail_ms    : ms of trajectory to colour by mode in each frame
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import mediapy

    MODE_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336", 3: "#4CAF50"}
    MODE_NAMES  = {0: "SURGE",   1: "CAST",    2: "AVOID",   3: "STOP"}

    # Environment geometry (must match OdorPlume source_position and arena.xml)
    ODOR_SOURCE_XY  = (0.0, 0.0)           # source_position=zeros in run()
    ARENA_HALF_X    = 1.0                   # wall_upwind/wall_downwind at ±1 m
    ARENA_HALF_Y    = 1.0
    WIND_DIR_XY     = (-0.2, 0.0)          # wind_mean from run()
    # Obstacle positions from arena.xml (visual only; d_obs not yet live)
    OBSTACLES_XY    = [(-0.25, 0.15), (-0.25, -0.15)]
    OBS_RADIUS      = 0.02

    with h5py.File(log_path, "r") as f:
        pos   = f["positions"][:]   # (T, 3): x, y, theta
        modes = f["modes"][:]       # (T,) int
        liks  = f["log_liks"][:]

    T = len(pos)
    if T == 0:
        print("No logged positions found — nothing to render.")
        return

    # Subsample: one frame every 20 logged steps (400 ms) for manageable video
    frame_stride = max(1, int(20))
    frame_indices = list(range(0, T, frame_stride))

    x, y, theta = pos[:, 0], pos[:, 1], pos[:, 2]
    trail_steps = trail_ms // 20  # 20 ms per control step

    # Expand axes to always include the odor source and arena bounds
    x_min = min(x.min() - 0.05, -ARENA_HALF_X * 0.1)
    x_max = max(x.max() + 0.05,  ODOR_SOURCE_XY[0] + 0.1)
    y_min = min(y.min() - 0.05,  ODOR_SOURCE_XY[1] - 0.1)
    y_max = max(y.max() + 0.05,  ODOR_SOURCE_XY[1] + 0.1)

    frames = []
    for fi in frame_indices:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        # Arena walls
        for wx in [-ARENA_HALF_X, ARENA_HALF_X]:
            ax.axvline(wx, color="#556", lw=1.0, ls="--", alpha=0.6)
        for wy in [-ARENA_HALF_Y, ARENA_HALF_Y]:
            ax.axhline(wy, color="#556", lw=1.0, ls="--", alpha=0.6)

        # Obstacle cylinders (from arena.xml; d_obs not yet live)
        for ox, oy in OBSTACLES_XY:
            circ = plt.Circle((ox, oy), OBS_RADIUS, color="#B05020", alpha=0.7, zorder=3)
            ax.add_patch(circ)
            ax.text(ox, oy + OBS_RADIUS + 0.01, "obs",
                    color="#B05020", fontsize=5, ha="center", va="bottom")

        # Wind direction arrow (at top-left of arena)
        wscale = 0.08
        ax.annotate("",
            xy=(x_min + 0.12 + WIND_DIR_XY[0] * wscale,
                y_max - 0.08 + WIND_DIR_XY[1] * wscale),
            xytext=(x_min + 0.12, y_max - 0.08),
            arrowprops=dict(arrowstyle="->", color="#88CCFF", lw=1.5))
        ax.text(x_min + 0.12, y_max - 0.04, "wind",
                color="#88CCFF", fontsize=6, ha="center")

        # Full trail (grey)
        ax.plot(x[:fi+1], y[:fi+1], color="#444", lw=0.5, alpha=0.5)

        # Recent trail coloured by mode
        start = max(0, fi - trail_steps)
        for t in range(start, fi):
            col = MODE_COLORS.get(int(modes[t]), "white")
            ax.plot(x[t:t+2], y[t:t+2], color=col, lw=1.5, alpha=0.8)

        # Current position + heading arrow
        ax.scatter(x[fi], y[fi], color="white", s=40, zorder=5)
        arrow_len = 0.015
        ax.annotate("",
            xy=(x[fi] + arrow_len * np.cos(theta[fi]),
                y[fi] + arrow_len * np.sin(theta[fi])),
            xytext=(x[fi], y[fi]),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

        # Odor source / food target at origin
        ax.scatter(*ODOR_SOURCE_XY, marker="*", color="#FFD700", s=250,
                   zorder=6, label="odor source / food")
        ax.text(ODOR_SOURCE_XY[0], ODOR_SOURCE_XY[1] + 0.025,
                "source", color="#FFD700", fontsize=6, ha="center")

        # Mode legend
        patches = [mpatches.Patch(color=v, label=k)
                   for k, v in zip(MODE_NAMES.values(), MODE_COLORS.values())]
        ax.legend(handles=patches, loc="upper right",
                  fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # Info text
        t_ms_now = fi * 20
        d_src = np.hypot(x[fi] - ODOR_SOURCE_XY[0], y[fi] - ODOR_SOURCE_XY[1])
        ax.set_title(
            f"t = {t_ms_now:,} ms  |  mode = {MODE_NAMES.get(int(modes[fi]), '?')}  "
            f"|  d_source = {d_src*100:.1f} cm",
            color="white", fontsize=9
        )

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    print(f"Saving {len(frames)} frames to {video_path} at {fps} fps ...")
    mediapy.write_video(video_path, frames, fps=fps)
    print(f"Video saved: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital fly twin closed-loop sim")
    parser.add_argument("--duration", type=float, default=60_000.0,
                        help="Simulation duration in ms (default: 60000)")
    parser.add_argument("--swc",  type=str, default=None,
                        help="Path to SWC skeleton file (data/connectome/skeletons/<bodyId>.swc). "
                             "Required for single-neuron HH mode; not needed with --connectome.")
    parser.add_argument("--log",   type=str, default=None,
                        help="HDF5 log file path (optional)")
    parser.add_argument("--beta",  type=str, default=None,
                        help="Path to .npy PP-GLM beta vector (optional)")
    parser.add_argument("--h5",   type=str, default=None,
                        help="Path to training_data.h5 — calibrates pymdp A matrices "
                             "from data (recommended: data/spikes/training_data.h5)")
    parser.add_argument("--connectome", type=str, default=None,
                        help="Path to data/connectome/ — activates ConnectomeRNN "
                             "(Brian2 LIF population with connectome weights) instead "
                             "of a single HHNeuron.  Build time ~2-4 min for full network.")
    parser.add_argument("--save-video", type=str, default=None,
                        metavar="PATH",
                        help="Render a top-down 2D trajectory video from the --log file "
                             "and save to PATH (e.g. data/spikes/run_pop_001.mp4). "
                             "Requires --log. No MuJoCo needed.")
    parser.add_argument("--npe", type=str, default=None,
                        help="Path to pickled NPE posterior (data/npe_posterior.pkl). "
                             "Enables online SBI amortized conductance inference.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(
        duration_ms=args.duration,
        log_path=args.log,
        beta_path=args.beta,
        h5_path=args.h5,
        swc_path=args.swc if args.swc else "",
        connectome_dir=args.connectome,
        npe_path=args.npe,
        verbose=not args.quiet,
    )
    if args.save_video:
        if not args.log:
            print("--save-video requires --log to be set.")
        else:
            save_trajectory_video(args.log, args.save_video)
