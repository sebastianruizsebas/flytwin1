"""
Closed-Loop Simulation Entry Point (Phase 4)
Master 1 ms clock; subsystems trigger on sub-sample counters.

Execution order per iteration (Phase 4b):
  1. Advance OdorPlume state (1 ms).
  2. Read bilateral antennal odor sensors; compute u_sens.
  3. Inject u_sens into HH neuron(s); advance 1 ms; collect spikes.
  4. Every 10 ms: evaluate PP-GLM on spike buffer; get log-likelihood.
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

from environment_sim.odor_plume import OdorPlume
from level1_biophysics.hh_neuron import HHNeuron
from level2_bridge.design_matrix import build_design_row, MAX_STIM_LAG_MS, MAX_HIST_LAG_MS
from level2_bridge.ppglm import evaluate_online
from level3_controller.active_inference import (
    ActiveInferenceController, IDX_THETA, IDX_C_LEFT, IDX_C_RIGHT, IDX_DELTA_C,
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
    log_path: str | None = None,
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
    log_path : str or None
        Path to HDF5 log file.  If None, logging is skipped.
    verbose : bool
        Print progress every 1000 ms.

    Returns
    -------
    dict with keys 'positions', 'modes', 'log_liks' — numpy arrays of logged
    quantities for post-hoc analysis.
    """
    # ── Initialise subsystems ──────────────────────────────────────────────
    plume = OdorPlume(
        wind_mean=np.array([0.2, 0.0, 0.0]),
        wind_noise_std=0.04,
        puff_rate=15.0,
        source_position=np.zeros(3),
    )

    # Single HH neuron representing a readout projection neuron
    neuron = HHNeuron(conductances={"g_KA": 1.0})
    neuron.build()

    controller = ActiveInferenceController()

    # Load pre-fitted PP-GLM beta (24-dim); fall back to zeros
    if beta_path is not None:
        beta = np.load(beta_path)
    else:
        beta = np.zeros(24)

    # Simulated body state (kinematic fallback)
    body_state: dict = {
        "x": -0.5, "y": 0.0, "theta": 0.0,
        "d_obs": 1.0, "d_food": 0.5,
        "w_x": 0.2, "w_y": 0.0,
        "c_left": 0.0, "c_right": 0.0,
    }

    # Running histories for design matrix construction
    u_hist: List[float] = [0.0] * MAX_STIM_LAG_MS
    s_hist: List[int]   = [0]   * MAX_HIST_LAG_MS
    spike_buffer_window: List[float] = []  # accumulates over DT_GLM_MS window
    x_buffer_window: List[np.ndarray] = []  # design rows for the same window

    # Counters and bookkeeping
    t_ms        = 0.0
    glm_counter = 0.0
    ctrl_counter = 0.0
    last_log_lik = 0.0
    last_mode    = None
    last_cmd: dict = {"forward_speed": 0.0, "yaw_rate": 0.0, "sidestep": 0.0, "active": True}

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
        u_val = 0.5 * (c_left + c_right) * 3.0  # mean drive in µA/cm²

        # Wind angle relative to heading
        wind_angle = float(np.arctan2(wind_vec[1], wind_vec[0]) - theta)

        # ── 3. Inject into HH neuron, collect spike ───────────────────────
        spiked = neuron.step(current_uA=u_val, dt_ms=DT_MS)

        # Build design row for this time step
        x_row = build_design_row(
            u_sens_history=np.array(u_hist),
            spike_history=np.array(s_hist, dtype=float),
            heading=theta,
            c_left=c_left,
            c_right=c_right,
            wind_angle=wind_angle,
        )

        # Update rolling histories
        u_hist.pop(0); u_hist.append(u_val)
        s_hist.pop(0); s_hist.append(int(spiked))

        # Accumulate spike buffer and design rows for GLM window
        spike_buffer_window.append(float(spiked))
        x_buffer_window.append(x_row)

        # Update body state with updated odor
        body_state["c_left"]  = c_left
        body_state["c_right"] = c_right
        body_state["w_x"]     = float(wind_vec[0])
        body_state["w_y"]     = float(wind_vec[1])

        glm_counter  += DT_MS
        ctrl_counter += DT_MS

        # ── 4. PP-GLM evaluation every 10 ms ─────────────────────────────
        if glm_counter >= DT_GLM_MS:
            glm_counter = 0.0
            if len(spike_buffer_window) > 0:
                spk_arr = np.array(spike_buffer_window)
                x_arr   = np.vstack(x_buffer_window)
                last_log_lik = evaluate_online(spk_arr, beta, x_arr)
            spike_buffer_window.clear()
            x_buffer_window.clear()

        # ── 5–7. Belief update + EFE + motor command every 20 ms ─────────
        if ctrl_counter >= DT_CTRL_MS:
            ctrl_counter = 0.0

            odor_obs = np.array([c_left, c_right, c_left - c_right])
            controller.update_beliefs(
                mujoco_obs=body_state,
                ppglm_log_lik=last_log_lik,
                odor_obs=odor_obs,
            )

            mode = select_action(controller)
            last_cmd = mode_to_motor_command(mode, controller)
            last_mode = int(mode)

            # Apply kinematic fallback step (replace with flybody.step())
            body_state = _kinematic_plant_step(body_state, last_cmd, DT_CTRL_MS * 1e-3)

            log_positions.append(np.array([body_state["x"], body_state["y"], body_state["theta"]]))
            log_modes.append(last_mode)
            log_log_liks.append(last_log_lik)

            # Check stop condition
            if body_state["d_food"] < 0.02 and 0.5 * (c_left + c_right) > 0.3:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital fly twin closed-loop sim")
    parser.add_argument("--duration", type=float, default=60_000.0,
                        help="Simulation duration in ms (default: 60000)")
    parser.add_argument("--log",   type=str, default=None,
                        help="HDF5 log file path (optional)")
    parser.add_argument("--beta",  type=str, default=None,
                        help="Path to .npy PP-GLM beta vector (optional)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(
        duration_ms=args.duration,
        log_path=args.log,
        beta_path=args.beta,
        verbose=not args.quiet,
    )
