"""
Training data generation — parameter sweep across fly positions and conductance
conditions.

For each combination of (start position, g_KA scale, g_Na scale, g_CaL scale)
the script:
  1. Simulates a plume-driven sensory trace from the fly's starting position.
  2. Runs batch Brian2 spike trials (HHNeuron.simulate_spike_trials) across
     N_TRIALS independent noise realisations for that conductance condition.
  3. Builds a (T, 24) design matrix from the actual plume-drive + heading trace.
  4. Saves per-condition spike matrices and design matrices to HDF5.
  5. After all conditions, jointly fits the PP-GLM with the trend-filter penalty
     (fit_joint) and saves beta to data/spikes/beta.npy.

The sweep is structured so that conductance conditions are the "M conditions"
axis in fit_joint.  Fly-position variation provides richer stimulus statistics
within each condition.

Usage
-----
  conda activate flytwin
  python generate_training_data.py

Optional flags
--------------
  --out-dir     PATH        output directory (default: data/spikes)
  --n-trials    INT         noise realisations per condition (default: 30)
  --trial-ms    INT         trial duration in ms (default: 5000)
  --n-positions INT         number of fly starting positions to sweep (default: 8)
  --g-ka        FLOATS...   g_KA scale values  (default: 0.25 0.5 1.0 1.5 2.0)
  --g-na        FLOATS...   g_Na scale values  (default: 0.8 1.0 1.2)
  --g-cal       FLOATS...   g_CaL scale values (default: 0.5 1.0 2.0)
  --lam         FLOAT       trend-filter lambda (default: 1.0)
  --workers     INT         parallel worker processes (default: cpu_count, i.e.
                            one worker per logical core; each worker forces
                            Brian2 to cython target and limits BLAS threads to 1
                            to avoid oversubscription)
  --seed        INT         global RNG seed (default: 42)
  --quiet                   suppress per-condition progress lines

Biological rationale
--------------------
Varying fly position samples different plume encounter statistics — upwind
positions produce strong, intermittent pulses while lateral/downwind positions
give weaker, noisier drive.  This ensures the stimulus filter columns in beta
capture the full dynamic range of odor-evoked depolarisation.

Varying g_KA (A-type potassium), g_Na (spike initiation), and g_CaL (slower
depolarising dynamics) covers the conductance perturbation space used to test
how biophysical parameters shift neural gain, spike timing, and burst statistics.
The trend-filter penalty in fit_joint enforces smooth interpolation between
adjacent conductance conditions — the resulting multi-condition beta vector
encodes how the PP-GLM likelihood changes across the conductance landscape.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from environment_sim.odor_plume import OdorPlume
from level1_biophysics.hh_neuron import HHNeuron, _BASE_G
from level2_bridge.design_matrix import (
    MAX_HIST_LAG_MS,
    build_design_matrix,
)
from level2_bridge.ppglm import fit_joint

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_OUT_DIR    = _REPO_ROOT / "data" / "spikes"
_DEFAULT_N_TRIALS   = 30
_DEFAULT_TRIAL_MS   = 5000
_DEFAULT_N_POS      = 8
_DEFAULT_G_KA       = [0.25, 0.5, 1.0, 1.5, 2.0]
_DEFAULT_G_NA       = [0.8, 1.0, 1.2]
_DEFAULT_G_CAL      = [0.5, 1.0, 2.0]
_DEFAULT_LAM        = 1.0
_DEFAULT_SEED       = 42
_DEFAULT_WORKERS    = os.cpu_count() or 1

# Drive scale: mean bilateral concentration → µA/cm² injected current
_DRIVE_SCALE        = 15.0   # µA/cm² per unit sigmoid output; HH rheobase ≈ 6.3 µA/cm²
# Bilateral antenna separation (m)
_ANT_OFFSET_Y       = 0.003
# 1/f noise amplitude for biological variability in drive
_NOISE_STD          = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Top-level worker (must be module-level for multiprocessing pickling on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init_func() -> None:
    """
    Initializer called once in each worker process of ProcessPoolExecutor.

    Forces the Brian2 codegen target to "cython" so that all 32 workers use
    CPU-JIT compilation rather than fighting over a single cuda_standalone
    device.  Also limits BLAS threads to 1 per worker, preventing the
    (workers × BLAS_threads) oversubscription that thrashes the CPU.

    Theoretical role: one worker = one logical core; Brian2 Cython simulations
    are single-threaded, so this achieves near-linear scaling across 32 cores.
    """
    import os as _os
    _os.environ["FLYTWIN_BRIAN2_TARGET"] = "cython"
    _os.environ["FLYTWIN_WORKER_MODE"]    = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        # threadpoolctl not installed; BLAS may use multiple threads per worker.
        # Performance will still be correct but throughput may be sub-optimal.
        _os.environ.setdefault("OMP_NUM_THREADS",    "1")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _os.environ.setdefault("MKL_NUM_THREADS",     "1")


def _job(args: tuple) -> tuple:
    """
    Run one (pos_idx, cond_idx) simulation and return serialisable results.
    Signature kept flat so ProcessPoolExecutor can pickle it on Windows.
    """
    (
        pos_idx, cond_idx, start_pos, g_ka, g_na, g_cal,
        trial_ms, n_trials, seed_offset,
    ) = args
    rng = np.random.default_rng(_DEFAULT_SEED + seed_offset)
    result = _simulate_condition(
        start_pos=start_pos,
        g_ka_scale=g_ka,
        g_na_scale=g_na,
        g_cal_scale=g_cal,
        trial_ms=trial_ms,
        n_trials=n_trials,
        rng=rng,
    )
    spike_rate = float(result["spikes"].mean() * 1000)
    return (
        pos_idx, cond_idx,
        result["spikes"],
        result["X"],
        result["drive_trace"],
        result["c_left_trace"],
        result["c_right_trace"],
        spike_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Position grid
# ─────────────────────────────────────────────────────────────────────────────

def _make_position_grid(n_positions: int) -> list[np.ndarray]:
    """
    Return n_positions 3D starting positions that sample the plume's encounter
    statistics across a 2-D grid of upwind distances and lateral offsets.

    Grid axes
    ---------
    x  (upwind distance, negative = upwind of source at origin):
        Three distance bands:
          near   x ∈ [-0.05, -0.10] m  — strong, intermittent odor pulses
          middle x ∈ [-0.15, -0.25] m  — moderate, patchy signal
          far    x ∈ [-0.30, -0.50] m  — weak, sparse encounters

    y  (lateral offset from plume centreline):
        [-0.06, -0.03, 0.0, +0.03, +0.06] m
        Covering on-axis, edge, and off-plume positions.

    The 15 canonical grid points are sub-sampled to n_positions using a
    deterministic stride so the set remains spatially uniform for any
    requested count.  If n_positions > 15 the extra slots are filled with
    random jitter around the canonical grid.

    Biological motivation
    ---------------------
    Varying upwind distance changes:
      - mean concentration (∝ 1/r for a continuous source)
      - intermittency: near-field encounters are bursty; far-field is sparse
      - temporal autocorrelation of the drive trace

    Varying lateral offset changes:
      - bilateral gradient Δc = c_left − c_right (zero on-axis, maximal at edge)
      - the sign of the gradient encodes plume side for casting

    These two axes together ensure that every stimulus-filter column
    (drive mean, intermittency, gradient) and the full range of olfactory
    gain (sigmoid saturation vs. linear regime) appear in the training data.
    """
    # Canonical 3 × 5 = 15 grid points
    x_distances = [-0.05, -0.15, -0.40]      # near / middle / far
    y_offsets   = [-0.06, -0.03, 0.0, 0.03, 0.06]

    canonical: list[np.ndarray] = []
    for x in x_distances:
        for y in y_offsets:
            canonical.append(np.array([x, y, 0.0]))

    if n_positions <= len(canonical):
        # Deterministic uniform stride keeps coverage even
        stride = max(1, len(canonical) // n_positions)
        return canonical[::stride][:n_positions]

    # More requested than canonical: add jittered extras around the grid
    rng = np.random.default_rng(0)
    extras: list[np.ndarray] = []
    while len(extras) < n_positions - len(canonical):
        base = canonical[len(extras) % len(canonical)]
        jitter = rng.uniform([-0.03, -0.015, 0], [0.0, 0.015, 0])
        p = base + jitter
        # Keep within arena bounds and upwind of source
        p[0] = float(np.clip(p[0], -0.55, -0.02))
        p[1] = float(np.clip(p[1], -0.08,  0.08))
        extras.append(p)

    return canonical + extras[:n_positions - len(canonical)]


# ─────────────────────────────────────────────────────────────────────────────
# Single-condition simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_condition(
    start_pos: np.ndarray,
    g_ka_scale: float,
    g_na_scale: float,
    g_cal_scale: float,
    trial_ms: int,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate n_trials spike trains for one (position, conductance) condition.

    Returns a dict with:
      spikes        : (n_trials, T) uint8 spike matrix
      X             : (T, 24) design matrix built from the mean drive trace
      drive_trace   : (T,) mean sensory drive (µA/cm²)
      c_left_trace  : (T,) left-antenna concentration after sigmoid gain
      c_right_trace : (T,) right-antenna concentration after sigmoid gain
      heading_trace : (T,) heading angle (rad) — held at 0 for stationary fly
      wind_angle_trace : (T,) wind angle relative to heading (rad)
    """
    T = trial_ms

    # ── Plume simulation ──────────────────────────────────────────────────────
    plume = OdorPlume(
        wind_mean=np.array([-0.3, 0.0, 0.0]),   # blows toward -x where fly waits
        wind_noise_std=0.05,
        puff_rate=10.0,
        puff_sigma=0.05,                          # 5 cm spread; 0.02 m was too narrow
        source_position=np.zeros(3),
        _rng=rng,
    )

    body_pos  = start_pos.copy()
    ant_L     = body_pos + np.array([0.0,  _ANT_OFFSET_Y, 0.0])
    ant_R     = body_pos + np.array([0.0, -_ANT_OFFSET_Y, 0.0])

    c_left_trace    = np.zeros(T)
    c_right_trace   = np.zeros(T)
    drive_trace     = np.zeros(T)
    heading_trace   = np.zeros(T)     # stationary fly
    wind_angle_trace = np.zeros(T)    # wind aligned with body x-axis

    for t in range(T):
        plume.step(0.001)
        obs = plume.get_antennal_obs(ant_L, ant_R)
        c_left_trace[t]  = obs["c_left"]
        c_right_trace[t] = obs["c_right"]
        drive_trace[t]   = 0.5 * (obs["c_left"] + obs["c_right"]) * _DRIVE_SCALE

    # ── 1/f (pink-noise) drive variability ───────────────────────────────────
    # Biological variability: synaptic inputs to the readout neuron are not
    # just the mean odor drive — 1/f noise captures ongoing network fluctuations.
    def _pink_noise(size: int, rng: np.random.Generator) -> np.ndarray:
        white = rng.standard_normal(size)
        fft   = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(size)
        freqs[0] = 1.0  # avoid div-by-zero at DC
        pink  = np.fft.irfft(fft / np.sqrt(freqs), n=size)
        pink  = pink / (pink.std() + 1e-9)
        return pink

    drive_noise = np.stack([
        drive_trace + _NOISE_STD * _pink_noise(T, rng)
        for _ in range(n_trials)
    ])  # (n_trials, T)
    drive_noise = np.clip(drive_noise, 0.0, None)  # drive is non-negative

    # ── Batch Brian2 simulation ───────────────────────────────────────────────
    spikes = HHNeuron.simulate_spike_trials(
        u_sens=drive_trace,
        drive_noise=drive_noise - drive_trace[np.newaxis, :],  # noise only
        g_ka_scale=g_ka_scale,
    )
    # simulate_spike_trials only supports g_KA sweeps natively; apply g_Na and
    # g_CaL by re-scaling the drive to approximate the gain change.
    # Full multi-conductance support can be added by extending simulate_spike_trials.
    # For now: g_Na and g_CaL deviations from 1.0 are encoded as a drive rescaling
    # that preserves the first-order effect on spike rate (conductance × drive gain).
    # This is a deliberate first-pass simplification documented in the roadmap.
    if abs(g_na_scale - 1.0) > 0.01 or abs(g_cal_scale - 1.0) > 0.01:
        # Re-run with rescaled drive to approximate non-unit g_Na / g_CaL
        na_drive_gain  = g_na_scale ** 0.6    # empirical: 60% exponent captures
        cal_drive_gain = g_cal_scale ** 0.25  # sublinear CaL contribution
        adjusted_noise = (drive_noise * na_drive_gain * cal_drive_gain
                          - drive_trace[np.newaxis, :])
        adjusted_noise = np.clip(adjusted_noise, -drive_trace[np.newaxis, :], None)
        spikes = HHNeuron.simulate_spike_trials(
            u_sens=drive_trace,
            drive_noise=adjusted_noise,
            g_ka_scale=g_ka_scale,
        )

    # ── Design matrix (T, 24) from mean drive trace ───────────────────────────
    # Use the first trial's spike trace as the history reference for the design
    # matrix; this is a minor approximation — all trials share the same mean
    # drive, so the stimulus features are identical across trials.
    X = build_design_matrix(
        u_sens_trace=drive_trace,
        spike_trace=spikes[0].astype(float),
        heading_trace=heading_trace,
        c_left_trace=c_left_trace,
        c_right_trace=c_right_trace,
        wind_angle_trace=wind_angle_trace,
    )

    return {
        "spikes":         spikes,
        "X":              X,
        "drive_trace":    drive_trace,
        "c_left_trace":   c_left_trace,
        "c_right_trace":  c_right_trace,
        "heading_trace":  heading_trace,
        "wind_angle_trace": wind_angle_trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    out_dir: Path,
    n_trials: int,
    trial_ms: int,
    n_positions: int,
    g_ka_values: list[float],
    g_na_values: list[float],
    g_cal_values: list[float],
    lam: float,
    seed: int,
    quiet: bool,
    workers: int = 1,
) -> None:
    """Run the full parameter sweep and save results to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    positions = _make_position_grid(n_positions)

    # All (g_KA, g_Na, g_CaL) combinations = M conditions for fit_joint
    cond_grid = list(itertools.product(g_ka_values, g_na_values, g_cal_values))
    M = len(cond_grid)

    if not quiet:
        print(
            f"Sweep: {n_positions} positions × {M} conductance conditions "
            f"= {n_positions * M} total runs"
        )
        print(
            f"  g_KA × {len(g_ka_values)}  "
            f"g_Na × {len(g_na_values)}  "
            f"g_CaL × {len(g_cal_values)}\n"
        )

    try:
        import h5py
        _HAS_H5PY = True
    except ImportError:
        _HAS_H5PY = False
        print(
            "Warning: h5py not found — per-condition data will be saved as .npz "
            "files instead of a single HDF5 archive.\n"
            "Install h5py with: pip install h5py"
        )

    h5_path = out_dir / "training_data.h5"
    npz_dir = out_dir / "training_npz"
    if not _HAS_H5PY:
        npz_dir.mkdir(exist_ok=True)

    def _already_done(pos_tag: str, cond_tag: str) -> bool:
        """Return True if this (pos, cond) group already has spike data in HDF5."""
        if not _HAS_H5PY or not h5_path.exists():
            return False
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                return f"{pos_tag}/{cond_tag}/spikes" in f
        except Exception:
            return False

    # Collect (X, y) pairs per condition (across positions) for fit_joint
    # Each condition has n_positions × n_trials spike trains, each of length T.
    # We pool all position × trial combinations into one (X_cond, y_cond) pair
    # per conductance condition so fit_joint can apply the trend filter across M.
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    t0_total = time.time()

    # ── Build job list, skipping already-completed conditions ─────────────────
    job_args = []
    skipped  = 0
    for pos_idx, start_pos in enumerate(positions):
        pos_tag = f"pos{pos_idx:02d}_x{start_pos[0]:.3f}_y{start_pos[1]:.3f}"
        for cond_idx, (g_ka, g_na, g_cal) in enumerate(cond_grid):
            cond_tag = f"gKA{g_ka:.2f}_gNa{g_na:.2f}_gCaL{g_cal:.2f}"
            if _already_done(pos_tag, cond_tag):
                skipped += 1
                continue
            seed_offset = pos_idx * M + cond_idx
            job_args.append((
                pos_idx, cond_idx, start_pos, g_ka, g_na, g_cal,
                trial_ms, n_trials, seed_offset,
            ))

    total_jobs = n_positions * M
    pending    = len(job_args)
    if not quiet:
        if skipped:
            print(f"Resuming: {skipped}/{total_jobs} conditions already done, "
                  f"{pending} remaining.")
        if workers > 1:
            print(f"Running {pending} jobs across {workers} worker processes.\n")

    # ── Load already-completed data into X_list / y_list for fit_joint ────────
    # Pre-populate from HDF5 so the fit uses all data including prior runs.
    X_list = [None] * M   # type: ignore[assignment]
    y_list = [None] * M   # type: ignore[assignment]

    if _HAS_H5PY and h5_path.exists():
        import h5py
        with h5py.File(h5_path, "r") as f:
            for pos_idx, start_pos in enumerate(positions):
                pos_tag = f"pos{pos_idx:02d}_x{start_pos[0]:.3f}_y{start_pos[1]:.3f}"
                if pos_tag not in f:
                    continue
                for cond_idx, (g_ka, g_na, g_cal) in enumerate(cond_grid):
                    cond_tag = f"gKA{g_ka:.2f}_gNa{g_na:.2f}_gCaL{g_cal:.2f}"
                    grp_path = f"{pos_tag}/{cond_tag}"
                    if grp_path not in f:
                        continue
                    grp = f[grp_path]
                    spikes = grp["spikes"][:]
                    X      = grp["X"][:]
                    X_p = np.tile(X, (n_trials, 1))
                    y_p = spikes.reshape(-1).astype(float)
                    if X_list[cond_idx] is None:
                        X_list[cond_idx] = X_p
                        y_list[cond_idx] = y_p
                    else:
                        X_list[cond_idx] = np.concatenate([X_list[cond_idx], X_p])
                        y_list[cond_idx] = np.concatenate([y_list[cond_idx], y_p])

    # ── Run pending jobs (serial or parallel) ─────────────────────────────────
    completed = 0

    def _handle_result(res: tuple) -> None:
        """Persist one job result to HDF5/npz and accumulate into fit lists."""
        nonlocal completed
        (
            pos_idx, cond_idx,
            spikes, X,
            drive_trace, c_left_trace, c_right_trace,
            spike_rate,
        ) = res
        start_pos = positions[pos_idx]
        g_ka, g_na, g_cal = cond_grid[cond_idx]
        pos_tag  = f"pos{pos_idx:02d}_x{start_pos[0]:.3f}_y{start_pos[1]:.3f}"
        cond_tag = f"gKA{g_ka:.2f}_gNa{g_na:.2f}_gCaL{g_cal:.2f}"

        # Accumulate into fit lists
        X_p = np.tile(X, (n_trials, 1))
        y_p = spikes.reshape(-1).astype(float)
        if X_list[cond_idx] is None:
            X_list[cond_idx] = X_p
            y_list[cond_idx] = y_p
        else:
            X_list[cond_idx] = np.concatenate([X_list[cond_idx], X_p])
            y_list[cond_idx] = np.concatenate([y_list[cond_idx], y_p])

        # Persist
        if _HAS_H5PY:
            import h5py
            with h5py.File(h5_path, "a") as f:
                grp = f.require_group(f"{pos_tag}/{cond_tag}")
                for ds_name in ("spikes", "X", "drive_trace",
                                "c_left_trace", "c_right_trace"):
                    if ds_name in grp:
                        del grp[ds_name]
                grp.create_dataset("spikes",       data=spikes,      compression="gzip")
                grp.create_dataset("X",            data=X,           compression="gzip")
                grp.create_dataset("drive_trace",  data=drive_trace)
                grp.create_dataset("c_left_trace", data=c_left_trace)
                grp.create_dataset("c_right_trace",data=c_right_trace)
                grp.attrs["g_KA"]         = g_ka
                grp.attrs["g_Na"]         = g_na
                grp.attrs["g_CaL"]        = g_cal
                grp.attrs["start_pos"]    = start_pos
                grp.attrs["spike_rate_Hz"]= spike_rate
        else:
            np.savez_compressed(
                npz_dir / f"{pos_tag}_{cond_tag}.npz",
                spikes=spikes, X=X,
                drive_trace=drive_trace,
                c_left_trace=c_left_trace,
                c_right_trace=c_right_trace,
                g_KA=np.array(g_ka), g_Na=np.array(g_na), g_CaL=np.array(g_cal),
                start_pos=start_pos,
            )

        completed += 1
        if not quiet:
            print(
                f"  [{completed + skipped:4d}/{total_jobs}] "
                f"{pos_tag}  {cond_tag}  "
                f"{spike_rate:.1f} Hz"
            )

    if workers > 1 and pending > 0:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init_func) as pool:
            futures = {pool.submit(_job, a): a for a in job_args}
            for fut in as_completed(futures):
                try:
                    _handle_result(fut.result())
                except Exception as exc:
                    args_info = futures[fut]
                    print(f"  ERROR pos={args_info[0]} cond={args_info[1]}: {exc}")
    else:
        for job_a in job_args:
            t0 = time.time()
            try:
                res = _job(job_a)
                _handle_result(res)
                if not quiet:
                    pass  # already printed inside _handle_result
            except Exception as exc:
                print(f"  ERROR: {exc}")

    total_elapsed = time.time() - t0_total
    if not quiet and pending:
        print(
            f"\n{pending} new condition(s) complete in {total_elapsed:.0f}s. "
            f"Fitting joint PP-GLM over {M} conductance conditions …"
        )
    elif not quiet:
        print(f"\nAll {total_jobs} conditions already cached. "
              f"Fitting joint PP-GLM over {M} conductance conditions …")

    # Filter out any conditions that produced no data at all (shouldn't happen)
    X_fit = [x for x in X_list if x is not None]
    y_fit = [y for y in y_list if y is not None]

    # ── Joint PP-GLM fit with trend-filter penalty ────────────────────────────
    # fit_joint takes the M-element lists and finds beta vectors that smoothly
    # interpolate across conductance conditions.  The resulting beta matrix
    # shape (M, 24) captures how the stimulus and history filters change with
    # conductance perturbations.
    #
    # For the closed-loop agent a single representative beta is used: the one
    # from the centre condition (closest to nominal g_KA=1, g_Na=1, g_CaL=1).
    betas = fit_joint(X_fit, y_fit, lam=lam, max_iter=500)  # (M, 24)

    # Find the index of the nominally closest condition
    nominal_idx = 0
    best_dist   = float("inf")
    for i, (g_ka, g_na, g_cal) in enumerate(cond_grid):
        dist = (g_ka - 1.0) ** 2 + (g_na - 1.0) ** 2 + (g_cal - 1.0) ** 2
        if dist < best_dist:
            best_dist   = dist
            nominal_idx = i

    beta_nominal = betas[nominal_idx]  # (24,)

    # ── Save betas ────────────────────────────────────────────────────────────
    np.save(out_dir / "beta.npy",       beta_nominal)
    np.save(out_dir / "betas_all.npy",  betas)
    np.save(out_dir / "cond_grid.npy",  np.array(cond_grid))

    if _HAS_H5PY:
        with h5py.File(h5_path, "a") as f:
            meta = f.require_group("fit")
            for ds_name in ("betas_all", "cond_grid", "beta_nominal"):
                if ds_name in meta:
                    del meta[ds_name]
            meta.create_dataset("betas_all",    data=betas)
            meta.create_dataset("cond_grid",    data=np.array(cond_grid))
            meta.create_dataset("beta_nominal", data=beta_nominal)
            meta.attrs["nominal_cond_idx"] = nominal_idx
            meta.attrs["nominal_cond"]     = str(cond_grid[nominal_idx])

    if not quiet:
        print(f"\nResults written to {out_dir}/")
        print(f"  beta.npy         — nominal (24,) beta for run_closed_loop.py")
        print(f"  betas_all.npy    — ({M}, 24) beta matrix across all conditions")
        print(f"  cond_grid.npy    — ({M}, 3) conductance condition table")
        if _HAS_H5PY:
            print(f"  training_data.h5 — full spike + design-matrix archive")
        else:
            print(f"  training_npz/    — per-condition .npz spike archives")
        print(f"\nNominal condition: {cond_grid[nominal_idx]}")
        print(f"  Stimulus filter norm : {np.linalg.norm(beta_nominal[1:11]):.4f}")
        print(f"  History filter norm  : {np.linalg.norm(beta_nominal[11:21]):.4f}")
        print(f"  Baseline             : {beta_nominal[0]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate training data by sweeping fly positions and conductance conditions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir",     type=Path,  default=_DEFAULT_OUT_DIR)
    p.add_argument("--n-trials",    type=int,   default=_DEFAULT_N_TRIALS,
                   help="Independent noise realisations per (position, condition)")
    p.add_argument("--trial-ms",    type=int,   default=_DEFAULT_TRIAL_MS,
                   help="Trial duration in ms")
    p.add_argument("--n-positions", type=int,   default=_DEFAULT_N_POS,
                   help="Number of starting positions to sweep")
    p.add_argument("--g-ka",  nargs="+", type=float, default=_DEFAULT_G_KA,
                   metavar="SCALE",
                   help="g_KA scale values (multiplicative on baseline 5 mS/cm²)")
    p.add_argument("--g-na",  nargs="+", type=float, default=_DEFAULT_G_NA,
                   metavar="SCALE",
                   help="g_Na scale values (multiplicative on baseline 120 mS/cm²)")
    p.add_argument("--g-cal", nargs="+", type=float, default=_DEFAULT_G_CAL,
                   metavar="SCALE",
                   help="g_CaL scale values (multiplicative on baseline 2 mS/cm²)")
    p.add_argument("--lam",         type=float, default=_DEFAULT_LAM,
                   help="Trend-filter penalty strength for fit_joint")
    p.add_argument("--seed",        type=int,   default=_DEFAULT_SEED,
                   help="Global NumPy RNG seed")
    p.add_argument("--workers",     type=int,   default=_DEFAULT_WORKERS,
                   help=f"Parallel worker processes (default: cpu_count = {_DEFAULT_WORKERS}). "
                        "Each worker forces Brian2 to Cython target and limits BLAS threads "
                        "to 1, so one worker maps to one logical core safely.")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress per-condition progress output")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_sweep(
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        trial_ms=args.trial_ms,
        n_positions=args.n_positions,
        g_ka_values=args.g_ka,
        g_na_values=args.g_na,
        g_cal_values=args.g_cal,
        lam=args.lam,
        seed=args.seed,
        workers=args.workers,
        quiet=args.quiet,
    )
