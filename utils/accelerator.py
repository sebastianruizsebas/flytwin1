"""
GPU / JAX accelerator configuration for the fly-twin project.

This module is imported by all numerically intensive modules.  It detects
available GPU backends at import time and provides a unified interface so
that no other module needs to branch on hardware availability.

Exported names
==============
xp          — array module: ``jax.numpy`` if JAX + GPU/CPU-JAX available,
              otherwise plain ``numpy``.  Usage: ``xp.array([1, 2, 3])``.
np          — always ``numpy``.  Use for integer index arrays and structural
              operations that do not benefit from GPU (e.g. puff culling).
jit         — JIT decorator: ``jax.jit`` when JAX is available, identity
              function otherwise.  Apply to pure-function numerical kernels.
vmap        — vectorising map: ``jax.vmap`` when JAX available, raises
              RuntimeError otherwise.
to_xp(arr)  — convert a numpy array to an xp-array (no-op when xp is numpy).
to_np(arr)  — convert an xp-array to a numpy array (no-op when xp is numpy).
BRIAN2_TARGET — string passed to ``prefs.codegen.target``:
              ``"cuda_standalone"`` (brian2cuda) > ``"cython"`` > ``"numpy"``.
DEVICE      — ``"gpu"``, ``"cpu-jax"``, or ``"cpu"``.
HAS_JAX     — True if JAX import succeeded.

Notes
=====
* JAX arrays are immutable; use ``xp.concatenate`` / functional patterns
  instead of in-place mutation (``arr[i] = v``) in JIT-compiled functions.
* JAX tracing requires fixed shapes.  Variable-size loops (e.g. puff spawn/
  cull) must stay in numpy.
* brian2cuda requires a separate ``pip install brian2cuda`` and a CUDA 12
  toolkit.  The cython fallback gives ~3-10× CPU speedup over the numpy
  target on large batch simulations.
"""
from __future__ import annotations

import logging
import subprocess

import numpy as np  # always available

_log = logging.getLogger(__name__)

# ── JAX detection ─────────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp

    # Force initialization and device discovery
    _devices = jax.devices()
    _gpu_devices = [d for d in _devices if d.platform == "gpu"]

    if _gpu_devices:
        jax.config.update("jax_platform_name", "gpu")
        DEVICE: str = "gpu"
        _log.info("Accelerator: JAX GPU detected — %s", _gpu_devices[0])
    else:
        jax.config.update("jax_platform_name", "cpu")
        DEVICE = "cpu-jax"
        _log.info("Accelerator: JAX available, no GPU — using JAX on CPU (JIT still active)")

    xp = jnp
    jit = jax.jit
    vmap = jax.vmap
    HAS_JAX: bool = True

    def to_xp(arr: np.ndarray):  # type: ignore[return]
        return jnp.asarray(arr)

    def to_np(arr) -> np.ndarray:
        return np.asarray(arr)

except ImportError:
    xp = np  # type: ignore[assignment]
    DEVICE = "cpu"
    HAS_JAX = False

    def jit(fn, *args, **kwargs):  # type: ignore[misc]
        """Identity decorator — JAX not available."""
        return fn

    def vmap(fn, *args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "vmap requires JAX.  Install with:\n"
            "  pip install 'jax[cuda12]'    # CUDA GPU\n"
            "  pip install 'jax[cpu]'       # CPU-only"
        )

    def to_xp(arr: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return np.asarray(arr)

    def to_np(arr) -> np.ndarray:
        return np.asarray(arr)

    _log.info("Accelerator: JAX not found — using numpy on CPU")


# ── Brian2 codegen target detection ───────────────────────────────────────────

def _detect_brian2_target() -> str:
    """
    Return the best available Brian2 codegen target string.

    Priority: brian2cuda (CUDA GPU) > cython (CPU JIT) > numpy (pure Python).

    To enable GPU sims install brian2cuda:
        pip install brian2cuda
    and ensure the CUDA 12 toolkit is on PATH.
    """
    # CUDA via brian2cuda — Linux only (brian2cuda does not support Windows/macOS).
    import sys as _sys
    if _sys.platform == "linux":
        try:
            import brian2cuda  # noqa: F401  — side-effect: registers cuda_standalone device
            from brian2.codegen.targets import codegen_targets
            target_names = {t.class_name for t in codegen_targets}
            if "cuda_standalone" not in target_names:
                raise ImportError("cuda_standalone not accepted by Brian2 codegen layer")
            _log.info("Brian2 target: cuda_standalone (brian2cuda + CUDA runtime verified)")
            return "cuda_standalone"
        except (ImportError, Exception):
            pass
    else:
        _log.info("Brian2 target: skipping cuda_standalone check (brian2cuda is Linux-only)")

    # CPU JIT via Cython
    try:
        # Cython target is bundled with Brian2 >= 2.5 if a C++ compiler exists.
        # Test by importing the relevant device without actually compiling.
        from brian2.devices import cpp_standalone  # noqa: F401
        # Verify a compiler is reachable (distutils heuristic)
        import distutils.core  # noqa: F401
        result = subprocess.run(
            ["g++", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            _log.info("Brian2 target: cython (g++ found)")
            return "cython"
    except Exception:
        pass

    _log.info("Brian2 target: numpy (fallback)")
    return "numpy"


BRIAN2_TARGET: str = _detect_brian2_target()


# ── Brian2 thread count ────────────────────────────────────────────────────────
# Brian2 runtime (cython/numpy) does not use OpenMP internally — that is only
# available in cpp_standalone mode, which is incompatible with the per-step
# call pattern used by ConnectomeRNN.step().
#
# What does work for multi-core in runtime mode:
#   1. OMP_NUM_THREADS env-var: the Cython compiled extensions call NumPy
#      routines and scipy sparse ops which use OpenBLAS/MKL threads internally.
#      Setting OMP_NUM_THREADS / MKL_NUM_THREADS lets those use all cores.
#   2. Multiple parallel processes (one per simulation) — each process owns its
#      own Brian2 network; no shared state, scales linearly with CPU count.
#
# BRIAN2_NUM_THREADS: number of physical cores (or value of OMP_NUM_THREADS if
# already set), exported so callers can log/display it.

import os as _os
import multiprocessing as _mp

def _resolve_num_threads() -> int:
    """Return the thread count Brian2/NumPy will actually use."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        val = _os.environ.get(var)
        if val is not None:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return _mp.cpu_count()


BRIAN2_NUM_THREADS: int = _resolve_num_threads()

# Apply to all relevant environment variables so NumPy/SciPy sub-libraries
# respect the same count (idempotent if already set).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    if _var not in _os.environ:
        _os.environ[_var] = str(BRIAN2_NUM_THREADS)

_log.info(
    "Brian2 thread budget: %d threads (OMP_NUM_THREADS=%s). "
    "Runtime mode uses NumPy/SciPy threading; OpenMP-level parallelism "
    "requires cpp_standalone device (Linux only, incompatible with per-step calls).",
    BRIAN2_NUM_THREADS,
    _os.environ.get("OMP_NUM_THREADS"),
)
