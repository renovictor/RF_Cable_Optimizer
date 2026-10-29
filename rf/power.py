from __future__ import annotations

import numpy as np


def real_power_from_phasors(v_rms: np.ndarray, i_rms: np.ndarray) -> np.ndarray:
    # RMS phasor convention: average real power is Re(V * conj(I)); no 1/2 factor.
    return np.real(v_rms * np.conjugate(i_rms))


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out

