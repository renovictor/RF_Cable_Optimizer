from __future__ import annotations

import numpy as np


def _as_complex_array(values: np.ndarray | complex) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)


def impedance_from_vi(
    voltage_rms: np.ndarray,
    current_rms: np.ndarray,
    current_epsilon: float = 1e-12,
) -> np.ndarray:
    voltage = _as_complex_array(voltage_rms)
    current = _as_complex_array(current_rms)
    if voltage.shape != current.shape:
        raise ValueError("voltage_rms and current_rms must have identical shape.")

    impedance = np.full(voltage.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.abs(current) >= current_epsilon
    np.divide(voltage, current, out=impedance, where=valid)
    return impedance


def normalized_impedance(
    impedance: np.ndarray,
    reference_z0: complex,
    denominator_epsilon: float = 1e-12,
) -> np.ndarray:
    values = _as_complex_array(impedance)
    if abs(reference_z0) < denominator_epsilon:
        return np.full(values.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    return values / reference_z0


def gamma_from_impedance(
    impedance: np.ndarray,
    reference_z0: complex,
    denominator_epsilon: float = 1e-12,
) -> np.ndarray:
    values = _as_complex_array(impedance)
    gamma = np.full(values.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    denom = values + reference_z0
    valid = np.isfinite(values.real) & np.isfinite(values.imag) & (np.abs(denom) >= denominator_epsilon)
    np.divide(values - reference_z0, denom, out=gamma, where=valid)
    return gamma


def impedance_from_gamma(
    gamma: np.ndarray | complex,
    reference_z0: complex,
    denominator_epsilon: float = 1e-12,
) -> np.ndarray:
    gamma_values = _as_complex_array(gamma)
    impedance = np.full(gamma_values.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    denom = 1.0 - gamma_values
    valid = np.isfinite(gamma_values.real) & np.isfinite(gamma_values.imag) & (np.abs(denom) >= denominator_epsilon)
    np.divide(reference_z0 * (1.0 + gamma_values), denom, out=impedance, where=valid)
    return impedance


def vswr_from_gamma(gamma: complex, gamma_epsilon: float = 1e-12) -> float:
    if not (np.isfinite(gamma.real) and np.isfinite(gamma.imag)):
        return float("nan")
    magnitude = abs(gamma)
    if magnitude >= 1.0 - gamma_epsilon:
        return float("inf")
    return float((1.0 + magnitude) / (1.0 - magnitude))
