from __future__ import annotations

import numpy as np


def transmission_line_abcd(z0: complex, gamma: complex, length_m: float) -> np.ndarray:
    gl = gamma * length_m
    a = np.cosh(gl)
    b = z0 * np.sinh(gl)
    c = np.sinh(gl) / z0
    d = a
    return np.array([[a, b], [c, d]], dtype=np.complex128)


def cascade_abcd(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    # Port convention: matrix multiplication follows source->load order.
    # For Match Output -> Connector -> Cable End, ABCD_total = ABCD_connector @ ABCD_cable.
    return first @ second


def input_impedance_from_abcd(abcd: np.ndarray, z_load: complex) -> complex:
    a = abcd[0, 0]
    b = abcd[0, 1]
    c = abcd[1, 0]
    d = abcd[1, 1]
    denom = c * z_load + d
    if abs(denom) <= 1e-18:
        raise ValueError("ABCD input-impedance denominator is near zero.")
    return (a * z_load + b) / denom


def load_impedance_from_abcd(abcd: np.ndarray, z_in: complex) -> complex:
    a = abcd[0, 0]
    b = abcd[0, 1]
    c = abcd[1, 0]
    d = abcd[1, 1]
    denom = z_in * c - a
    if abs(denom) <= 1e-18:
        raise ValueError("ABCD de-embedding denominator is near zero.")
    return (b - z_in * d) / denom
