from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SParameters:
    s11: complex = 0j
    s12: complex = 1 + 0j
    s21: complex = 1 + 0j
    s22: complex = 0j
    zref_ohm: float = 50.0

    def matrix(self) -> np.ndarray:
        return np.array([[self.s11, self.s12], [self.s21, self.s22]], dtype=complex)


def s_to_z(s: SParameters) -> np.ndarray:
    sm = s.matrix()
    eye = np.eye(2, dtype=complex)
    den = eye - sm
    if abs(np.linalg.det(den)) < 1e-12:
        raise ValueError("(I-S) is singular for S->Z conversion.")
    return s.zref_ohm * (eye + sm) @ np.linalg.inv(den)


def z_to_abcd(z: np.ndarray) -> np.ndarray:
    z11, z12, z21, z22 = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
    if abs(z21) < 1e-12:
        raise ValueError("z21 is too small for stable Z->ABCD conversion.")
    a = z11 / z21
    b = (z11 * z22 - z12 * z21) / z21
    c = 1 / z21
    d = z22 / z21
    return np.array([[a, b], [c, d]], dtype=complex)


def evaluate_match_efficiency(s: SParameters, z_source: complex, z_load: complex, v_source_rms: complex = 1 + 0j) -> tuple[float, float, float]:
    _ = v_source_rms
    zref = complex(s.zref_ohm, 0.0)
    gs = (z_source - zref) / (z_source + zref)
    gl = (z_load - zref) / (z_load + zref)
    numerator = abs(s.s21) ** 2 * (1.0 - abs(gs) ** 2) * (1.0 - abs(gl) ** 2)
    denominator = abs((1.0 - s.s11 * gs) * (1.0 - s.s22 * gl) - s.s12 * s.s21 * gs * gl) ** 2
    if denominator <= 1e-18:
        raise ValueError("Numerical instability in transducer-gain efficiency calculation.")
    eta = float(np.real(numerator / denominator))
    p_in = 1.0
    p_out = eta
    return p_in, p_out, eta
