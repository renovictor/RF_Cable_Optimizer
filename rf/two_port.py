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


@dataclass
class LoadedTwoPortResult:
    z11_ohm: complex
    z12_ohm: complex
    z21_ohm: complex
    z22_open_circuit_parameter_ohm: complex
    zin_ohm: complex
    zout_ohm: complex
    zload_ohm: complex
    source_impedance_ohm: complex
    i1_into_matcher_a: complex
    i2_into_matcher_a: complex
    iload_a: complex
    v1_rms_v: complex
    v2_rms_v: complex
    p_match_input_w: float
    p_match_output_w: float
    p_match_loss_w: float
    eta_match: float


def _validate_finite_complex(value: complex, name: str) -> None:
    if not (np.isfinite(value.real) and np.isfinite(value.imag)):
        raise ValueError(f"{name} must be finite.")


def s_to_z(s: SParameters) -> np.ndarray:
    if not np.isfinite(s.zref_ohm) or s.zref_ohm <= 0:
        raise ValueError("S-parameter reference impedance Zref must be > 0.")
    for name, value in (("S11", s.s11), ("S12", s.s12), ("S21", s.s21), ("S22", s.s22)):
        _validate_finite_complex(value, name)
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


def matcher_input_impedance(z: np.ndarray, z_load: complex) -> complex:
    z11, z12, z21, z22 = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
    denom = z22 + z_load
    if abs(denom) < 1e-12:
        raise ValueError("Z22 + ZL is too small for stable Zin calculation.")
    return z11 - (z12 * z21) / denom


def matcher_output_impedance(z: np.ndarray, z_source: complex) -> complex:
    z11, z12, z21, z22 = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
    denom = z11 + z_source
    if abs(denom) < 1e-12:
        raise ValueError("Z11 + ZS is too small for stable Zout calculation.")
    return z22 - (z21 * z12) / denom


def largest_singular_value_of_s(s: SParameters) -> float:
    singular_values = np.linalg.svd(s.matrix(), compute_uv=False)
    return float(np.max(np.abs(singular_values)))


def solve_loaded_two_port(s: SParameters, z_source: complex, z_load: complex, p_match_input_w: float) -> LoadedTwoPortResult:
    _validate_finite_complex(z_source, "Source impedance ZS")
    _validate_finite_complex(z_load, "Load impedance ZL")
    if p_match_input_w <= 0:
        raise ValueError("Match input power must be > 0 for loaded two-port solution.")

    z = s_to_z(s)
    z11, z12, z21, z22 = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
    zin = matcher_input_impedance(z, z_load)
    zout = matcher_output_impedance(z, z_source)

    if zin.real <= 0:
        raise ValueError("Loaded matcher input resistance must be > 0 for power normalization.")
    i1 = complex(np.sqrt(p_match_input_w / zin.real), 0.0)
    denom = z22 + z_load
    if abs(denom) < 1e-12:
        raise ValueError("Z22 + ZL is too small for stable loaded two-port current solution.")
    i2 = -(z21 / denom) * i1
    iload = -i2
    v1 = z11 * i1 + z12 * i2
    v2 = z21 * i1 + z22 * i2

    p_in = float(np.real(v1 * np.conj(i1)))
    p_out = float(-np.real(v2 * np.conj(i2)))
    p_loss = p_in - p_out
    eta = p_out / p_in if p_in > 0 else float("nan")

    return LoadedTwoPortResult(
        z11_ohm=z11,
        z12_ohm=z12,
        z21_ohm=z21,
        z22_open_circuit_parameter_ohm=z22,
        zin_ohm=zin,
        zout_ohm=zout,
        zload_ohm=z_load,
        source_impedance_ohm=z_source,
        i1_into_matcher_a=i1,
        i2_into_matcher_a=i2,
        iload_a=iload,
        v1_rms_v=v1,
        v2_rms_v=v2,
        p_match_input_w=p_in,
        p_match_output_w=p_out,
        p_match_loss_w=p_loss,
        eta_match=eta,
    )


def evaluate_match_efficiency(s: SParameters, z_source: complex, z_load: complex, v_source_rms: complex = 1 + 0j) -> tuple[float, float, float]:
    _ = v_source_rms
    try:
        solved = solve_loaded_two_port(s, z_source=z_source, z_load=z_load, p_match_input_w=1.0)
        return solved.p_match_input_w, solved.p_match_output_w, float(solved.eta_match)
    except ValueError:
        zref = complex(s.zref_ohm, 0.0)
        gs = (z_source - zref) / (z_source + zref)
        gl = (z_load - zref) / (z_load + zref)
        numerator = abs(s.s21) ** 2 * (1.0 - abs(gs) ** 2) * (1.0 - abs(gl) ** 2)
        denominator = abs((1.0 - s.s11 * gs) * (1.0 - s.s22 * gl) - s.s12 * s.s21 * gs * gl) ** 2
        if denominator <= 1e-18:
            raise ValueError("Numerical instability in transducer-gain efficiency calculation.")
        eta = float(np.real(numerator / denominator))
        return 1.0, eta, eta
