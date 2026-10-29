from __future__ import annotations

import cmath
import math

import numpy as np

from models.cable_parameters import AttenuationUnit

C0 = 299_792_458.0
DB_TO_NP = 8.685889638


def to_db_per_m(value: float, unit: AttenuationUnit) -> float:
    if unit == AttenuationUnit.DB_PER_M:
        return value
    if unit == AttenuationUnit.DB_PER_100M:
        return value / 100.0
    if unit == AttenuationUnit.DB_PER_100FT:
        return value / 30.48
    raise ValueError(f"Unsupported attenuation unit: {unit}")


def db_per_m_to_np_per_m(alpha_db_per_m: float) -> float:
    # Uses user-specified conversion convention.
    return alpha_db_per_m / DB_TO_NP


def electrical_length_rad(frequency_hz: float, velocity_factor: float, length_m: float) -> float:
    wavelength = velocity_factor * C0 / frequency_hz
    return 2.0 * math.pi * length_m / wavelength


def electrical_length_deg(frequency_hz: float, velocity_factor: float, length_m: float) -> float:
    return math.degrees(electrical_length_rad(frequency_hz, velocity_factor, length_m))


def propagation_constant_from_datasheet(frequency_hz: float, velocity_factor: float, alpha_db_per_m: float) -> complex:
    alpha_np_per_m = db_per_m_to_np_per_m(alpha_db_per_m)
    beta = electrical_length_rad(frequency_hz, velocity_factor, 1.0)
    return complex(alpha_np_per_m, beta)


def propagation_constant_from_rlgc(
    frequency_hz: float, r_per_m: float, l_per_m: float, g_per_m: float, c_per_m: float
) -> complex:
    omega = 2.0 * math.pi * frequency_hz
    return cmath.sqrt(complex(r_per_m, omega * l_per_m) * complex(g_per_m, omega * c_per_m))


def z0_from_rlgc(frequency_hz: float, r_per_m: float, l_per_m: float, g_per_m: float, c_per_m: float) -> complex:
    omega = 2.0 * math.pi * frequency_hz
    return cmath.sqrt(complex(r_per_m, omega * l_per_m) / complex(g_per_m, omega * c_per_m))


def reflection_coefficient(z_load: complex, z0: complex) -> complex:
    return (z_load - z0) / (z_load + z0)


def return_loss_db(gamma: complex) -> float:
    mag = abs(gamma)
    if mag == 0:
        return float("inf")
    return -20.0 * math.log10(mag)


def vswr(gamma: complex) -> float:
    mag = abs(gamma)
    if mag >= 1:
        return float("inf")
    return (1 + mag) / (1 - mag)


def input_impedance(z_load: complex, z0: complex, gamma: complex, length_m: float) -> complex:
    gl = reflection_coefficient(z_load, z0)
    gin = gl * cmath.exp(-2.0 * gamma * length_m)
    return z0 * (1 + gin) / (1 - gin)


def deembed_load_impedance(z_in: complex, z0: complex, gamma: complex, length_m: float) -> complex:
    gin = (z_in - z0) / (z_in + z0)
    gl = gin * cmath.exp(2.0 * gamma * length_m)
    return z0 * (1 + gl) / (1 - gl)


def normalize_boundary_from_real_power(z_in: complex, p_in_real_w: float) -> tuple[complex, complex]:
    if z_in.real <= 0:
        raise ValueError("Input resistance at cable input must be > 0 for real-power normalization.")
    i_rms_mag = math.sqrt(p_in_real_w / z_in.real)
    i0 = complex(i_rms_mag, 0.0)
    v0 = z_in * i0
    return v0, i0


def solve_distributed_vi(
    z_positions_m: np.ndarray,
    line_length_m: float,
    z0: complex,
    gamma: complex,
    z_load: complex,
    v0: complex,
) -> tuple[np.ndarray, np.ndarray]:
    # Coordinate convention:
    # z = 0 at Match Output (cable input), z = L at Chamber (load).
    gl = reflection_coefficient(z_load, z0)
    gin = gl * np.exp(-2.0 * gamma * line_length_m)
    vf0 = v0 / (1.0 + gin)
    vr0 = gin * vf0
    vf = vf0 * np.exp(-gamma * z_positions_m)
    vr = vr0 * np.exp(gamma * z_positions_m)
    v = vf + vr
    i = (vf - vr) / z0
    return v, i

