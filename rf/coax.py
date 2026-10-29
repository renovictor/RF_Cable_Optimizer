from __future__ import annotations

import math


def z0_from_geometry(d_outer_inner_mm: float, d_inner_outer_mm: float, epsilon_r: float) -> float:
    if d_outer_inner_mm <= 0 or d_inner_outer_mm <= 0 or epsilon_r <= 0:
        raise ValueError("D, d, and epsilon_r must be > 0.")
    if d_outer_inner_mm <= d_inner_outer_mm:
        raise ValueError("Outer-conductor inner diameter D must be larger than inner-conductor outer diameter d.")
    return (60.0 / math.sqrt(epsilon_r)) * math.log(d_outer_inner_mm / d_inner_outer_mm)


def inner_diameter_for_target_z0(target_z0_ohm: float, d_outer_inner_mm: float, epsilon_r: float) -> float:
    if target_z0_ohm <= 0 or d_outer_inner_mm <= 0 or epsilon_r <= 0:
        raise ValueError("Target Z0, D, and epsilon_r must be > 0.")
    return d_outer_inner_mm / math.exp(target_z0_ohm * math.sqrt(epsilon_r) / 60.0)

