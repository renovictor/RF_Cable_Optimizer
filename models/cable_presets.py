from __future__ import annotations

from dataclasses import dataclass
from math import log10


AA11459_ATTENUATION_DB_PER_100FT: tuple[tuple[float, float], ...] = (
    (13.56, 0.29),
    (100.0, 0.80),
    (1000.0, 2.63),
    (2450.0, 3.68),
)

AA11459_POWER_RATING_KW: tuple[tuple[float, float], ...] = (
    (13.56, 53.24),
    (100.0, 19.39),
    (1000.0, 5.91),
    (2450.0, 4.30),
)


@dataclass(frozen=True)
class InterpolationResult:
    value: float
    lower_point: tuple[float, float]
    upper_point: tuple[float, float]


def interpolate_log_frequency(points: tuple[tuple[float, float], ...], target_mhz: float) -> InterpolationResult:
    if len(points) < 2:
        raise ValueError("At least two points are required for interpolation.")
    if target_mhz <= 0:
        raise ValueError("Target frequency must be > 0 MHz.")
    sorted_points = sorted(points, key=lambda p: p[0])
    if target_mhz < sorted_points[0][0] or target_mhz > sorted_points[-1][0]:
        raise ValueError("Target frequency is outside the published range.")

    for idx in range(1, len(sorted_points)):
        f1, v1 = sorted_points[idx - 1]
        f2, v2 = sorted_points[idx]
        if f1 <= target_mhz <= f2:
            if target_mhz == f1:
                return InterpolationResult(v1, (f1, v1), (f2, v2))
            if target_mhz == f2:
                return InterpolationResult(v2, (f1, v1), (f2, v2))
            x1 = log10(f1)
            x2 = log10(f2)
            xt = log10(target_mhz)
            t = (xt - x1) / (x2 - x1)
            return InterpolationResult(v1 + (v2 - v1) * t, (f1, v1), (f2, v2))
    raise ValueError("No interpolation segment was found.")
