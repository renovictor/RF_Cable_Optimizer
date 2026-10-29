from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    power_w: float
    load_setting: float
    tune_setting: float
    vpp_v: float
    iout_a_untrusted: float
    zreal_ohm: float
    zimag_ohm: float


DX7_OPERATING_POINTS: list[OperatingPoint] = [
    OperatingPoint(1300, 66.6, 28.4, 2751, 8.4, 2.62069, 19.702),
    OperatingPoint(1200, 61.9, 28.6, 2663, 8.0, 2.61526, 19.6662),
    OperatingPoint(1100, 57.2, 32.4, 2476, 7.8, 2.96938, 19.6843),
    OperatingPoint(1000, 57.9, 31.9, 2325, 7.3, 2.74712, 19.6089),
    OperatingPoint(900, 57.9, 31.9, 2206, 6.8, 2.74712, 19.6089),
    OperatingPoint(800, 56.3, 35.7, 2048, 6.4, 2.95271, 19.2561),
    OperatingPoint(700, 57.0, 35.3, 1877, 5.9, 2.9043, 19.2595),
    OperatingPoint(600, 56.3, 35.09, 1729, 5.3, 2.97486, 19.3792),
    OperatingPoint(500, 51.6, 39.8, 1545, 4.8, 3.21201, 19.1126),
    OperatingPoint(400, 44.5, 44.4, 1321, 4.2, 3.42165, 18.7923),
    OperatingPoint(300, 42.9, 48.7, 1088, 3.4, 3.83006, 18.5494),
    OperatingPoint(200, 42.2, 54.1, 825, 2.7, 3.71685, 17.8433),
    OperatingPoint(100, 64.19, 67.3, 839, 3.1, 1.60883, 13.9863),
]


def find_operating_point(power_w: float) -> OperatingPoint | None:
    for p in DX7_OPERATING_POINTS:
        if p.power_w == power_w:
            return p
    return None

