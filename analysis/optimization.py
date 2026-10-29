from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import CableParameters
from models.chamber_parameters import ChamberParameters
from models.system_parameters import SystemParameters


@dataclass
class OptimizationWeights:
    w_loss: float = 1.0
    w_voltage: float = 1.0
    w_current: float = 1.0


@dataclass
class SweepResult:
    z0_ohm: np.ndarray
    cable_loss_w: np.ndarray
    cable_loss_percent: np.ndarray
    max_voltage_rms: np.ndarray
    max_current_rms: np.ndarray
    vswr: np.ndarray
    chamber_power_w: np.ndarray
    score: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def sweep_z0(
    system: SystemParameters,
    cable: CableParameters,
    chamber: ChamberParameters,
    z0_min: float,
    z0_max: float,
    z0_step: float,
    weights: OptimizationWeights,
) -> SweepResult:
    if z0_min <= 0 or z0_max <= 0 or z0_step <= 0 or z0_max < z0_min:
        raise ValueError("Invalid Z0 sweep bounds.")

    z_values = np.arange(z0_min, z0_max + 0.5 * z0_step, z0_step)
    loss_w = []
    loss_pct = []
    max_v = []
    max_i = []
    swr = []
    p_ch = []
    for z0 in z_values:
        local_cable = CableParameters(**{**cable.__dict__, "z0_ohm": float(z0)})
        result = analyze_cable(system, local_cable, chamber)
        loss_w.append(result.cable_loss_w)
        loss_pct.append(result.cable_loss_percent)
        max_v.append(float(np.max(np.abs(result.v_rms))))
        max_i.append(float(np.max(np.abs(result.i_rms))))
        swr.append(result.vswr)
        p_ch.append(result.chamber_power_w)

    loss_w_arr = np.array(loss_w, dtype=float)
    max_v_arr = np.array(max_v, dtype=float)
    max_i_arr = np.array(max_i, dtype=float)
    score = (
        weights.w_loss * _normalize(loss_w_arr)
        + weights.w_voltage * _normalize(max_v_arr)
        + weights.w_current * _normalize(max_i_arr)
    )
    return SweepResult(
        z0_ohm=z_values,
        cable_loss_w=loss_w_arr,
        cable_loss_percent=np.array(loss_pct, dtype=float),
        max_voltage_rms=max_v_arr,
        max_current_rms=max_i_arr,
        vswr=np.array(swr, dtype=float),
        chamber_power_w=np.array(p_ch, dtype=float),
        score=score,
    )

