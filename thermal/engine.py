from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.cable_analysis import CableAnalysisResult
from models.connector_parameters import ConnectorModelMode
from thermal.models import (
    ThermalHeatingSelection,
    ThermalHeatingSource,
    ThermalModel,
    ThermalRequest,
    ThermalSimulationResult,
)


@dataclass(frozen=True)
class ThermalSourceAvailability:
    source: ThermalHeatingSource
    enabled: bool


def _integrate_loss_over_region(z_m: np.ndarray, loss_w_per_m: np.ndarray, start_m: float, end_m: float) -> float:
    z_start = min(start_m, end_m)
    z_end = max(start_m, end_m)
    if z_end <= z_start:
        return 0.0
    z_min = float(z_m[0])
    z_max = float(z_m[-1])
    z_start = min(max(z_start, z_min), z_max)
    z_end = min(max(z_end, z_min), z_max)
    if z_end <= z_start:
        return 0.0
    base_mask = (z_m >= z_start) & (z_m <= z_end)
    z_slice = z_m[base_mask]
    loss_slice = loss_w_per_m[base_mask]
    if len(z_slice) == 0:
        z_slice = np.array([z_start, z_end], dtype=float)
        loss_slice = np.array(
            [np.interp(z_start, z_m, loss_w_per_m), np.interp(z_end, z_m, loss_w_per_m)],
            dtype=float,
        )
    else:
        if z_slice[0] > z_start:
            z_slice = np.insert(z_slice, 0, z_start)
            loss_slice = np.insert(loss_slice, 0, np.interp(z_start, z_m, loss_w_per_m))
        if z_slice[-1] < z_end:
            z_slice = np.append(z_slice, z_end)
            loss_slice = np.append(loss_slice, np.interp(z_end, z_m, loss_w_per_m))
    return float(np.trapezoid(loss_slice, z_slice))


def available_heating_sources(result: CableAnalysisResult) -> list[ThermalSourceAvailability]:
    has_profile = len(result.z_m) > 1 and len(result.p_total_loss_w_per_m) == len(result.z_m)
    has_connector_loss = (
        result.connector_enabled
        and result.connector_mode == ConnectorModelMode.TRANSMISSION_LINE.value
        and result.connector_loss_w is not None
    )
    has_rigid_section = any("rigid" in str(name).lower() for name in result.section_name)
    has_sections = len(result.section_name) > 0
    return [
        ThermalSourceAvailability(ThermalHeatingSource.FLEXIBLE_CABLE_TOTAL_DISSIPATION, True),
        ThermalSourceAvailability(ThermalHeatingSource.WHOLE_CABLE_TOTAL_DISSIPATION, True),
        ThermalSourceAvailability(ThermalHeatingSource.SELECTED_HOT_SPOT_REGION, has_profile),
        ThermalSourceAvailability(ThermalHeatingSource.RIGID_SECTION_TOTAL_DISSIPATION, has_rigid_section),
        ThermalSourceAvailability(ThermalHeatingSource.CONNECTOR_TOTAL_DISSIPATION, has_connector_loss),
        ThermalSourceAvailability(ThermalHeatingSource.SELECTED_RF_SECTION, has_sections and has_profile),
        ThermalSourceAvailability(ThermalHeatingSource.USER_ENTERED_HEATING_POWER, True),
    ]


def _whole_cable_total_dissipation_w(result: CableAnalysisResult) -> float:
    if len(result.p_cumulative_loss_w) > 0:
        return float(result.p_cumulative_loss_w[-1])
    connector_loss = float(result.connector_loss_w or 0.0)
    return float(result.flexible_cable_total_loss_w + connector_loss)


def _section_total_dissipation_w(result: CableAnalysisResult, section_name: str) -> float:
    mask = np.asarray([str(name) == section_name for name in result.section_name], dtype=bool)
    if not np.any(mask):
        return 0.0
    z = result.z_m[mask]
    wpm = result.p_total_loss_w_per_m[mask]
    if len(z) < 2:
        return 0.0
    return float(np.trapezoid(wpm, z))


def resolve_heating_selection(request: ThermalRequest, result: CableAnalysisResult) -> ThermalHeatingSelection:
    source = request.heating_source
    rf_source_label = "RF Cable Analysis"
    if source == ThermalHeatingSource.FLEXIBLE_CABLE_TOTAL_DISSIPATION:
        return ThermalHeatingSelection(
            source=source,
            power_w=float(result.flexible_cable_total_loss_w),
            source_label=source.value,
            source_origin_label=rf_source_label,
            rf_derived=True,
        )
    if source == ThermalHeatingSource.WHOLE_CABLE_TOTAL_DISSIPATION:
        return ThermalHeatingSelection(
            source=source,
            power_w=_whole_cable_total_dissipation_w(result),
            source_label=source.value,
            source_origin_label=rf_source_label,
            rf_derived=True,
        )
    if source == ThermalHeatingSource.SELECTED_HOT_SPOT_REGION:
        p_heat = _integrate_loss_over_region(
            result.z_m,
            result.p_total_loss_w_per_m,
            request.region.start_m,
            request.region.end_m,
        )
        return ThermalHeatingSelection(
            source=source,
            power_w=p_heat,
            source_label=f"{source.value} ({min(request.region.start_m, request.region.end_m):.3f} to {max(request.region.start_m, request.region.end_m):.3f} m)",
            source_origin_label=rf_source_label,
            rf_derived=True,
        )
    if source == ThermalHeatingSource.RIGID_SECTION_TOTAL_DISSIPATION:
        rigid_sections = sorted({str(name) for name in result.section_name if "rigid" in str(name).lower()})
        section = rigid_sections[0] if rigid_sections else ""
        p_heat = _section_total_dissipation_w(result, section) if section else 0.0
        return ThermalHeatingSelection(
            source=source,
            power_w=p_heat,
            source_label=f"{source.value} ({section})" if section else source.value,
            source_origin_label=rf_source_label,
            rf_derived=True,
        )
    if source == ThermalHeatingSource.CONNECTOR_TOTAL_DISSIPATION:
        connector_loss = float(result.connector_loss_w or 0.0)
        return ThermalHeatingSelection(
            source=source,
            power_w=connector_loss,
            source_label=source.value,
            source_origin_label=rf_source_label,
            rf_derived=True,
        )
    if source == ThermalHeatingSource.SELECTED_RF_SECTION:
        section_name = request.region.section_name or str(result.section_name[0])
        p_heat = _section_total_dissipation_w(result, section_name)
        return ThermalHeatingSelection(
            source=source,
            power_w=p_heat,
            source_label=f"{source.value} ({section_name})",
            source_origin_label=rf_source_label,
            rf_derived=True,
        )
    if source == ThermalHeatingSource.USER_ENTERED_HEATING_POWER:
        return ThermalHeatingSelection(
            source=source,
            power_w=float(request.user_entered_heating_power_w or 0.0),
            source_label=source.value,
            source_origin_label="USER INPUT / WHAT-IF ANALYSIS",
            rf_derived=False,
        )
    raise ValueError(f"Unsupported thermal heating source: {source}")


def solve_lumped_rc(request: ThermalRequest, heating: ThermalHeatingSelection) -> ThermalSimulationResult:
    if request.model != ThermalModel.LUMPED_RC:
        raise ValueError(f"Unsupported thermal model for solve_lumped_rc: {request.model.value}")
    ti = request.thermal_inputs
    if ti.time_step_s <= 0:
        raise ValueError("Thermal time step must be > 0 s.")
    if ti.duration_s <= 0:
        raise ValueError("Thermal duration must be > 0 s.")
    if ti.r_th_k_per_w <= 0:
        raise ValueError("Thermal resistance R_th must be > 0 K/W.")
    if ti.c_th_j_per_k <= 0:
        raise ValueError("Thermal capacitance C_th must be > 0 J/K.")

    n_steps = max(2, int(np.floor(ti.duration_s / ti.time_step_s)) + 1)
    time_s = np.linspace(0.0, ti.time_step_s * (n_steps - 1), n_steps)
    tau_s = ti.r_th_k_per_w * ti.c_th_j_per_k
    steady_state_temp_c = ti.ambient_temp_c + heating.power_w * ti.r_th_k_per_w
    transient_term = ti.initial_temp_c - steady_state_temp_c
    temperature_c = steady_state_temp_c + transient_term * np.exp(-time_s / tau_s)

    target_crossing_time_s: float | None = None
    above_target_indices = np.flatnonzero(temperature_c >= ti.target_temp_c)
    if len(above_target_indices) > 0:
        target_crossing_time_s = float(time_s[int(above_target_indices[0])])

    warnings: list[str] = []
    if not heating.rf_derived:
        warnings.append("Thermal heating uses USER INPUT / WHAT-IF ANALYSIS; not RF-derived.")
    return ThermalSimulationResult(
        model=request.model,
        heating=heating,
        time_s=time_s,
        temperature_c=temperature_c,
        steady_state_temp_c=steady_state_temp_c,
        thermal_time_constant_s=tau_s,
        target_crossing_time_s=target_crossing_time_s,
        warnings=warnings,
    )


def run_thermal_analysis(request: ThermalRequest, result: CableAnalysisResult) -> ThermalSimulationResult:
    heating = resolve_heating_selection(request, result)
    if request.model == ThermalModel.LUMPED_RC:
        return solve_lumped_rc(request, heating)
    raise NotImplementedError("Distributed 1D thermal model is not implemented in this version.")
