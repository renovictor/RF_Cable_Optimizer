from __future__ import annotations

from dataclasses import dataclass

from analysis.cable_analysis import CableAnalysisResult
from thermal.engine import available_heating_sources
from thermal.models import ThermalHeatingSource, ThermalRequest


@dataclass
class ThermalValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_thermal_request(request: ThermalRequest, rf_result: CableAnalysisResult) -> ThermalValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    inputs = request.thermal_inputs
    if inputs.r_th_k_per_w <= 0:
        errors.append("Thermal resistance R_th must be > 0 K/W.")
    if inputs.c_th_j_per_k <= 0:
        errors.append("Thermal capacitance C_th must be > 0 J/K.")
    if inputs.time_step_s <= 0:
        errors.append("Thermal time step must be > 0 s.")
    if inputs.duration_s <= 0:
        errors.append("Thermal duration must be > 0 s.")
    if request.heating_source == ThermalHeatingSource.USER_ENTERED_HEATING_POWER:
        if request.user_entered_heating_power_w is None:
            errors.append("User-entered heating power is required for this source.")
        elif request.user_entered_heating_power_w < 0:
            errors.append("User-entered heating power must be >= 0 W.")
    if request.heating_source == ThermalHeatingSource.SELECTED_HOT_SPOT_REGION:
        if request.region.start_m == request.region.end_m:
            warnings.append("Hot-spot region start and end are equal; region heating is zero.")
    if request.heating_source == ThermalHeatingSource.SELECTED_RF_SECTION and not request.region.section_name:
        errors.append("Selected RF section must be provided for section heating source.")

    available_map = {item.source: item.enabled for item in available_heating_sources(rf_result)}
    if not available_map.get(request.heating_source, False):
        errors.append(f"Selected thermal heating source is not available: {request.heating_source.value}.")

    return ThermalValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
