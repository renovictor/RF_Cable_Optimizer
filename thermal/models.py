from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ThermalModel(Enum):
    LUMPED_RC = "Lumped Thermal RC"
    DISTRIBUTED_1D = "Distributed 1D Thermal (Future)"


class ThermalHeatingSource(Enum):
    FLEXIBLE_CABLE_TOTAL_DISSIPATION = "Flexible Cable Total Dissipation"
    WHOLE_CABLE_TOTAL_DISSIPATION = "Whole Cable Total Dissipation"
    SELECTED_HOT_SPOT_REGION = "Selected Hot-Spot Region"
    RIGID_SECTION_TOTAL_DISSIPATION = "Rigid Section Total Dissipation"
    CONNECTOR_TOTAL_DISSIPATION = "Connector Total Dissipation"
    SELECTED_RF_SECTION = "Selected RF Section"
    USER_ENTERED_HEATING_POWER = "User-Entered Heating Power"


@dataclass
class ThermalInputs:
    r_th_k_per_w: float = 1.5
    c_th_j_per_k: float = 300.0
    ambient_temp_c: float = 20.0
    initial_temp_c: float = 20.0
    target_temp_c: float = 80.0
    duration_s: float = 900.0
    time_step_s: float = 1.0


@dataclass
class ThermalRegionSelection:
    start_m: float = 0.0
    end_m: float = 0.3
    section_name: str | None = None


@dataclass
class ThermalRequest:
    model: ThermalModel
    heating_source: ThermalHeatingSource
    thermal_inputs: ThermalInputs
    region: ThermalRegionSelection
    user_entered_heating_power_w: float | None = None


@dataclass
class ThermalHeatingSelection:
    source: ThermalHeatingSource
    power_w: float
    source_label: str
    source_origin_label: str
    rf_derived: bool


@dataclass
class ThermalSimulationResult:
    model: ThermalModel
    heating: ThermalHeatingSelection
    time_s: np.ndarray
    temperature_c: np.ndarray
    steady_state_temp_c: float
    thermal_time_constant_s: float
    target_crossing_time_s: float | None
    warnings: list[str]
