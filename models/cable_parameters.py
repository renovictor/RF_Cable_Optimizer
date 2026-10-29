from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CableModelMode(str, Enum):
    DATASHEET = "Datasheet Attenuation"
    RLGC = "Full RLGC"


class AttenuationUnit(str, Enum):
    DB_PER_M = "dB/m"
    DB_PER_100M = "dB/100m"
    DB_PER_100FT = "dB/100ft"


@dataclass
class CableSpecification:
    preset_name: str = "Custom"
    cable_model: str = "Unknown / Input Required"
    part_number: str = "Unknown / Input Required"
    max_voltage_rms: float | None = None
    max_current_rms: float | None = None
    max_temperature_c: float | None = None
    dielectric_material: str = "Unknown / Input Required"
    source_status: str = "USER INPUT"


@dataclass
class CableParameters:
    # Reference: cable starts at match output (z=0) and ends at chamber (z=L)
    z0_ohm: float = 50.0
    length_m: float | None = 1.03
    velocity_factor: float = 0.76
    attenuation_value: float | None = 0.6
    attenuation_unit: AttenuationUnit = AttenuationUnit.DB_PER_100FT
    model_mode: CableModelMode = CableModelMode.DATASHEET
    # RLGC per-meter parameters
    r_per_m: float | None = None
    l_per_m: float | None = None
    g_per_m: float | None = None
    c_per_m: float | None = None
    # Optional reference-plane electrical offsets
    match_plane_offset_m: float = 0.0
    chamber_plane_offset_m: float = 0.0
    specification: CableSpecification = field(default_factory=CableSpecification)
