from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from models.cable_parameters import AttenuationUnit


class ConnectorModelMode(str, Enum):
    DISABLED = "DISABLED"
    COORDINATE_OFFSET_ONLY = "COORDINATE OFFSET ONLY"
    TRANSMISSION_LINE = "TRANSMISSION-LINE SECTION"


class HistoricalCoordinateReference(str, Enum):
    FROM_MATCH_OUTPUT = "FROM MATCH OUTPUT"
    FROM_FLEXIBLE_CABLE_START = "FROM FLEXIBLE CABLE START"
    UNKNOWN_UNVERIFIED = "UNKNOWN / UNVERIFIED"


@dataclass
class ConnectorParameters:
    enabled: bool = False
    model_mode: ConnectorModelMode = ConnectorModelMode.DISABLED
    model_name: str = "Unknown / Input Required"
    physical_length_m: float | None = 0.0
    z0_ohm: float | None = 50.0
    velocity_factor: float | None = None
    attenuation_value: float | None = None
    attenuation_unit: AttenuationUnit = AttenuationUnit.DB_PER_M
    ideal_lossless: bool = False
    source_status: str = "USER INPUT"
    notes: str = ""
