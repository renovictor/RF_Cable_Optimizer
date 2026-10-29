from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MatchOutputPowerSource(str, Enum):
    USER = "MEASURED/USER"
    CALCULATED = "CALCULATED"
    IDEAL = "IDEAL/ASSUMED"


class MatcherModelMode(str, Enum):
    IDEAL = "IDEAL"
    USER_S = "USER S-PARAMETERS"


@dataclass
class ComplexImpedance:
    real_ohm: float
    imag_ohm: float

    @property
    def value(self) -> complex:
        return complex(self.real_ohm, self.imag_ohm)


@dataclass
class WarningThresholds:
    vswr: float = 5.0
    cable_loss_percent: float = 10.0
    voltage_rms: float = 1000.0
    current_rms: float = 50.0
    power_balance_percent: float = 2.0


@dataclass
class SystemParameters:
    # Reference plane: generator output
    frequency_mhz: float = 60.0
    generator_power_w: float = 1300.0
    generator_impedance_ohm: float = 50.0
    # Matcher behavior
    match_efficiency_percent: float = 100.0
    match_output_power_source: MatchOutputPowerSource = MatchOutputPowerSource.IDEAL
    matcher_model_mode: MatcherModelMode = MatcherModelMode.IDEAL
    # Reference plane: match output / cable input
    match_output_power_w: float | None = None
    match_output_impedance: ComplexImpedance = field(default_factory=lambda: ComplexImpedance(3.0, 19.0))
    measured_vpp_v: float | None = None
    measured_iout_untrusted_a: float | None = None
    s11: complex = 0j
    s12: complex = 1 + 0j
    s21: complex = 1 + 0j
    s22: complex = 0j
    sparam_ref_impedance_ohm: float = 50.0
    sparam_frequency_mhz: float | None = None
    thresholds: WarningThresholds = field(default_factory=WarningThresholds)
