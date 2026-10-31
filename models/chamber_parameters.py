from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChamberImpedanceMode(str, Enum):
    MEASURED_OR_USER_LOAD = "MEASURED / USER-ENTERED LOAD ZL"
    DERIVED_FROM_DOWNSTREAM_PATH = "DERIVED FROM DOWNSTREAM RF PATH"
    CONJUGATE_MATCH_TO_ZOUT = "CONJUGATE MATCH TO Zout"
    INVERSE_MEASUREMENT_SOLVER = "INVERSE SOLVED FROM MEASUREMENTS (FUTURE / EXPERIMENTAL)"
    USER_ENTERED = DERIVED_FROM_DOWNSTREAM_PATH
    MEASURED = DERIVED_FROM_DOWNSTREAM_PATH
    DERIVED_FROM_MATCH = MEASURED_OR_USER_LOAD


@dataclass
class ChamberParameters:
    mode: ChamberImpedanceMode = ChamberImpedanceMode.MEASURED_OR_USER_LOAD
    impedance_real_ohm: float | None = None
    impedance_imag_ohm: float | None = None

    @property
    def impedance(self) -> complex | None:
        if self.impedance_real_ohm is None or self.impedance_imag_ohm is None:
            return None
        return complex(self.impedance_real_ohm, self.impedance_imag_ohm)
