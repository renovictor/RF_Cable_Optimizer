from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChamberImpedanceMode(str, Enum):
    USER_ENTERED = "USER-ENTERED CHAMBER IMPEDANCE"
    DERIVED_FROM_MATCH = "DERIVED FROM MATCH OUTPUT IMPEDANCE"
    MEASURED = "MEASURED CHAMBER IMPEDANCE"


@dataclass
class ChamberParameters:
    mode: ChamberImpedanceMode = ChamberImpedanceMode.DERIVED_FROM_MATCH
    impedance_real_ohm: float | None = None
    impedance_imag_ohm: float | None = None

    @property
    def impedance(self) -> complex | None:
        if self.impedance_real_ohm is None or self.impedance_imag_ohm is None:
            return None
        return complex(self.impedance_real_ohm, self.impedance_imag_ohm)

