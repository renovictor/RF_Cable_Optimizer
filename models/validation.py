from __future__ import annotations

from dataclasses import dataclass

from models.cable_parameters import CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.system_parameters import SystemParameters


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_inputs(system: SystemParameters, cable: CableParameters, chamber: ChamberParameters) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if system.frequency_mhz <= 0:
        errors.append("Frequency must be > 0 MHz.")
    if system.match_input_power_w < 0:
        errors.append("Match input power must be >= 0 W.")
    if cable.length_m is None or cable.length_m <= 0:
        errors.append("Cable length is required and must be > 0 m.")
    if cable.z0_ohm <= 0:
        errors.append("Cable characteristic impedance Z0 must be > 0 ohm.")
    if cable.velocity_factor <= 0 or cable.velocity_factor > 1:
        errors.append("Velocity factor must be in (0, 1].")

    if cable.model_mode == CableModelMode.DATASHEET:
        if cable.attenuation_value is None or cable.attenuation_value < 0:
            errors.append("Datasheet attenuation must be provided and >= 0.")
    else:
        if None in (cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m):
            errors.append("RLGC mode requires R', L', G', C'.")

    if chamber.mode in (ChamberImpedanceMode.USER_ENTERED, ChamberImpedanceMode.MEASURED):
        if chamber.impedance is None:
            errors.append("Chamber impedance real and imaginary parts are required for selected mode.")
    if chamber.impedance is not None and chamber.impedance.real <= 0:
        warnings.append("Chamber resistance is non-positive; passive-load assumptions may be violated.")
    if system.match_output_impedance.real_ohm <= 0:
        warnings.append("Match output resistance is non-positive; normalization may be invalid.")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
