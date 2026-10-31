from __future__ import annotations

from dataclasses import dataclass
import math

from models.cable_parameters import CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import MatchOutputPowerSource, MatcherModelMode, SystemParameters


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_inputs(
    system: SystemParameters,
    cable: CableParameters,
    chamber: ChamberParameters,
    connector: ConnectorParameters | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if system.frequency_mhz <= 0:
        errors.append("Frequency must be > 0 MHz.")
    if system.match_input_power_w < 0:
        errors.append("Match input power must be >= 0 W.")
    if system.generator_impedance_ohm <= 0:
        errors.append("Source impedance ZS real part must be > 0 ohm.")
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

    if chamber.mode == ChamberImpedanceMode.DERIVED_FROM_DOWNSTREAM_PATH and chamber.impedance is None:
        warnings.append("Downstream RF-path mode is selected without chamber impedance; using historical load ZL as fallback.")
    if chamber.mode == ChamberImpedanceMode.INVERSE_MEASUREMENT_SOLVER:
        errors.append("Load mode is FUTURE / EXPERIMENTAL and is not yet implemented.")
    if chamber.mode == ChamberImpedanceMode.MEASURED_OR_USER_LOAD:
        if system.match_output_impedance.real_ohm is None or system.match_output_impedance.imag_ohm is None:
            errors.append("Load impedance ZL is required for measured/user-entered load mode.")
    if chamber.impedance is not None and chamber.impedance.real <= 0:
        warnings.append("Chamber resistance is non-positive; passive-load assumptions may be violated.")
    if system.match_output_impedance.real_ohm <= 0:
        warnings.append("Load resistance RL is non-positive; passive-load assumptions may be violated.")

    if system.matcher_model_mode == MatcherModelMode.USER_S:
        if system.sparam_ref_impedance_ohm <= 0:
            errors.append("S-parameter reference impedance must be > 0 ohm.")
        if system.sparam_frequency_mhz is None or system.sparam_frequency_mhz <= 0:
            errors.append("S-parameter frequency must be > 0 MHz.")
        for name, value in (("S11", system.s11), ("S12", system.s12), ("S21", system.s21), ("S22", system.s22)):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                errors.append(f"{name} must be finite.")
        if chamber.mode == ChamberImpedanceMode.DERIVED_FROM_DOWNSTREAM_PATH and chamber.impedance is None:
            warnings.append("Load impedance ZL is being taken from historical load fallback in downstream mode.")
        if system.match_output_power_source == MatchOutputPowerSource.CALCULATED and chamber.mode == ChamberImpedanceMode.INVERSE_MEASUREMENT_SOLVER:
            errors.append("Load impedance ZL is required for loaded S-parameter efficiency calculation.")

    if connector is not None and connector.enabled:
        connector_length = connector.physical_length_m
        if connector_length is None or connector_length < 0:
            errors.append("Connector length is required and must be >= 0 m.")
        if connector.model_mode == ConnectorModelMode.TRANSMISSION_LINE:
            if connector_length is None or connector_length <= 0:
                errors.append("Transmission-line connector mode requires length > 0 m.")
            if connector.z0_ohm is None or connector.z0_ohm <= 0:
                errors.append("Connector Z0 must be > 0 ohm.")
            if connector.velocity_factor is None or connector.velocity_factor <= 0 or connector.velocity_factor > 1:
                errors.append("Connector velocity factor must be in (0, 1].")
            if not connector.ideal_lossless and (connector.attenuation_value is None or connector.attenuation_value < 0):
                errors.append("Connector attenuation must be provided and >= 0 unless Ideal Lossless is enabled.")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
