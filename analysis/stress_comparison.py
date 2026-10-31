from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.cable_analysis import CableAnalysisResult


def matched_current_rms(match_output_power_w: float, reference_z0_ohm: float) -> float:
    if reference_z0_ohm <= 0:
        raise ValueError("reference_z0_ohm must be > 0.")
    if match_output_power_w <= 0:
        return 0.0
    return float(np.sqrt(match_output_power_w / reference_z0_ohm))


def matched_voltage_rms(match_output_power_w: float, reference_z0_ohm: float) -> float:
    if reference_z0_ohm <= 0:
        raise ValueError("reference_z0_ohm must be > 0.")
    if match_output_power_w <= 0:
        return 0.0
    return float(np.sqrt(match_output_power_w * reference_z0_ohm))


def current_enhancement(actual_max_current_rms_a: float, matched_current_rms_a: float) -> float:
    if matched_current_rms_a <= 0:
        return float("nan")
    return float(actual_max_current_rms_a / matched_current_rms_a)


def voltage_enhancement(actual_max_voltage_rms_v: float, matched_voltage_rms_v: float) -> float:
    if matched_voltage_rms_v <= 0:
        return float("nan")
    return float(actual_max_voltage_rms_v / matched_voltage_rms_v)


def current_squared_stress_index(current_enhancement_factor: float) -> float:
    if not np.isfinite(current_enhancement_factor):
        return float("nan")
    return float(current_enhancement_factor * current_enhancement_factor)


def equivalent_matched_current_from_rating(manufacturer_power_rating_w: float, reference_z0_ohm: float) -> float:
    if reference_z0_ohm <= 0:
        raise ValueError("reference_z0_ohm must be > 0.")
    if manufacturer_power_rating_w <= 0:
        return 0.0
    return float(np.sqrt(manufacturer_power_rating_w / reference_z0_ohm))


def local_current_enhancement_profile(i_rms: np.ndarray, matched_current_rms_a: float) -> np.ndarray:
    i_abs = np.abs(np.asarray(i_rms, dtype=np.complex128))
    if matched_current_rms_a <= 0:
        return np.full(i_abs.shape, np.nan, dtype=float)
    return i_abs / matched_current_rms_a


@dataclass
class RFStressComparisonResult:
    match_output_power_w: float
    reference_z0_ohm: float
    matched_current_rms_a: float
    matched_voltage_rms_v: float
    actual_max_current_rms_a: float
    actual_max_current_position_m: float
    actual_max_voltage_rms_v: float
    actual_max_voltage_position_m: float
    minimum_impedance_ohm: complex
    minimum_impedance_position_m: float
    maximum_local_loss_w_per_m: float
    maximum_local_loss_position_m: float
    current_enhancement_factor: float
    voltage_enhancement_factor: float
    current_squared_stress_index: float
    manufacturer_power_rating_w: float | None
    manufacturer_rating_frequency_hz: float | None
    manufacturer_equivalent_current_rms_a: float | None
    manufacturer_value_interpolated: bool
    manufacturer_rating_status: str
    interpretation_text: str
    warnings: list[str] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    local_current_enhancement: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def build_stress_comparison_result(
    analysis_result: CableAnalysisResult,
    manufacturer_power_rating_w: float | None = None,
    manufacturer_rating_frequency_hz: float | None = None,
    manufacturer_value_interpolated: bool = False,
    manufacturer_rating_status: str = "NOT DIRECTLY SPECIFIED",
) -> RFStressComparisonResult:
    reference_z0_ohm = float(np.real(analysis_result.z0_reference))
    try:
        i_matched = matched_current_rms(analysis_result.match_output_power_w, reference_z0_ohm)
        v_matched = matched_voltage_rms(analysis_result.match_output_power_w, reference_z0_ohm)
    except ValueError:
        i_matched = float("nan")
        v_matched = float("nan")

    max_i_idx = (
        int(analysis_result.max_current_index)
        if analysis_result.max_current_index is not None
        else int(np.argmax(np.abs(analysis_result.i_rms)))
    )
    max_v_idx = int(np.argmax(np.abs(analysis_result.v_rms)))
    max_loss_idx = int(np.argmax(analysis_result.p_total_loss_w_per_m))

    actual_i = float(np.abs(analysis_result.i_rms[max_i_idx]))
    actual_v = float(np.abs(analysis_result.v_rms[max_v_idx]))
    k_i = current_enhancement(actual_i, i_matched)
    k_v = voltage_enhancement(actual_v, v_matched)
    k_i2 = current_squared_stress_index(k_i)
    local_k_i = local_current_enhancement_profile(analysis_result.i_rms, i_matched if np.isfinite(i_matched) else 0.0)

    manufacturer_equivalent_current_rms_a: float | None = None
    if manufacturer_power_rating_w is not None:
        try:
            manufacturer_equivalent_current_rms_a = equivalent_matched_current_from_rating(
                manufacturer_power_rating_w, reference_z0_ohm
            )
        except ValueError:
            manufacturer_equivalent_current_rms_a = None

    interpretation_text = (
        f"At the current Match Output Power, an ideal matched line would carry {i_matched:.2f} Arms. "
        f"The distributed RF result predicts a local maximum of {actual_i:.2f} Arms at {analysis_result.z_m[max_i_idx]:.3f} m. "
        f"Current enhancement is {k_i:.2f}x and the relative local I-squared stress index is {k_i2:.1f}x."
        if np.isfinite(i_matched)
        else "At the current operating condition, matched-reference current is not available."
    )

    source_status = {
        "Manufacturer Power Rating": "MANUFACTURER SPEC" if manufacturer_power_rating_w is not None else "UNKNOWN",
        "Matched Current": "DERIVED",
        "Actual Maximum Current": "RF MODEL",
        "Current Enhancement": "DERIVED",
        "I-squared Stress Index": "DERIVED",
        "Current Limit": "UNKNOWN",
    }
    warnings = [
        "RELATIVE STRESS ONLY. LIMIT NOT DEFINED.",
        "I-squared stress index is not a temperature prediction or manufacturer derating factor.",
    ]

    min_z = analysis_result.min_impedance_z_ohm if analysis_result.min_impedance_z_ohm is not None else complex("nan+nanj")
    min_z_pos = analysis_result.min_impedance_z_m if analysis_result.min_impedance_z_m is not None else float("nan")

    return RFStressComparisonResult(
        match_output_power_w=float(analysis_result.match_output_power_w),
        reference_z0_ohm=reference_z0_ohm,
        matched_current_rms_a=i_matched,
        matched_voltage_rms_v=v_matched,
        actual_max_current_rms_a=actual_i,
        actual_max_current_position_m=float(analysis_result.z_m[max_i_idx]),
        actual_max_voltage_rms_v=actual_v,
        actual_max_voltage_position_m=float(analysis_result.z_m[max_v_idx]),
        minimum_impedance_ohm=complex(min_z),
        minimum_impedance_position_m=float(min_z_pos),
        maximum_local_loss_w_per_m=float(analysis_result.p_total_loss_w_per_m[max_loss_idx]),
        maximum_local_loss_position_m=float(analysis_result.z_m[max_loss_idx]),
        current_enhancement_factor=k_i,
        voltage_enhancement_factor=k_v,
        current_squared_stress_index=k_i2,
        manufacturer_power_rating_w=manufacturer_power_rating_w,
        manufacturer_rating_frequency_hz=manufacturer_rating_frequency_hz,
        manufacturer_equivalent_current_rms_a=manufacturer_equivalent_current_rms_a,
        manufacturer_value_interpolated=manufacturer_value_interpolated,
        manufacturer_rating_status=manufacturer_rating_status,
        interpretation_text=interpretation_text,
        warnings=warnings,
        source_status=source_status,
        local_current_enhancement=local_k_i,
    )
