from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.cable_parameters import CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.system_parameters import MatchOutputPowerSource, MatcherModelMode, SystemParameters
from rf.power import cumulative_trapezoid, real_power_from_phasors
from rf.smith_math import gamma_from_impedance, impedance_from_vi, normalized_impedance
from rf.transmission_line import (
    deembed_load_impedance,
    electrical_length_deg,
    input_impedance,
    normalize_boundary_from_real_power,
    propagation_constant_from_datasheet,
    propagation_constant_from_rlgc,
    reflection_coefficient,
    return_loss_db,
    solve_distributed_vi,
    to_db_per_m,
    vswr,
    z0_from_rlgc,
)
from rf.two_port import SParameters, evaluate_match_efficiency


@dataclass
class CableAnalysisResult:
    z_m: np.ndarray
    v_rms: np.ndarray
    i_rms: np.ndarray
    p_real_w: np.ndarray
    p_cond_w_per_m: np.ndarray
    p_diel_w_per_m: np.ndarray
    p_total_loss_w_per_m: np.ndarray
    p_cumulative_loss_w: np.ndarray
    z0_reference: complex
    z_local_ohm: np.ndarray
    z_normalized: np.ndarray
    gamma_local: np.ndarray
    z_load: complex
    z_in: complex
    z_from_vi_match: complex
    z_from_vi_end: complex
    endpoint_match_delta_ohm: complex
    endpoint_end_delta_ohm: complex
    endpoint_match_rel_error: float
    endpoint_end_rel_error: float
    gamma_l: complex
    gamma_phase_deg: float
    vswr: float
    return_loss_db: float
    electrical_length_deg: float
    match_input_power_w: float
    match_output_power_w: float
    match_output_power_source_label: str
    match_loss_w: float
    cable_input_power_w: float
    cable_output_power_w: float
    chamber_power_w: float
    cable_loss_w: float
    cable_loss_percent: float
    power_balance_error_w: float
    power_balance_error_percent: float
    warnings: list[str]


def _match_output_power(system: SystemParameters, z_in: complex) -> tuple[float, str]:
    if system.match_output_power_source == MatchOutputPowerSource.USER:
        if system.match_output_power_w is None:
            return system.match_input_power_w, "IDEAL MATCHER - NO MATCH LOSS ASSUMED"
        return system.match_output_power_w, MatchOutputPowerSource.USER.value
    if system.match_output_power_source == MatchOutputPowerSource.CALCULATED:
        if system.matcher_model_mode == MatcherModelMode.USER_S:
            try:
                s = SParameters(
                    s11=system.s11,
                    s12=system.s12,
                    s21=system.s21,
                    s22=system.s22,
                    zref_ohm=system.sparam_ref_impedance_ohm,
                )
                _, _, eta = evaluate_match_efficiency(
                    s, complex(system.generator_impedance_ohm, 0.0), z_in, v_source_rms=1 + 0j
                )
                eta = min(max(float(eta), 0.0), 1.0)
                return system.match_input_power_w * eta, "CALCULATED FROM MATCHER MODEL (S-PARAMETERS)"
            except Exception:
                return system.match_input_power_w, "IDEAL MATCHER - NO MATCH LOSS ASSUMED"
        eta = min(max(system.match_efficiency_percent / 100.0, 0.0), 1.0)
        return system.match_input_power_w * eta, "CALCULATED FROM MATCHER MODEL (EFFICIENCY)"
    return system.match_input_power_w, "IDEAL MATCHER - NO MATCH LOSS ASSUMED"


def analyze_cable(
    system: SystemParameters, cable: CableParameters, chamber: ChamberParameters, points: int = 1200
) -> CableAnalysisResult:
    if cable.length_m is None:
        raise ValueError("Cable length is required.")
    frequency_hz = system.frequency_mhz * 1e6
    line_length = cable.length_m + cable.match_plane_offset_m + cable.chamber_plane_offset_m
    if line_length <= 0:
        raise ValueError("Total electrical line length must be > 0.")

    if cable.model_mode == CableModelMode.DATASHEET:
        if cable.attenuation_value is None:
            raise ValueError("Attenuation value required in datasheet mode.")
        alpha_db_per_m = to_db_per_m(cable.attenuation_value, cable.attenuation_unit)
        gamma = propagation_constant_from_datasheet(frequency_hz, cable.velocity_factor, alpha_db_per_m)
        z0 = complex(cable.z0_ohm, 0.0)
    else:
        if None in (cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m):
            raise ValueError("RLGC mode requires R', L', G', C'.")
        gamma = propagation_constant_from_rlgc(
            frequency_hz, cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m  # type: ignore[arg-type]
        )
        z0 = z0_from_rlgc(
            frequency_hz, cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m  # type: ignore[arg-type]
        )

    z_match_measured = system.match_output_impedance.value
    if chamber.mode in (ChamberImpedanceMode.USER_ENTERED, ChamberImpedanceMode.MEASURED):
        if chamber.impedance is None:
            raise ValueError("Chamber impedance required for selected chamber mode.")
        z_load = chamber.impedance
        z_in = input_impedance(z_load, z0, gamma, line_length)
    else:
        z_in = z_match_measured
        z_load = deembed_load_impedance(z_in, z0, gamma, line_length)

    p_match_input = system.match_input_power_w
    p_match_output, p_match_source = _match_output_power(system, z_in)
    p_match_loss = p_match_input - p_match_output
    p_cable_input = p_match_output
    v0, _ = normalize_boundary_from_real_power(z_in, p_cable_input)

    z_m = np.linspace(0.0, line_length, points)
    v_rms, i_rms = solve_distributed_vi(z_m, line_length, z0, gamma, z_load, v0)
    p_real_w = real_power_from_phasors(v_rms, i_rms)
    z_local_ohm = impedance_from_vi(v_rms, i_rms)
    z_normalized = normalized_impedance(z_local_ohm, z0)
    gamma_local = gamma_from_impedance(z_local_ohm, z0)

    if cable.model_mode == CableModelMode.RLGC:
        p_cond_w_per_m = cable.r_per_m * np.abs(i_rms) ** 2  # type: ignore[operator]
        p_diel_w_per_m = cable.g_per_m * np.abs(v_rms) ** 2  # type: ignore[operator]
        p_total_loss_w_per_m = p_cond_w_per_m + p_diel_w_per_m
    else:
        p_total_loss_w_per_m = -np.gradient(p_real_w, z_m)
        p_cond_w_per_m = np.zeros_like(p_total_loss_w_per_m)
        p_diel_w_per_m = np.zeros_like(p_total_loss_w_per_m)

    p_cumulative_loss_w = cumulative_trapezoid(p_total_loss_w_per_m, z_m)
    p_cable_output = float(p_real_w[-1])
    p_chamber = p_cable_output
    p_cable_loss = p_cable_input - p_cable_output
    p_cable_loss_pct = (p_cable_loss / p_cable_input * 100.0) if p_cable_input > 0 else 0.0
    p_balance_err = p_match_input - p_match_loss - p_cable_loss - p_chamber
    p_balance_err_pct = (p_balance_err / p_match_input * 100.0) if p_match_input > 0 else 0.0

    gl = reflection_coefficient(z_load, z0)
    z_from_vi_match = complex(z_local_ohm[0])
    z_from_vi_end = complex(z_local_ohm[-1])
    endpoint_match_delta = z_from_vi_match - z_match_measured
    endpoint_end_delta = z_from_vi_end - z_load
    endpoint_match_rel_error = (
        abs(endpoint_match_delta) / abs(z_match_measured) if abs(z_match_measured) > 0 else float("inf")
    )
    endpoint_end_rel_error = abs(endpoint_end_delta) / abs(z_load) if abs(z_load) > 0 else float("inf")

    warnings: list[str] = []
    if vswr(gl) > system.thresholds.vswr:
        warnings.append("High VSWR threshold exceeded.")
    if p_cable_loss_pct > system.thresholds.cable_loss_percent:
        warnings.append("High cable loss threshold exceeded.")
    if float(np.max(np.abs(v_rms))) > system.thresholds.voltage_rms:
        warnings.append("High RMS voltage threshold exceeded.")
    if float(np.max(np.abs(i_rms))) > system.thresholds.current_rms:
        warnings.append("High RMS current threshold exceeded.")
    if abs(p_balance_err_pct) > system.thresholds.power_balance_percent:
        warnings.append("Power-balance error threshold exceeded.")
    if cable.model_mode == CableModelMode.DATASHEET:
        warnings.append("Conductor/dielectric loss split is not modeled in datasheet mode.")
    endpoint_warning_threshold = 1e-4
    if endpoint_match_rel_error > endpoint_warning_threshold:
        warnings.append(
            f"Endpoint mismatch at z=0 exceeds threshold: rel err={endpoint_match_rel_error:.3e}, "
            f"Delta_Z={endpoint_match_delta.real:+.4g}{endpoint_match_delta.imag:+.4g}j ohm."
        )
    if endpoint_end_rel_error > endpoint_warning_threshold:
        warnings.append(
            f"Endpoint mismatch at z=L exceeds threshold: rel err={endpoint_end_rel_error:.3e}, "
            f"Delta_Z={endpoint_end_delta.real:+.4g}{endpoint_end_delta.imag:+.4g}j ohm."
        )

    return CableAnalysisResult(
        z_m=z_m,
        v_rms=v_rms,
        i_rms=i_rms,
        p_real_w=p_real_w,
        p_cond_w_per_m=p_cond_w_per_m,
        p_diel_w_per_m=p_diel_w_per_m,
        p_total_loss_w_per_m=p_total_loss_w_per_m,
        p_cumulative_loss_w=p_cumulative_loss_w,
        z0_reference=z0,
        z_local_ohm=z_local_ohm,
        z_normalized=z_normalized,
        gamma_local=gamma_local,
        z_load=z_load,
        z_in=z_in,
        z_from_vi_match=z_from_vi_match,
        z_from_vi_end=z_from_vi_end,
        endpoint_match_delta_ohm=endpoint_match_delta,
        endpoint_end_delta_ohm=endpoint_end_delta,
        endpoint_match_rel_error=endpoint_match_rel_error,
        endpoint_end_rel_error=endpoint_end_rel_error,
        gamma_l=gl,
        gamma_phase_deg=float(np.degrees(np.angle(gl))),
        vswr=vswr(gl),
        return_loss_db=return_loss_db(gl),
        electrical_length_deg=electrical_length_deg(frequency_hz, cable.velocity_factor, line_length),
        match_input_power_w=p_match_input,
        match_output_power_w=p_match_output,
        match_output_power_source_label=p_match_source,
        match_loss_w=p_match_loss,
        cable_input_power_w=p_cable_input,
        cable_output_power_w=p_cable_output,
        chamber_power_w=p_chamber,
        cable_loss_w=p_cable_loss,
        cable_loss_percent=p_cable_loss_pct,
        power_balance_error_w=p_balance_err,
        power_balance_error_percent=p_balance_err_pct,
        warnings=warnings,
    )
