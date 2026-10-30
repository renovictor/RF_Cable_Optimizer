from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.cable_parameters import CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import MatchOutputPowerSource, MatcherModelMode, SystemParameters
from rf.network_cascade import cascade_abcd, input_impedance_from_abcd, transmission_line_abcd
from rf.power import cumulative_trapezoid, real_power_from_phasors
from rf.smith_math import gamma_from_impedance, impedance_from_vi, normalized_impedance
from rf.transmission_line import (
    C0,
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


CURRENT_EPSILON_A = 1e-12


@dataclass
class CableAnalysisResult:
    z_m: np.ndarray
    global_position_m: np.ndarray
    section_name: np.ndarray
    section_local_position_m: np.ndarray
    position_from_flexible_cable_start_m: np.ndarray
    cable_start_global_position_m: float
    connector_length_m: float
    connector_enabled: bool
    connector_mode: str
    connector_model_name: str
    connector_interface_index: int | None
    connector_electrical_length_deg: float | None
    connector_electrical_length_wavelength: float | None
    connector_propagation_delay_ns: float | None
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
    connector_input_power_w: float | None
    connector_output_power_w: float | None
    connector_loss_w: float | None
    cable_input_power_w: float
    flexible_cable_input_power_w: float
    cable_output_power_w: float
    chamber_power_w: float
    cable_loss_w: float
    flexible_cable_total_loss_w: float
    cable_end_delivered_power_w: float
    cable_loss_percent: float
    power_balance_error_w: float
    power_balance_error_percent: float
    min_impedance_index: int | None
    min_impedance_z_m: float | None
    min_impedance_z_ohm: complex | None
    min_impedance_r_ohm: float | None
    min_impedance_x_ohm: float | None
    min_impedance_magnitude_ohm: float | None
    min_impedance_irms_a: float | None
    min_impedance_vrms_v: float | None
    min_impedance_real_power_w: float | None
    max_current_index: int | None
    max_current_z_m: float | None
    max_current_z_ohm: complex | None
    max_current_r_ohm: float | None
    max_current_x_ohm: float | None
    max_current_magnitude_ohm: float | None
    max_current_irms_a: float | None
    max_current_vrms_v: float | None
    max_current_real_power_w: float | None
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


def _line_model(frequency_hz: float, cable: CableParameters) -> tuple[complex, complex]:
    if cable.length_m is None:
        raise ValueError("Cable length is required.")
    if cable.model_mode == CableModelMode.DATASHEET:
        if cable.attenuation_value is None:
            raise ValueError("Attenuation value required in datasheet mode.")
        alpha_db_per_m = to_db_per_m(cable.attenuation_value, cable.attenuation_unit)
        gamma = propagation_constant_from_datasheet(frequency_hz, cable.velocity_factor, alpha_db_per_m)
        z0 = complex(cable.z0_ohm, 0.0)
        return gamma, z0
    if None in (cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m):
        raise ValueError("RLGC mode requires R', L', G', C'.")
    gamma = propagation_constant_from_rlgc(
        frequency_hz, cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m  # type: ignore[arg-type]
    )
    z0 = z0_from_rlgc(
        frequency_hz, cable.r_per_m, cable.l_per_m, cable.g_per_m, cable.c_per_m  # type: ignore[arg-type]
    )
    return gamma, z0


def _connector_gamma(frequency_hz: float, connector: ConnectorParameters) -> complex:
    vf = connector.velocity_factor
    if vf is None:
        raise ValueError("Connector velocity factor is required.")
    attenuation = 0.0 if connector.ideal_lossless else float(connector.attenuation_value or 0.0)
    alpha_db_per_m = to_db_per_m(attenuation, connector.attenuation_unit)
    return propagation_constant_from_datasheet(frequency_hz, vf, alpha_db_per_m)


def _point_metrics(
    index: int | None, z_m: np.ndarray, z_local_ohm: np.ndarray, v_rms: np.ndarray, i_rms: np.ndarray, p_real_w: np.ndarray
) -> tuple[float | None, complex | None, float | None, float | None, float | None, float | None, float | None, float | None]:
    if index is None:
        return (None, None, None, None, None, None, None, None)
    z_value = complex(z_local_ohm[index])
    z_abs = float(abs(z_value))
    return (
        float(z_m[index]),
        z_value,
        float(z_value.real),
        float(z_value.imag),
        z_abs,
        float(abs(i_rms[index])),
        float(abs(v_rms[index])),
        float(p_real_w[index]),
    )


def analyze_cable(
    system: SystemParameters,
    cable: CableParameters,
    chamber: ChamberParameters,
    points: int = 1200,
    connector: ConnectorParameters | None = None,
) -> CableAnalysisResult:
    frequency_hz = system.frequency_mhz * 1e6
    if cable.length_m is None or cable.length_m <= 0:
        raise ValueError("Cable length is required and must be > 0.")
    cable_length = float(cable.length_m + cable.match_plane_offset_m + cable.chamber_plane_offset_m)
    if cable_length <= 0:
        raise ValueError("Total cable electrical length must be > 0.")

    connector = connector or ConnectorParameters()
    connector_enabled = connector.enabled and connector.model_mode != ConnectorModelMode.DISABLED
    connector_mode = connector.model_mode.value if connector_enabled else ConnectorModelMode.DISABLED.value
    connector_length = float(connector.physical_length_m or 0.0) if connector_enabled else 0.0

    cable_gamma, cable_z0 = _line_model(frequency_hz, cable)
    z_match_measured = system.match_output_impedance.value

    connector_electrical_length_deg: float | None = None
    connector_electrical_length_wavelength: float | None = None
    connector_propagation_delay_ns: float | None = None
    connector_input_power_w: float | None = None
    connector_output_power_w: float | None = None
    connector_loss_w: float | None = None
    connector_interface_index: int | None = None

    if connector_enabled and connector.model_mode == ConnectorModelMode.TRANSMISSION_LINE:
        if connector_length <= 0:
            raise ValueError("Transmission-line connector mode requires connector length > 0.")
        if connector.z0_ohm is None or connector.z0_ohm <= 0:
            raise ValueError("Connector Z0 must be > 0.")
        if connector.velocity_factor is None or connector.velocity_factor <= 0 or connector.velocity_factor > 1:
            raise ValueError("Connector velocity factor must be in (0, 1].")

        connector_z0 = complex(connector.z0_ohm, 0.0)
        connector_gamma = _connector_gamma(frequency_hz, connector)

        lambda_connector = connector.velocity_factor * C0 / frequency_hz
        connector_electrical_length_wavelength = connector_length / lambda_connector
        connector_electrical_length_deg = connector_electrical_length_wavelength * 360.0
        connector_propagation_delay_ns = connector_length / (connector.velocity_factor * C0) * 1e9

        abcd_connector = transmission_line_abcd(connector_z0, connector_gamma, connector_length)
        abcd_cable = transmission_line_abcd(cable_z0, cable_gamma, cable_length)
        abcd_total = cascade_abcd(abcd_connector, abcd_cable)

        if chamber.mode in (ChamberImpedanceMode.USER_ENTERED, ChamberImpedanceMode.MEASURED):
            if chamber.impedance is None:
                raise ValueError("Chamber impedance required for selected chamber mode.")
            z_load = chamber.impedance
            z_in = input_impedance_from_abcd(abcd_total, z_load)
            z_interface = input_impedance(z_load, cable_z0, cable_gamma, cable_length)
        else:
            z_in = z_match_measured
            z_interface = deembed_load_impedance(z_in, connector_z0, connector_gamma, connector_length)
            z_load = deembed_load_impedance(z_interface, cable_z0, cable_gamma, cable_length)

        p_match_input = system.match_input_power_w
        p_match_output, p_match_source = _match_output_power(system, z_in)
        p_match_loss = p_match_input - p_match_output
        connector_input_power_w = p_match_output
        v0, _ = normalize_boundary_from_real_power(z_in, p_match_output)

        conn_points = max(3, int(points * connector_length / (connector_length + cable_length)))
        cable_points = max(3, points - conn_points + 1)
        z_conn_local = np.linspace(0.0, connector_length, conn_points)
        z_cable_local = np.linspace(0.0, cable_length, cable_points)
        v_conn, i_conn = solve_distributed_vi(z_conn_local, connector_length, connector_z0, connector_gamma, z_interface, v0)

        v_interface = complex(v_conn[-1])
        v_cable, i_cable = solve_distributed_vi(
            z_cable_local, cable_length, cable_z0, cable_gamma, z_load, v_interface
        )

        z_m = np.concatenate([z_conn_local, connector_length + z_cable_local[1:]])
        global_position_m = z_m.copy()
        section_name = np.array(
            ["Connector / Transition"] * len(z_conn_local) + ["Flexible Cable"] * (len(z_cable_local) - 1), dtype=object
        )
        section_local_position_m = np.concatenate([z_conn_local, z_cable_local[1:]])
        position_from_flexible_cable_start_m = np.concatenate(
            [np.full(len(z_conn_local), np.nan, dtype=float), z_cable_local[1:]]
        )
        connector_interface_index = len(z_conn_local) - 1

        v_rms = np.concatenate([v_conn, v_cable[1:]])
        i_rms = np.concatenate([i_conn, i_cable[1:]])
        p_real_w = real_power_from_phasors(v_rms, i_rms)

        z_conn_local_ohm = impedance_from_vi(v_conn, i_conn)
        z_cable_local_ohm = impedance_from_vi(v_cable, i_cable)
        z_local_ohm = np.concatenate([z_conn_local_ohm, z_cable_local_ohm[1:]])

        z_conn_norm = normalized_impedance(z_conn_local_ohm, connector_z0)
        z_cable_norm = normalized_impedance(z_cable_local_ohm, cable_z0)
        z_normalized = np.concatenate([z_conn_norm, z_cable_norm[1:]])

        gamma_conn_local = gamma_from_impedance(z_conn_local_ohm, connector_z0)
        gamma_cable_local = gamma_from_impedance(z_cable_local_ohm, cable_z0)
        gamma_local = np.concatenate([gamma_conn_local, gamma_cable_local[1:]])

        p_conn_real = real_power_from_phasors(v_conn, i_conn)
        p_cable_real = real_power_from_phasors(v_cable, i_cable)
        p_conn_total = -np.gradient(p_conn_real, z_conn_local)
        if cable.model_mode == CableModelMode.RLGC:
            p_cable_cond = cable.r_per_m * np.abs(i_cable) ** 2  # type: ignore[operator]
            p_cable_diel = cable.g_per_m * np.abs(v_cable) ** 2  # type: ignore[operator]
            p_cable_total = p_cable_cond + p_cable_diel
        else:
            p_cable_total = -np.gradient(p_cable_real, z_cable_local)
            p_cable_cond = np.zeros_like(p_cable_total)
            p_cable_diel = np.zeros_like(p_cable_total)
        p_cond_w_per_m = np.concatenate([np.zeros_like(p_conn_total), p_cable_cond[1:]])
        p_diel_w_per_m = np.concatenate([np.zeros_like(p_conn_total), p_cable_diel[1:]])
        p_total_loss_w_per_m = np.concatenate([p_conn_total, p_cable_total[1:]])
        p_cumulative_loss_w = cumulative_trapezoid(p_total_loss_w_per_m, z_m)

        connector_output_power_w = float(p_conn_real[-1])
        connector_loss_w = connector_input_power_w - connector_output_power_w
        flexible_cable_input_power_w = connector_output_power_w
        p_cable_output = float(p_cable_real[-1])
        p_chamber = p_cable_output
        flexible_cable_loss_w = flexible_cable_input_power_w - p_cable_output
        p_cable_loss = flexible_cable_loss_w
        p_cable_loss_pct = (p_cable_loss / flexible_cable_input_power_w * 100.0) if flexible_cable_input_power_w > 0 else 0.0
        p_balance_err = p_match_input - p_match_loss - connector_loss_w - flexible_cable_loss_w - p_chamber
        p_balance_err_pct = (p_balance_err / p_match_input * 100.0) if p_match_input > 0 else 0.0

        electrical_length_total_deg = connector_electrical_length_deg + electrical_length_deg(
            frequency_hz, cable.velocity_factor, cable_length
        )
    else:
        line_length = cable_length
        if chamber.mode in (ChamberImpedanceMode.USER_ENTERED, ChamberImpedanceMode.MEASURED):
            if chamber.impedance is None:
                raise ValueError("Chamber impedance required for selected chamber mode.")
            z_load = chamber.impedance
            z_in = input_impedance(z_load, cable_z0, cable_gamma, line_length)
        else:
            z_in = z_match_measured
            z_load = deembed_load_impedance(z_in, cable_z0, cable_gamma, line_length)

        p_match_input = system.match_input_power_w
        p_match_output, p_match_source = _match_output_power(system, z_in)
        p_match_loss = p_match_input - p_match_output
        v0, _ = normalize_boundary_from_real_power(z_in, p_match_output)
        z_m = np.linspace(0.0, line_length, points)
        global_position_m = z_m.copy()
        v_rms, i_rms = solve_distributed_vi(z_m, line_length, cable_z0, cable_gamma, z_load, v0)
        p_real_w = real_power_from_phasors(v_rms, i_rms)
        z_local_ohm = impedance_from_vi(v_rms, i_rms)
        z_normalized = normalized_impedance(z_local_ohm, cable_z0)
        gamma_local = gamma_from_impedance(z_local_ohm, cable_z0)

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
        p_cable_loss = p_match_output - p_cable_output
        p_cable_loss_pct = (p_cable_loss / p_match_output * 100.0) if p_match_output > 0 else 0.0
        p_balance_err = p_match_input - p_match_loss - p_cable_loss - p_chamber
        p_balance_err_pct = (p_balance_err / p_match_input * 100.0) if p_match_input > 0 else 0.0

        if connector_enabled and connector.model_mode == ConnectorModelMode.COORDINATE_OFFSET_ONLY:
            section_name = np.where(
                z_m < connector_length, "Connector / Transition", "Flexible Cable"
            ).astype(object)
            section_local_position_m = np.where(z_m < connector_length, z_m, z_m - connector_length)
            position_from_flexible_cable_start_m = np.where(z_m < connector_length, np.nan, z_m - connector_length)
        else:
            section_name = np.array(["Flexible Cable"] * len(z_m), dtype=object)
            section_local_position_m = z_m.copy()
            position_from_flexible_cable_start_m = z_m.copy()
            connector_length = 0.0
        flexible_cable_input_power_w = p_match_output
        flexible_cable_loss_w = p_cable_loss
        electrical_length_total_deg = electrical_length_deg(frequency_hz, cable.velocity_factor, line_length)

    gl = reflection_coefficient(z_load, cable_z0)
    z_from_vi_match = complex(z_local_ohm[0])
    z_from_vi_end = complex(z_local_ohm[-1])
    endpoint_match_delta = z_from_vi_match - z_match_measured
    endpoint_end_delta = z_from_vi_end - z_load
    endpoint_match_rel_error = abs(endpoint_match_delta) / abs(z_match_measured) if abs(z_match_measured) > 0 else float("inf")
    endpoint_end_rel_error = abs(endpoint_end_delta) / abs(z_load) if abs(z_load) > 0 else float("inf")

    i_abs = np.abs(i_rms)
    z_abs = np.abs(z_local_ohm)
    valid_point_mask = (
        np.isfinite(np.abs(v_rms))
        & np.isfinite(i_abs)
        & np.isfinite(p_real_w)
        & np.isfinite(z_local_ohm.real)
        & np.isfinite(z_local_ohm.imag)
        & np.isfinite(z_abs)
        & (i_abs >= CURRENT_EPSILON_A)
    )
    valid_indices = np.flatnonzero(valid_point_mask)
    min_impedance_index = int(valid_indices[np.argmin(z_abs[valid_indices])]) if len(valid_indices) > 0 else None
    max_current_index = int(valid_indices[np.argmax(i_abs[valid_indices])]) if len(valid_indices) > 0 else None

    (
        min_impedance_z_m,
        min_impedance_z_ohm,
        min_impedance_r_ohm,
        min_impedance_x_ohm,
        min_impedance_magnitude_ohm,
        min_impedance_irms_a,
        min_impedance_vrms_v,
        min_impedance_real_power_w,
    ) = _point_metrics(min_impedance_index, z_m, z_local_ohm, v_rms, i_rms, p_real_w)

    (
        max_current_z_m,
        max_current_z_ohm,
        max_current_r_ohm,
        max_current_x_ohm,
        max_current_magnitude_ohm,
        max_current_irms_a,
        max_current_vrms_v,
        max_current_real_power_w,
    ) = _point_metrics(max_current_index, z_m, z_local_ohm, v_rms, i_rms, p_real_w)

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
    if connector_enabled and connector.model_mode == ConnectorModelMode.COORDINATE_OFFSET_ONLY:
        warnings.append("COORDINATE OFFSET ONLY: CONNECTOR RF BEHAVIOR NOT MODELED.")
    if connector_enabled and connector.model_mode == ConnectorModelMode.TRANSMISSION_LINE:
        warnings.append("Real connector interface parasitics are not modeled unless explicit parameters are provided.")
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
        global_position_m=global_position_m,
        section_name=section_name,
        section_local_position_m=section_local_position_m,
        position_from_flexible_cable_start_m=position_from_flexible_cable_start_m,
        cable_start_global_position_m=connector_length if connector_enabled else 0.0,
        connector_length_m=connector_length if connector_enabled else 0.0,
        connector_enabled=connector_enabled,
        connector_mode=connector_mode,
        connector_model_name=connector.model_name if connector_enabled else "N/A",
        connector_interface_index=connector_interface_index,
        connector_electrical_length_deg=connector_electrical_length_deg,
        connector_electrical_length_wavelength=connector_electrical_length_wavelength,
        connector_propagation_delay_ns=connector_propagation_delay_ns,
        v_rms=v_rms,
        i_rms=i_rms,
        p_real_w=p_real_w,
        p_cond_w_per_m=p_cond_w_per_m,
        p_diel_w_per_m=p_diel_w_per_m,
        p_total_loss_w_per_m=p_total_loss_w_per_m,
        p_cumulative_loss_w=p_cumulative_loss_w,
        z0_reference=cable_z0,
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
        electrical_length_deg=electrical_length_total_deg,
        match_input_power_w=p_match_input,
        match_output_power_w=p_match_output,
        match_output_power_source_label=p_match_source,
        match_loss_w=p_match_loss,
        connector_input_power_w=connector_input_power_w,
        connector_output_power_w=connector_output_power_w,
        connector_loss_w=connector_loss_w,
        cable_input_power_w=flexible_cable_input_power_w,
        flexible_cable_input_power_w=flexible_cable_input_power_w,
        cable_output_power_w=p_cable_output,
        chamber_power_w=p_chamber,
        cable_loss_w=p_cable_loss,
        flexible_cable_total_loss_w=flexible_cable_loss_w,
        cable_end_delivered_power_w=p_chamber,
        cable_loss_percent=p_cable_loss_pct,
        power_balance_error_w=p_balance_err,
        power_balance_error_percent=p_balance_err_pct,
        min_impedance_index=min_impedance_index,
        min_impedance_z_m=min_impedance_z_m,
        min_impedance_z_ohm=min_impedance_z_ohm,
        min_impedance_r_ohm=min_impedance_r_ohm,
        min_impedance_x_ohm=min_impedance_x_ohm,
        min_impedance_magnitude_ohm=min_impedance_magnitude_ohm,
        min_impedance_irms_a=min_impedance_irms_a,
        min_impedance_vrms_v=min_impedance_vrms_v,
        min_impedance_real_power_w=min_impedance_real_power_w,
        max_current_index=max_current_index,
        max_current_z_m=max_current_z_m,
        max_current_z_ohm=max_current_z_ohm,
        max_current_r_ohm=max_current_r_ohm,
        max_current_x_ohm=max_current_x_ohm,
        max_current_magnitude_ohm=max_current_magnitude_ohm,
        max_current_irms_a=max_current_irms_a,
        max_current_vrms_v=max_current_vrms_v,
        max_current_real_power_w=max_current_real_power_w,
        warnings=warnings,
    )
