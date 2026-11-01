from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from models.cable_parameters import CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import MatchOutputPowerSource, MatcherModelMode, PowerNormalizationPlane, SystemParameters
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
    solve_distributed_vi_components,
    to_db_per_m,
    vswr,
    z0_from_rlgc,
)
from rf.two_port import SParameters
from rf.two_port import (
    largest_singular_value_of_s,
    matcher_input_impedance,
    matcher_output_impedance,
    s_to_z,
    solve_loaded_two_port,
)


CURRENT_EPSILON_A = 1e-12
COMPLEX_Z0_IMAG_RATIO_MAX = 1e-3
TRAVELING_WAVE_POWER_ABS_TOL_W = 1e-3
TRAVELING_WAVE_POWER_REL_TOL = 1e-3


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
    v_forward: np.ndarray
    v_reflected: np.ndarray
    i_forward: np.ndarray
    i_reflected: np.ndarray
    vpp: np.ndarray
    p_forward_w: np.ndarray
    p_reflected_w: np.ndarray
    p_real_w: np.ndarray
    forward_power_w: np.ndarray
    reflected_power_w: np.ndarray
    net_traveling_wave_power_w: np.ndarray
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
    wavelength_m: float
    velocity_m_per_s: float
    alpha_np_per_m: float
    beta_rad_per_m: float
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
    matcher_source_impedance_zs: complex
    matcher_input_impedance_zin: complex | None
    matcher_output_impedance_zout: complex | None
    z22_open_circuit_parameter_ohm: complex | None
    z11_ohm: complex | None
    z12_ohm: complex | None
    z21_ohm: complex | None
    load_impedance_zl: complex | None
    conjugate_zout_target: complex | None
    conjugate_match_delta_z: complex | None
    conjugate_match_delta_r_ohm: float | None
    conjugate_match_delta_x_ohm: float | None
    conjugate_match_error_percent: float | None
    active_load_mode_label: str
    passivity_largest_singular_value: float | None
    passivity_check_label: str | None
    loaded_model_v2_rms_v: float | None
    loaded_model_v2_vpp_v: float | None
    measured_vpp_residual_v: float | None
    power_normalization_plane: str
    power_normalization_power_w: float
    traveling_wave_reference_plane: str
    traveling_wave_available: bool
    traveling_wave_warning: str | None
    match_plane_forward_power_w: float | None
    match_plane_reflected_power_w: float | None
    match_plane_net_power_w: float | None
    match_plane_vi_power_w: float | None
    match_plane_power_residual_w: float | None
    match_plane_power_residual_percent: float | None
    max_traveling_wave_vi_residual_w: float | None
    direct_load_reference_irms_a: float | None
    direct_load_reference_vrms_v: float | None
    direct_load_reference_vpp_v: float | None
    measured_vpp_v: float | None
    maximum_voltage_rms_v: float
    maximum_voltage_vpp_v: float
    maximum_voltage_position_from_match_m: float
    maximum_voltage_position_from_load_m: float
    maximum_voltage_section_name: str
    minimum_voltage_rms_v: float
    minimum_voltage_vpp_v: float
    minimum_voltage_position_from_match_m: float
    minimum_voltage_position_from_load_m: float
    minimum_voltage_section_name: str
    vpp_residual_v: float | None
    vpp_residual_percent: float | None
    standing_wave_voltage_enhancement: float | None
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


def _match_output_power(system: SystemParameters, z_in: complex) -> tuple[float, str, object | None]:
    if system.match_output_power_source == MatchOutputPowerSource.USER:
        if system.match_output_power_w is None:
            return system.match_input_power_w, "IDEAL MATCHER - NO MATCH LOSS ASSUMED", None
        return system.match_output_power_w, MatchOutputPowerSource.USER.value, None
    if system.match_output_power_source == MatchOutputPowerSource.CALCULATED:
        if system.matcher_model_mode == MatcherModelMode.USER_S:
            s = SParameters(
                s11=system.s11,
                s12=system.s12,
                s21=system.s21,
                s22=system.s22,
                zref_ohm=system.sparam_ref_impedance_ohm,
            )
            solved = solve_loaded_two_port(
                s=s,
                z_source=system.source_impedance,
                z_load=z_in,
                p_match_input_w=system.match_input_power_w,
            )
            return solved.p_match_output_w, "CALCULATED FROM MATCHER MODEL (S-PARAMETERS, LOADED TWO-PORT)", solved
        eta = min(max(system.match_efficiency_percent / 100.0, 0.0), 1.0)
        return system.match_input_power_w * eta, "CALCULATED FROM MATCHER MODEL (EFFICIENCY)", None
    return system.match_input_power_w, "IDEAL MATCHER - NO MATCH LOSS ASSUMED", None


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
    active_load_mode_label = chamber.mode.value

    z11_ohm: complex | None = None
    z12_ohm: complex | None = None
    z21_ohm: complex | None = None
    z22_open_circuit_parameter_ohm: complex | None = None
    matcher_output_impedance_zout: complex | None = None
    matcher_input_impedance_zin: complex | None = None
    passivity_largest_singular_value: float | None = None
    passivity_check_label: str | None = None
    loaded_matcher_solution = None
    if system.matcher_model_mode == MatcherModelMode.USER_S:
        sparams = SParameters(
            s11=system.s11,
            s12=system.s12,
            s21=system.s21,
            s22=system.s22,
            zref_ohm=system.sparam_ref_impedance_ohm,
        )
        z_matrix = s_to_z(sparams)
        z11_ohm = complex(z_matrix[0, 0])
        z12_ohm = complex(z_matrix[0, 1])
        z21_ohm = complex(z_matrix[1, 0])
        z22_open_circuit_parameter_ohm = complex(z_matrix[1, 1])
        matcher_output_impedance_zout = matcher_output_impedance(z_matrix, system.source_impedance)
        passivity_largest_singular_value = largest_singular_value_of_s(sparams)
        passivity_check_label = (
            "PASS"
            if passivity_largest_singular_value <= 1.0 + 1e-6
            else "WARNING"
        )

    connector_electrical_length_deg: float | None = None
    connector_electrical_length_wavelength: float | None = None
    connector_propagation_delay_ns: float | None = None
    connector_input_power_w: float | None = None
    connector_output_power_w: float | None = None
    connector_loss_w: float | None = None
    connector_interface_index: int | None = None

    power_normalization_plane = system.power_normalization_plane

    z0_reference_profile: np.ndarray

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

        if chamber.mode == ChamberImpedanceMode.DERIVED_FROM_DOWNSTREAM_PATH:
            if chamber.impedance is None:
                z_in = z_match_measured
                z_interface = deembed_load_impedance(z_in, connector_z0, connector_gamma, connector_length)
                z_load = deembed_load_impedance(z_interface, cable_z0, cable_gamma, cable_length)
            else:
                z_load = chamber.impedance
                z_in = input_impedance_from_abcd(abcd_total, z_load)
                z_interface = input_impedance(z_load, cable_z0, cable_gamma, cable_length)
        elif chamber.mode == ChamberImpedanceMode.MEASURED_OR_USER_LOAD:
            z_in = z_match_measured
            z_interface = deembed_load_impedance(z_in, connector_z0, connector_gamma, connector_length)
            z_load = deembed_load_impedance(z_interface, cable_z0, cable_gamma, cable_length)
        elif chamber.mode == ChamberImpedanceMode.CONJUGATE_MATCH_TO_ZOUT:
            if matcher_output_impedance_zout is None:
                raise ValueError("Zout is required before conjugate-match load mode can be applied.")
            z_in = np.conjugate(matcher_output_impedance_zout)
            z_interface = deembed_load_impedance(z_in, connector_z0, connector_gamma, connector_length)
            z_load = deembed_load_impedance(z_interface, cable_z0, cable_gamma, cable_length)
        else:
            raise ValueError("Load mode is FUTURE / EXPERIMENTAL and not implemented.")

        p_match_input = system.match_input_power_w
        p_match_output, p_match_source, loaded_matcher_solution = _match_output_power(system, z_in)
        p_match_loss = p_match_input - p_match_output
        v0, _ = normalize_boundary_from_real_power(z_in, max(p_match_output, 0.0))

        conn_points = max(3, int(points * connector_length / (connector_length + cable_length)))
        cable_points = max(3, points - conn_points + 1)
        z_conn_local = np.linspace(0.0, connector_length, conn_points)
        z_cable_local = np.linspace(0.0, cable_length, cable_points)
        vf_conn, vr_conn, if_conn, ir_conn = solve_distributed_vi_components(
            z_conn_local, connector_length, connector_z0, connector_gamma, z_interface, v0
        )
        v_conn = vf_conn + vr_conn
        i_conn = if_conn + ir_conn
        v_interface = complex(v_conn[-1])
        vf_cable, vr_cable, if_cable, ir_cable = solve_distributed_vi_components(
            z_cable_local, cable_length, cable_z0, cable_gamma, z_load, v_interface
        )
        v_cable = vf_cable + vr_cable
        i_cable = if_cable + ir_cable

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

        v_forward = np.concatenate([vf_conn, vf_cable[1:]])
        v_reflected = np.concatenate([vr_conn, vr_cable[1:]])
        i_forward = np.concatenate([if_conn, if_cable[1:]])
        i_reflected = np.concatenate([ir_conn, ir_cable[1:]])
        z0_reference_profile = np.concatenate(
            [np.full(len(vf_conn), connector_z0, dtype=complex), np.full(len(vf_cable) - 1, cable_z0, dtype=complex)]
        )
        v_rms = v_forward + v_reflected
        i_rms = i_forward + i_reflected

        p_real_unscaled = real_power_from_phasors(v_rms, i_rms)
        target_power_w = max(p_match_output, 0.0)
        ref_idx = 0 if power_normalization_plane == PowerNormalizationPlane.MATCH_OUTPUT_CABLE_INPUT else -1
        ref_power = float(p_real_unscaled[ref_idx])
        if target_power_w <= 0.0:
            scale = 0.0
        elif ref_power <= 0.0:
            raise ValueError("Selected power normalization plane has non-positive real power in the unscaled solution.")
        else:
            scale = math.sqrt(target_power_w / ref_power)
        v_forward *= scale
        v_reflected *= scale
        i_forward *= scale
        i_reflected *= scale
        v_rms = v_forward + v_reflected
        i_rms = i_forward + i_reflected
        v_conn *= scale
        i_conn *= scale
        v_cable *= scale
        i_cable *= scale

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

        connector_input_power_w = float(p_conn_real[0])
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
        if chamber.mode == ChamberImpedanceMode.DERIVED_FROM_DOWNSTREAM_PATH:
            if chamber.impedance is None:
                z_in = z_match_measured
                z_load = deembed_load_impedance(z_in, cable_z0, cable_gamma, line_length)
            else:
                z_load = chamber.impedance
                z_in = input_impedance(z_load, cable_z0, cable_gamma, line_length)
        elif chamber.mode == ChamberImpedanceMode.MEASURED_OR_USER_LOAD:
            z_in = z_match_measured
            z_load = deembed_load_impedance(z_in, cable_z0, cable_gamma, line_length)
        elif chamber.mode == ChamberImpedanceMode.CONJUGATE_MATCH_TO_ZOUT:
            if matcher_output_impedance_zout is None:
                raise ValueError("Zout is required before conjugate-match load mode can be applied.")
            z_in = np.conjugate(matcher_output_impedance_zout)
            z_load = deembed_load_impedance(z_in, cable_z0, cable_gamma, line_length)
        else:
            raise ValueError("Load mode is FUTURE / EXPERIMENTAL and not implemented.")

        p_match_input = system.match_input_power_w
        p_match_output, p_match_source, loaded_matcher_solution = _match_output_power(system, z_in)
        p_match_loss = p_match_input - p_match_output
        v0, _ = normalize_boundary_from_real_power(z_in, max(p_match_output, 0.0))
        z_m = np.linspace(0.0, line_length, points)
        global_position_m = z_m.copy()
        v_forward, v_reflected, i_forward, i_reflected = solve_distributed_vi_components(
            z_m, line_length, cable_z0, cable_gamma, z_load, v0
        )
        z0_reference_profile = np.full(len(z_m), cable_z0, dtype=complex)
        v_rms = v_forward + v_reflected
        i_rms = i_forward + i_reflected
        p_real_unscaled = real_power_from_phasors(v_rms, i_rms)
        target_power_w = max(p_match_output, 0.0)
        ref_idx = 0 if power_normalization_plane == PowerNormalizationPlane.MATCH_OUTPUT_CABLE_INPUT else -1
        ref_power = float(p_real_unscaled[ref_idx])
        if target_power_w <= 0.0:
            scale = 0.0
        elif ref_power <= 0.0:
            raise ValueError("Selected power normalization plane has non-positive real power in the unscaled solution.")
        else:
            scale = math.sqrt(target_power_w / ref_power)
        v_forward *= scale
        v_reflected *= scale
        i_forward *= scale
        i_reflected *= scale
        v_rms = v_forward + v_reflected
        i_rms = i_forward + i_reflected
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
        flexible_cable_input_power_w = float(p_real_w[0])
        p_cable_output = float(p_real_w[-1])
        p_chamber = p_cable_output
        p_cable_loss = flexible_cable_input_power_w - p_cable_output
        p_cable_loss_pct = (p_cable_loss / flexible_cable_input_power_w * 100.0) if flexible_cable_input_power_w > 0 else 0.0
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
        flexible_cable_loss_w = p_cable_loss
        electrical_length_total_deg = electrical_length_deg(frequency_hz, cable.velocity_factor, line_length)

    if system.matcher_model_mode == MatcherModelMode.USER_S and z11_ohm is not None:
        z_matrix_for_loaded = np.array(
            [
                [z11_ohm, z12_ohm if z12_ohm is not None else 0.0 + 0.0j],
                [z21_ohm if z21_ohm is not None else 0.0 + 0.0j, z22_open_circuit_parameter_ohm if z22_open_circuit_parameter_ohm is not None else 0.0 + 0.0j],
            ],
            dtype=complex,
        )
        matcher_input_impedance_zin = matcher_input_impedance(z_matrix_for_loaded, z_in)
    if loaded_matcher_solution is not None:
        matcher_input_impedance_zin = loaded_matcher_solution.zin_ohm

    conjugate_zout_target = np.conjugate(matcher_output_impedance_zout) if matcher_output_impedance_zout is not None else None
    conjugate_match_delta_z = (z_in - conjugate_zout_target) if conjugate_zout_target is not None else None
    conjugate_match_delta_r_ohm = float(conjugate_match_delta_z.real) if conjugate_match_delta_z is not None else None
    conjugate_match_delta_x_ohm = float(conjugate_match_delta_z.imag) if conjugate_match_delta_z is not None else None
    conjugate_match_error_percent = None
    if conjugate_match_delta_z is not None and abs(conjugate_zout_target) > 1e-12:
        conjugate_match_error_percent = float(100.0 * abs(conjugate_match_delta_z) / abs(conjugate_zout_target))

    gl = reflection_coefficient(z_load, cable_z0)
    z_from_vi_match = complex(z_local_ohm[0])
    z_from_vi_end = complex(z_local_ohm[-1])
    endpoint_match_delta = z_from_vi_match - z_in
    endpoint_end_delta = z_from_vi_end - z_load
    endpoint_match_rel_error = abs(endpoint_match_delta) / abs(z_in) if abs(z_in) > 0 else float("inf")
    endpoint_end_rel_error = abs(endpoint_end_delta) / abs(z_load) if abs(z_load) > 0 else float("inf")

    z0_real = np.real(z0_reference_profile)
    z0_imag = np.imag(z0_reference_profile)
    z0_real_abs = np.maximum(np.abs(z0_real), 1e-12)
    z0_imag_ratio = np.abs(z0_imag) / z0_real_abs
    has_significant_complex_z0 = bool(np.any(z0_imag_ratio > COMPLEX_Z0_IMAG_RATIO_MAX))
    has_nonpositive_real_z0 = bool(np.any(z0_real <= 0.0))
    traveling_wave_available = not has_significant_complex_z0 and not has_nonpositive_real_z0
    traveling_wave_warning: str | None = None
    traveling_wave_reference_plane = "MATCH OUTPUT / CABLE INPUT (z = 0)"

    if traveling_wave_available:
        forward_power_w = (np.abs(v_forward) ** 2) / z0_real
        reflected_power_w = (np.abs(v_reflected) ** 2) / z0_real
        net_traveling_wave_power_w = forward_power_w - reflected_power_w
        match_plane_forward_power_w = float(forward_power_w[0])
        match_plane_reflected_power_w = float(reflected_power_w[0])
        match_plane_net_power_w = float(net_traveling_wave_power_w[0])
        match_plane_vi_power_w = float(p_real_w[0])
        match_plane_power_residual_w = float(match_plane_net_power_w - p_match_output)
        match_plane_power_residual_percent = (
            float(100.0 * match_plane_power_residual_w / p_match_output) if abs(p_match_output) > 1e-12 else 0.0
        )
        max_traveling_wave_vi_residual_w = float(np.max(np.abs(net_traveling_wave_power_w - p_real_w)))
        p_forward_w = forward_power_w
        p_reflected_w = reflected_power_w
    else:
        forward_power_w = np.full_like(p_real_w, np.nan, dtype=float)
        reflected_power_w = np.full_like(p_real_w, np.nan, dtype=float)
        net_traveling_wave_power_w = np.full_like(p_real_w, np.nan, dtype=float)
        p_forward_w = forward_power_w
        p_reflected_w = reflected_power_w
        match_plane_forward_power_w = None
        match_plane_reflected_power_w = None
        match_plane_net_power_w = None
        match_plane_vi_power_w = None
        match_plane_power_residual_w = None
        match_plane_power_residual_percent = None
        max_traveling_wave_vi_residual_w = None
        if has_nonpositive_real_z0:
            traveling_wave_warning = (
                "TRAVELING-WAVE POWER DECOMPOSITION NOT AVAILABLE FOR NON-POSITIVE Re(Z0) IN CURRENT CONVENTION."
            )
        else:
            traveling_wave_warning = (
                "TRAVELING-WAVE POWER DECOMPOSITION NOT AVAILABLE FOR CURRENT COMPLEX-Z0 CONVENTION."
            )

    vpp = 2.0 * np.sqrt(2.0) * np.abs(v_rms)

    rl = float(z_load.real)
    if p_match_output > 0.0 and rl > 0.0:
        direct_i_rms = float(math.sqrt(p_match_output / rl))
        direct_v_rms = float(direct_i_rms * abs(z_load))
        direct_vpp = float(2.0 * math.sqrt(2.0) * direct_v_rms)
    else:
        direct_i_rms = None
        direct_v_rms = None
        direct_vpp = None

    vmax_idx = int(np.argmax(np.abs(v_rms)))
    vmin_idx = int(np.argmin(np.abs(v_rms)))
    z_total = float(z_m[-1])
    max_vrms = float(np.abs(v_rms[vmax_idx]))
    max_vpp = float(vpp[vmax_idx])
    min_vrms = float(np.abs(v_rms[vmin_idx]))
    min_vpp = float(vpp[vmin_idx])
    measured_vpp = system.measured_vpp_v
    vpp_residual_v = (max_vpp - measured_vpp) if measured_vpp is not None else None
    vpp_residual_percent = (vpp_residual_v / measured_vpp * 100.0) if (measured_vpp is not None and measured_vpp != 0) else None
    standing_wave_voltage_enhancement = (max_vpp / direct_vpp) if (direct_vpp is not None and direct_vpp > 0) else None

    velocity_m_per_s = float(cable.velocity_factor * C0)
    wavelength_m = float(velocity_m_per_s / frequency_hz)
    alpha_np_per_m = float(np.real(cable_gamma))
    beta_rad_per_m = float(np.imag(cable_gamma))

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
    if loaded_matcher_solution is not None and loaded_matcher_solution.eta_match > 1.0 + 1e-6:
        warnings.append(
            "WARNING: Calculated matcher efficiency exceeds 100%. Check S-parameter passivity, current convention and load/reference planes."
        )
    if passivity_largest_singular_value is not None and passivity_largest_singular_value > 1.0 + 1e-6:
        warnings.append("PASSIVITY CHECK: WARNING (largest singular value of S exceeds 1).")
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
    if traveling_wave_warning is not None:
        warnings.append(traveling_wave_warning)
    if traveling_wave_available and match_plane_power_residual_w is not None:
        allowed_residual_w = max(TRAVELING_WAVE_POWER_ABS_TOL_W, TRAVELING_WAVE_POWER_REL_TOL * max(abs(p_match_output), 1.0))
        if abs(match_plane_power_residual_w) > allowed_residual_w:
            warnings.append("TRAVELING-WAVE POWER CONSISTENCY WARNING")

    power_normalization_power_w = float(p_real_w[0]) if power_normalization_plane == PowerNormalizationPlane.MATCH_OUTPUT_CABLE_INPUT else float(p_real_w[-1])

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
        v_forward=v_forward,
        v_reflected=v_reflected,
        i_forward=i_forward,
        i_reflected=i_reflected,
        vpp=vpp,
        p_forward_w=p_forward_w,
        p_reflected_w=p_reflected_w,
        p_real_w=p_real_w,
        forward_power_w=forward_power_w,
        reflected_power_w=reflected_power_w,
        net_traveling_wave_power_w=net_traveling_wave_power_w,
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
        wavelength_m=wavelength_m,
        velocity_m_per_s=velocity_m_per_s,
        alpha_np_per_m=alpha_np_per_m,
        beta_rad_per_m=beta_rad_per_m,
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
        matcher_source_impedance_zs=system.source_impedance,
        matcher_input_impedance_zin=matcher_input_impedance_zin,
        matcher_output_impedance_zout=matcher_output_impedance_zout,
        z22_open_circuit_parameter_ohm=z22_open_circuit_parameter_ohm,
        z11_ohm=z11_ohm,
        z12_ohm=z12_ohm,
        z21_ohm=z21_ohm,
        load_impedance_zl=z_load,
        conjugate_zout_target=conjugate_zout_target,
        conjugate_match_delta_z=conjugate_match_delta_z,
        conjugate_match_delta_r_ohm=conjugate_match_delta_r_ohm,
        conjugate_match_delta_x_ohm=conjugate_match_delta_x_ohm,
        conjugate_match_error_percent=conjugate_match_error_percent,
        active_load_mode_label=active_load_mode_label,
        passivity_largest_singular_value=passivity_largest_singular_value,
        passivity_check_label=passivity_check_label,
        loaded_model_v2_rms_v=float(abs(loaded_matcher_solution.v2_rms_v)) if loaded_matcher_solution is not None else None,
        loaded_model_v2_vpp_v=float(2.0 * np.sqrt(2.0) * abs(loaded_matcher_solution.v2_rms_v))
        if loaded_matcher_solution is not None
        else None,
        measured_vpp_residual_v=(
            float(system.measured_vpp_v - (2.0 * np.sqrt(2.0) * abs(loaded_matcher_solution.v2_rms_v)))
            if (loaded_matcher_solution is not None and system.measured_vpp_v is not None)
            else None
        ),
        power_normalization_plane=power_normalization_plane.value,
        power_normalization_power_w=power_normalization_power_w,
        traveling_wave_reference_plane=traveling_wave_reference_plane,
        traveling_wave_available=traveling_wave_available,
        traveling_wave_warning=traveling_wave_warning,
        match_plane_forward_power_w=match_plane_forward_power_w,
        match_plane_reflected_power_w=match_plane_reflected_power_w,
        match_plane_net_power_w=match_plane_net_power_w,
        match_plane_vi_power_w=match_plane_vi_power_w,
        match_plane_power_residual_w=match_plane_power_residual_w,
        match_plane_power_residual_percent=match_plane_power_residual_percent,
        max_traveling_wave_vi_residual_w=max_traveling_wave_vi_residual_w,
        direct_load_reference_irms_a=direct_i_rms,
        direct_load_reference_vrms_v=direct_v_rms,
        direct_load_reference_vpp_v=direct_vpp,
        measured_vpp_v=measured_vpp,
        maximum_voltage_rms_v=max_vrms,
        maximum_voltage_vpp_v=max_vpp,
        maximum_voltage_position_from_match_m=float(z_m[vmax_idx]),
        maximum_voltage_position_from_load_m=z_total - float(z_m[vmax_idx]),
        maximum_voltage_section_name=str(section_name[vmax_idx]),
        minimum_voltage_rms_v=min_vrms,
        minimum_voltage_vpp_v=min_vpp,
        minimum_voltage_position_from_match_m=float(z_m[vmin_idx]),
        minimum_voltage_position_from_load_m=z_total - float(z_m[vmin_idx]),
        minimum_voltage_section_name=str(section_name[vmin_idx]),
        vpp_residual_v=vpp_residual_v,
        vpp_residual_percent=vpp_residual_percent,
        standing_wave_voltage_enhancement=standing_wave_voltage_enhancement,
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
