from __future__ import annotations

import unittest

import numpy as np

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.system_parameters import (
    ComplexImpedance,
    MatchOutputPowerSource,
    PowerNormalizationPlane,
    SystemParameters,
)


def _regression_inputs(power_plane: PowerNormalizationPlane = PowerNormalizationPlane.MATCH_OUTPUT_CABLE_INPUT) -> tuple[
    SystemParameters, CableParameters, ChamberParameters
]:
    system = SystemParameters(
        frequency_mhz=60.0,
        match_input_power_w=1300.0,
        match_efficiency_percent=70.0,
        match_output_power_source=MatchOutputPowerSource.CALCULATED,
        power_normalization_plane=power_plane,
        match_output_impedance=ComplexImpedance(3.0, 19.0),
        measured_vpp_v=2751.0,
    )
    cable = CableParameters(
        z0_ohm=50.0,
        length_m=1.03,
        velocity_factor=0.76,
        attenuation_value=0.6,
        attenuation_unit=AttenuationUnit.DB_PER_100FT,
        model_mode=CableModelMode.DATASHEET,
    )
    chamber = ChamberParameters(
        mode=ChamberImpedanceMode.DERIVED_FROM_DOWNSTREAM_PATH,
        impedance_real_ohm=2.6,
        impedance_imag_ohm=19.7,
    )
    return system, cable, chamber


class StandingWaveDecompositionTests(unittest.TestCase):
    def test_direct_load_reference_regression(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        assert result.direct_load_reference_irms_a is not None
        assert result.direct_load_reference_vrms_v is not None
        assert result.direct_load_reference_vpp_v is not None
        self.assertAlmostEqual(result.direct_load_reference_irms_a, 18.708, places=2)
        self.assertAlmostEqual(result.direct_load_reference_vrms_v, 371.75, places=1)
        self.assertAlmostEqual(result.direct_load_reference_vpp_v, 1051.5, places=1)

    def test_reflection_coefficient_and_vswr_regression(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        self.assertAlmostEqual(abs(result.gamma_l), 0.914, places=3)
        self.assertAlmostEqual(result.vswr, 22.22, places=1)

    def test_lossy_distributed_regression_case(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        self.assertAlmostEqual(result.maximum_voltage_rms_v, 980.47, places=0)
        self.assertAlmostEqual(result.maximum_voltage_vpp_v, 2773.0, places=0)
        self.assertAlmostEqual(result.maximum_voltage_position_from_match_m, 0.308, places=2)
        self.assertAlmostEqual(result.cable_end_delivered_power_w, 865.0, places=0)
        self.assertAlmostEqual(result.flexible_cable_total_loss_w, 45.0, places=0)

    def test_model_to_measurement_residual(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        assert result.vpp_residual_v is not None
        assert result.vpp_residual_percent is not None
        self.assertAlmostEqual(result.vpp_residual_v, 22.0, places=0)
        self.assertAlmostEqual(result.vpp_residual_percent, 0.8, places=1)

    def test_power_normalization_plane_modes_are_distinct(self) -> None:
        system_a, cable_a, chamber_a = _regression_inputs(PowerNormalizationPlane.MATCH_OUTPUT_CABLE_INPUT)
        result_a = analyze_cable(system_a, cable_a, chamber_a)
        self.assertAlmostEqual(float(result_a.p_real_w[0]), 910.0, places=1)

        system_b, cable_b, chamber_b = _regression_inputs(PowerNormalizationPlane.CABLE_END_LOAD_ABSORBED)
        result_b = analyze_cable(system_b, cable_b, chamber_b)
        self.assertAlmostEqual(float(result_b.p_real_w[-1]), 910.0, places=1)
        self.assertNotAlmostEqual(result_a.maximum_voltage_vpp_v, result_b.maximum_voltage_vpp_v, places=3)

    def test_complex_wave_addition_matches_total_phasors(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        np.testing.assert_allclose(result.v_rms, result.v_forward + result.v_reflected, atol=1e-10)
        np.testing.assert_allclose(result.i_rms, result.i_forward + result.i_reflected, atol=1e-10)
        scalar_sum_error = np.abs(result.v_rms) - (np.abs(result.v_forward) + np.abs(result.v_reflected))
        self.assertGreater(float(np.max(np.abs(scalar_sum_error))), 1e-6)

    def test_match_plane_traveling_wave_power_matches_match_output_power(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        assert result.match_plane_forward_power_w is not None
        assert result.match_plane_reflected_power_w is not None
        assert result.match_plane_net_power_w is not None
        self.assertAlmostEqual(result.match_plane_net_power_w, 910.0, places=1)
        self.assertAlmostEqual(
            result.match_plane_forward_power_w - result.match_plane_reflected_power_w,
            result.match_plane_net_power_w,
            places=6,
        )
        self.assertAlmostEqual(result.match_plane_net_power_w, result.match_output_power_w, places=1)

    def test_net_traveling_wave_power_matches_vi_power_profile(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        self.assertTrue(result.traveling_wave_available)
        np.testing.assert_allclose(result.net_traveling_wave_power_w, result.p_real_w, rtol=1e-6, atol=1e-6)

    def test_lossless_line_has_constant_forward_reflected_and_net_power(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1000.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(3.0, 19.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.03,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertAlmostEqual(float(np.ptp(result.forward_power_w)), 0.0, places=5)
        self.assertAlmostEqual(float(np.ptp(result.reflected_power_w)), 0.0, places=5)
        self.assertAlmostEqual(float(np.ptp(result.net_traveling_wave_power_w)), 0.0, places=5)
        np.testing.assert_allclose(result.net_traveling_wave_power_w, result.forward_power_w - result.reflected_power_w, atol=1e-8)

    def test_lossy_line_shows_position_dependent_traveling_wave_powers(self) -> None:
        system, cable, chamber = _regression_inputs()
        result = analyze_cable(system, cable, chamber)
        self.assertGreater(float(np.ptp(result.forward_power_w)), 1e-6)
        self.assertGreater(float(np.ptp(result.reflected_power_w)), 1e-6)
        self.assertGreater(float(result.net_traveling_wave_power_w[0]), float(result.net_traveling_wave_power_w[-1]))

    def test_complex_z0_traveling_wave_decomposition_is_conservatively_disabled(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1000.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(3.0, 19.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.03,
            velocity_factor=0.76,
            model_mode=CableModelMode.RLGC,
            r_per_m=0.8,
            l_per_m=2.5e-7,
            g_per_m=2.0e-4,
            c_per_m=1.0e-10,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertFalse(result.traveling_wave_available)
        self.assertIn("NOT AVAILABLE", result.traveling_wave_warning or "")
        self.assertTrue(np.isnan(result.forward_power_w).all())


if __name__ == "__main__":
    unittest.main()
