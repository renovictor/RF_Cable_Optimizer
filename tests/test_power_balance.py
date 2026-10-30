from __future__ import annotations

import unittest

import numpy as np

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters


class PowerBalanceTests(unittest.TestCase):
    def test_engineering_points_use_valid_indices(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1300.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(3.0, 19.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.03,
            velocity_factor=0.76,
            attenuation_value=0.6,
            attenuation_unit=AttenuationUnit.DB_PER_100FT,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertIsNotNone(result.min_impedance_index)
        self.assertIsNotNone(result.max_current_index)
        assert result.min_impedance_index is not None
        assert result.max_current_index is not None
        valid = (
            np.isfinite(np.abs(result.v_rms))
            & np.isfinite(np.abs(result.i_rms))
            & np.isfinite(result.p_real_w)
            & np.isfinite(result.z_local_ohm.real)
            & np.isfinite(result.z_local_ohm.imag)
            & np.isfinite(np.abs(result.z_local_ohm))
            & (np.abs(result.i_rms) >= 1e-12)
        )
        self.assertAlmostEqual(
            result.min_impedance_magnitude_ohm,
            float(np.min(np.abs(result.z_local_ohm[valid]))),
            places=6,
        )
        self.assertAlmostEqual(result.max_current_irms_a, float(np.max(np.abs(result.i_rms[valid]))), places=6)

    def test_engineering_points_are_not_available_without_valid_current(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=0.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(50.0, 0.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertIsNone(result.min_impedance_index)
        self.assertIsNone(result.max_current_index)
        self.assertIsNone(result.min_impedance_magnitude_ohm)
        self.assertIsNone(result.max_current_irms_a)

    def test_lossless_mismatch_has_zero_cable_dissipation(self) -> None:
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
        self.assertAlmostEqual(result.cable_loss_w, 0.0, places=4)
        self.assertGreater(np.max(np.abs(result.v_rms)), 0.0)

    def test_lossy_matched_line_has_positive_loss(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1000.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(50.0, 0.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.03,
            velocity_factor=0.76,
            attenuation_value=0.2,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertGreater(result.cable_loss_w, 0.0)
        self.assertLess(result.cable_output_power_w, result.cable_input_power_w)

    def test_power_conservation_chain(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1300.0,
            match_output_power_source=MatchOutputPowerSource.CALCULATED,
            match_efficiency_percent=95.0,
            match_output_impedance=ComplexImpedance(3.0, 19.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.03,
            velocity_factor=0.76,
            attenuation_value=0.6,
            attenuation_unit=AttenuationUnit.DB_PER_100FT,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertAlmostEqual(
            result.match_input_power_w,
            result.match_loss_w + result.cable_loss_w + result.chamber_power_w,
            places=6,
        )

    def test_user_match_output_power_derives_efficiency(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1000.0,
            match_output_power_source=MatchOutputPowerSource.USER,
            match_output_power_w=900.0,
            match_output_impedance=ComplexImpedance(50.0, 0.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertAlmostEqual(result.match_output_power_w / result.match_input_power_w, 0.9, places=6)

    def test_ideal_mode_announces_assumed_label(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=500.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(50.0, 0.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber)
        self.assertEqual(result.match_output_power_w, result.match_input_power_w)
        self.assertIn("IDEAL MATCHER - NO MATCH LOSS ASSUMED", result.match_output_power_source_label)


if __name__ == "__main__":
    unittest.main()
