from __future__ import annotations

import unittest

import numpy as np

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters


class SmithDistributedTests(unittest.TestCase):
    def test_lossless_matched_line_collapses_to_center(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=800.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(50.0, 0.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.2,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber, points=600)
        np.testing.assert_allclose(result.z_local_ohm, np.full_like(result.z_local_ohm, 50 + 0j), atol=1e-8)
        np.testing.assert_allclose(result.gamma_local, np.zeros_like(result.gamma_local), atol=1e-10)

    def test_lossless_mismatched_line_constant_gamma_magnitude(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=700.0,
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
        result = analyze_cable(system, cable, chamber, points=1200)
        gamma_mag = np.abs(result.gamma_local)
        self.assertLess(float(np.max(gamma_mag) - np.min(gamma_mag)), 1e-6)
        self.assertGreater(float(np.ptp(np.unwrap(np.angle(result.gamma_local)))), 0.1)
        self.assertAlmostEqual(result.cable_loss_w, 0.0, places=5)

    def test_lossy_mismatched_line_spirals_and_loses_power(self) -> None:
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
            attenuation_value=0.2,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(system, cable, chamber, points=800)
        gamma_mag = np.abs(result.gamma_local)
        self.assertGreater(float(np.max(gamma_mag) - np.min(gamma_mag)), 1e-4)
        self.assertGreater(float(result.p_real_w[0]), float(result.p_real_w[-1]))
        self.assertLessEqual(result.endpoint_match_rel_error, 1e-6)
        self.assertLessEqual(result.endpoint_end_rel_error, 1e-6)

    def test_endpoint_consistency(self) -> None:
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=900.0,
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
        result = analyze_cable(system, cable, chamber, points=1400)
        self.assertLessEqual(result.endpoint_match_rel_error, 1e-6)
        self.assertLessEqual(result.endpoint_end_rel_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
