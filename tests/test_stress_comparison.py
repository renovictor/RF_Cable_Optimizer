from __future__ import annotations

import math
import unittest

import numpy as np

from analysis.cable_analysis import analyze_cable
from analysis.stress_comparison import (
    build_stress_comparison_result,
    current_enhancement,
    current_squared_stress_index,
    equivalent_matched_current_from_rating,
    local_current_enhancement_profile,
    matched_current_rms,
    matched_voltage_rms,
)
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters


def _base_result():
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
    return analyze_cable(system, cable, chamber)


class StressComparisonTests(unittest.TestCase):
    def test_matched_current(self) -> None:
        self.assertAlmostEqual(matched_current_rms(1300.0, 50.0), math.sqrt(1300.0 / 50.0), places=8)

    def test_matched_voltage(self) -> None:
        self.assertAlmostEqual(matched_voltage_rms(1300.0, 50.0), math.sqrt(1300.0 * 50.0), places=8)

    def test_current_enhancement_ratio(self) -> None:
        matched = math.sqrt(1300.0 / 50.0)
        self.assertAlmostEqual(current_enhancement(24.0, matched), 24.0 / matched, places=8)

    def test_current_squared_stress_index(self) -> None:
        k_i = 2.35
        self.assertAlmostEqual(current_squared_stress_index(k_i), k_i * k_i, places=8)

    def test_matched_case_returns_unity_factors(self) -> None:
        self.assertEqual(current_enhancement(5.0, 5.0), 1.0)
        self.assertEqual(current_squared_stress_index(1.0), 1.0)

    def test_zero_match_output_power_is_safe(self) -> None:
        self.assertEqual(matched_current_rms(0.0, 50.0), 0.0)
        self.assertTrue(np.isnan(current_enhancement(10.0, 0.0)))

    def test_invalid_z0_raises(self) -> None:
        with self.assertRaises(ValueError):
            matched_current_rms(1000.0, 0.0)
        with self.assertRaises(ValueError):
            matched_voltage_rms(1000.0, -5.0)

    def test_manufacturer_equivalent_current(self) -> None:
        self.assertAlmostEqual(equivalent_matched_current_from_rating(2000.0, 50.0), math.sqrt(2000.0 / 50.0), places=8)

    def test_derived_manufacturer_value_is_not_current_limit_label(self) -> None:
        result = _base_result()
        stress = build_stress_comparison_result(result, manufacturer_power_rating_w=4300.0)
        self.assertIn("Manufacturer Power Rating", stress.source_status)
        self.assertIn("Current Limit", stress.source_status)
        self.assertEqual(stress.source_status["Current Limit"], "UNKNOWN")

    def test_local_stress_profile_uses_abs_i_over_matched(self) -> None:
        i = np.array([1 + 1j, 2 + 0j, 3 - 4j], dtype=np.complex128)
        profile = local_current_enhancement_profile(i, 2.0)
        np.testing.assert_allclose(profile, np.abs(i) / 2.0, atol=1e-12)

    def test_actual_maxima_are_taken_from_distributed_result(self) -> None:
        result = _base_result()
        stress = build_stress_comparison_result(result)
        self.assertAlmostEqual(stress.actual_max_current_rms_a, float(np.max(np.abs(result.i_rms))), places=6)
        self.assertAlmostEqual(stress.actual_max_voltage_rms_v, float(np.max(np.abs(result.v_rms))), places=6)


if __name__ == "__main__":
    unittest.main()
