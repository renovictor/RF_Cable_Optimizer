from __future__ import annotations

import unittest

import numpy as np

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters


class ConnectorTransmissionLineTests(unittest.TestCase):
    def _system(self) -> SystemParameters:
        return SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1000.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(50.0, 0.0),
        )

    def _connector(self, z0: float = 50.0, vf: float = 0.76, attenuation: float = 0.0, ideal: bool = True) -> ConnectorParameters:
        return ConnectorParameters(
            enabled=True,
            model_mode=ConnectorModelMode.TRANSMISSION_LINE,
            physical_length_m=0.10,
            z0_ohm=z0,
            velocity_factor=vf,
            attenuation_value=attenuation,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            ideal_lossless=ideal,
        )

    def test_equal_z0_lossless_no_artificial_step_reflection(self) -> None:
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.USER_ENTERED, impedance_real_ohm=50.0, impedance_imag_ohm=0.0)
        result = analyze_cable(self._system(), cable, chamber, connector=self._connector())
        self.assertAlmostEqual(abs(result.gamma_local[0]), 0.0, places=6)
        self.assertAlmostEqual(result.connector_loss_w or 0.0, 0.0, places=6)

    def test_impedance_step_appears_with_mismatched_sections(self) -> None:
        cable = CableParameters(
            z0_ohm=25.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.USER_ENTERED, impedance_real_ohm=25.0, impedance_imag_ohm=0.0)
        result = analyze_cable(self._system(), cable, chamber, connector=self._connector(z0=50.0))
        assert result.connector_interface_index is not None
        interface_gamma = abs(result.gamma_local[result.connector_interface_index])
        self.assertGreater(interface_gamma, 1e-6)

    def test_equal_z0_different_vf_changes_total_electrical_length(self) -> None:
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        r_same = analyze_cable(self._system(), cable, chamber, connector=self._connector(vf=0.76))
        r_diff = analyze_cable(self._system(), cable, chamber, connector=self._connector(vf=0.50))
        self.assertAlmostEqual(abs(r_same.gamma_local[0]), abs(r_diff.gamma_local[0]), places=6)
        self.assertNotAlmostEqual(r_same.electrical_length_deg, r_diff.electrical_length_deg, places=6)

    def test_lossy_connector_dissipates_power(self) -> None:
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.0,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        result = analyze_cable(self._system(), cable, chamber, connector=self._connector(attenuation=1.0, ideal=False))
        self.assertIsNotNone(result.connector_input_power_w)
        self.assertIsNotNone(result.connector_output_power_w)
        assert result.connector_input_power_w is not None
        assert result.connector_output_power_w is not None
        assert result.connector_loss_w is not None
        self.assertGreater(result.connector_input_power_w, result.connector_output_power_w)
        self.assertGreater(result.connector_loss_w, 0.0)

    def test_coordinate_offset_mode_preserves_rf_arrays(self) -> None:
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.2,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        baseline = analyze_cable(self._system(), cable, chamber)
        offset = analyze_cable(
            self._system(),
            cable,
            chamber,
            connector=ConnectorParameters(
                enabled=True,
                model_mode=ConnectorModelMode.COORDINATE_OFFSET_ONLY,
                physical_length_m=0.10,
            ),
        )
        np.testing.assert_allclose(baseline.v_rms, offset.v_rms, atol=1e-10)
        np.testing.assert_allclose(baseline.i_rms, offset.i_rms, atol=1e-10)
        np.testing.assert_allclose(baseline.gamma_local, offset.gamma_local, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
