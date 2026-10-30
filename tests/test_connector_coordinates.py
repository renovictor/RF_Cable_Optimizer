from __future__ import annotations

import unittest
import math

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters


class ConnectorCoordinateTests(unittest.TestCase):
    def _base(self):
        system = SystemParameters(
            frequency_mhz=60.0,
            match_input_power_w=1000.0,
            match_output_power_source=MatchOutputPowerSource.IDEAL,
            match_output_impedance=ComplexImpedance(3.0, 19.0),
        )
        cable = CableParameters(
            z0_ohm=50.0,
            length_m=1.0,
            velocity_factor=0.76,
            attenuation_value=0.2,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        return system, cable, chamber

    def test_coordinate_offset_maps_global_to_cable_local(self) -> None:
        system, cable, chamber = self._base()
        connector = ConnectorParameters(
            enabled=True,
            model_mode=ConnectorModelMode.COORDINATE_OFFSET_ONLY,
            physical_length_m=0.10,
        )
        result = analyze_cable(system, cable, chamber, connector=connector)
        idx = int(min(range(len(result.z_m)), key=lambda i: abs(float(result.z_m[i]) - 0.25)))
        self.assertAlmostEqual(float(result.position_from_flexible_cable_start_m[idx]), 0.15, places=2)

    def test_inside_connector_does_not_return_negative_cable_position(self) -> None:
        system, cable, chamber = self._base()
        connector = ConnectorParameters(
            enabled=True,
            model_mode=ConnectorModelMode.COORDINATE_OFFSET_ONLY,
            physical_length_m=0.10,
        )
        result = analyze_cable(system, cable, chamber, connector=connector)
        idx = int(min(range(len(result.z_m)), key=lambda i: abs(float(result.z_m[i]) - 0.05)))
        self.assertEqual(str(result.section_name[idx]), "Connector / Transition")
        self.assertTrue(math.isnan(float(result.position_from_flexible_cable_start_m[idx])))

    def test_disabled_connector_keeps_positions_identical(self) -> None:
        system, cable, chamber = self._base()
        result = analyze_cable(system, cable, chamber)
        self.assertAlmostEqual(float(result.global_position_m[100]), float(result.position_from_flexible_cable_start_m[100]), places=6)


if __name__ == "__main__":
    unittest.main()
