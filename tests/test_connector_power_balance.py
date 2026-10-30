from __future__ import annotations

import unittest

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters


class ConnectorPowerBalanceTests(unittest.TestCase):
    def test_transmission_line_connector_power_balance(self) -> None:
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
        connector = ConnectorParameters(
            enabled=True,
            model_mode=ConnectorModelMode.TRANSMISSION_LINE,
            physical_length_m=0.10,
            z0_ohm=50.0,
            velocity_factor=0.70,
            attenuation_value=0.4,
            attenuation_unit=AttenuationUnit.DB_PER_M,
            ideal_lossless=False,
        )
        result = analyze_cable(system, cable, chamber, connector=connector)
        assert result.connector_loss_w is not None
        self.assertAlmostEqual(
            result.match_input_power_w,
            result.match_loss_w + result.connector_loss_w + result.flexible_cable_total_loss_w + result.cable_end_delivered_power_w,
            places=5,
        )

    def test_coordinate_offset_connector_reports_loss_not_modeled(self) -> None:
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
            attenuation_value=0.6,
            attenuation_unit=AttenuationUnit.DB_PER_100FT,
            model_mode=CableModelMode.DATASHEET,
        )
        chamber = ChamberParameters(mode=ChamberImpedanceMode.DERIVED_FROM_MATCH)
        connector = ConnectorParameters(
            enabled=True,
            model_mode=ConnectorModelMode.COORDINATE_OFFSET_ONLY,
            physical_length_m=0.10,
        )
        result = analyze_cable(system, cable, chamber, connector=connector)
        self.assertIsNone(result.connector_loss_w)


if __name__ == "__main__":
    unittest.main()
