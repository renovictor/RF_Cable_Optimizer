from __future__ import annotations

import unittest

from analysis.cable_analysis import analyze_cable
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, SystemParameters
from thermal.engine import available_heating_sources, resolve_heating_selection, run_thermal_analysis
from thermal.models import ThermalHeatingSource, ThermalInputs, ThermalModel, ThermalRegionSelection, ThermalRequest


def _base_result(connector: ConnectorParameters | None = None):
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
    return analyze_cable(system, cable, chamber, connector=connector)


class ThermalEngineTests(unittest.TestCase):
    def test_default_flexible_source_uses_rf_flexible_dissipation(self) -> None:
        rf_result = _base_result()
        request = ThermalRequest(
            model=ThermalModel.LUMPED_RC,
            heating_source=ThermalHeatingSource.FLEXIBLE_CABLE_TOTAL_DISSIPATION,
            thermal_inputs=ThermalInputs(),
            region=ThermalRegionSelection(),
        )
        result = run_thermal_analysis(request, rf_result)
        self.assertAlmostEqual(result.heating.power_w, rf_result.flexible_cable_total_loss_w, places=6)
        self.assertTrue(result.heating.rf_derived)
        self.assertEqual(result.heating.source_origin_label, "RF Cable Analysis")

    def test_connector_source_only_available_when_modeled(self) -> None:
        rf_result_no_conn = _base_result(
            ConnectorParameters(enabled=True, model_mode=ConnectorModelMode.COORDINATE_OFFSET_ONLY, physical_length_m=0.10)
        )
        availability = {item.source: item.enabled for item in available_heating_sources(rf_result_no_conn)}
        self.assertFalse(availability[ThermalHeatingSource.CONNECTOR_TOTAL_DISSIPATION])

        rf_result_with_conn = _base_result(
            ConnectorParameters(
                enabled=True,
                model_mode=ConnectorModelMode.TRANSMISSION_LINE,
                physical_length_m=0.10,
                z0_ohm=50.0,
                velocity_factor=0.70,
                attenuation_value=0.4,
                attenuation_unit=AttenuationUnit.DB_PER_M,
                ideal_lossless=False,
            )
        )
        availability = {item.source: item.enabled for item in available_heating_sources(rf_result_with_conn)}
        self.assertTrue(availability[ThermalHeatingSource.CONNECTOR_TOTAL_DISSIPATION])
        request = ThermalRequest(
            model=ThermalModel.LUMPED_RC,
            heating_source=ThermalHeatingSource.CONNECTOR_TOTAL_DISSIPATION,
            thermal_inputs=ThermalInputs(),
            region=ThermalRegionSelection(),
        )
        heating = resolve_heating_selection(request, rf_result_with_conn)
        self.assertIsNotNone(rf_result_with_conn.connector_loss_w)
        assert rf_result_with_conn.connector_loss_w is not None
        self.assertAlmostEqual(heating.power_w, rf_result_with_conn.connector_loss_w, places=6)

    def test_user_entered_source_is_labeled_what_if(self) -> None:
        rf_result = _base_result()
        request = ThermalRequest(
            model=ThermalModel.LUMPED_RC,
            heating_source=ThermalHeatingSource.USER_ENTERED_HEATING_POWER,
            thermal_inputs=ThermalInputs(),
            region=ThermalRegionSelection(),
            user_entered_heating_power_w=42.0,
        )
        result = run_thermal_analysis(request, rf_result)
        self.assertAlmostEqual(result.heating.power_w, 42.0, places=6)
        self.assertFalse(result.heating.rf_derived)
        self.assertEqual(result.heating.source_origin_label, "USER INPUT / WHAT-IF ANALYSIS")

    def test_lumped_rc_reaches_expected_steady_state(self) -> None:
        rf_result = _base_result()
        request = ThermalRequest(
            model=ThermalModel.LUMPED_RC,
            heating_source=ThermalHeatingSource.FLEXIBLE_CABLE_TOTAL_DISSIPATION,
            thermal_inputs=ThermalInputs(r_th_k_per_w=1.5, c_th_j_per_k=300.0, duration_s=5000.0),
            region=ThermalRegionSelection(),
        )
        result = run_thermal_analysis(request, rf_result)
        expected_ss = request.thermal_inputs.ambient_temp_c + result.heating.power_w * request.thermal_inputs.r_th_k_per_w
        self.assertAlmostEqual(result.steady_state_temp_c, expected_ss, places=6)
        self.assertAlmostEqual(result.temperature_c[-1], expected_ss, delta=0.5)


if __name__ == "__main__":
    unittest.main()
