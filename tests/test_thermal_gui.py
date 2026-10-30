from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QScrollArea

from gui.main_window import MainWindow
from thermal.models import ThermalHeatingSource


class ThermalGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_thermal_tab_exists(self) -> None:
        tab_names = [self.window.tabs.tabText(i) for i in range(self.window.tabs.count())]
        self.assertIn("Thermal Analysis", tab_names)

    def test_thermal_source_and_parameter_panels_use_scroll_areas(self) -> None:
        self.assertIsInstance(self.window.thermal_source_scroll_area, QScrollArea)
        self.assertIsInstance(self.window.thermal_params_scroll_area, QScrollArea)
        self.assertTrue(self.window.thermal_source_scroll_area.verticalScrollBar().maximum() >= 0)
        self.assertTrue(self.window.thermal_params_scroll_area.verticalScrollBar().maximum() >= 0)

    def test_default_thermal_source_is_flexible_cable_dissipation(self) -> None:
        self.assertEqual(
            self.window.thermal_source_combo.currentText(),
            ThermalHeatingSource.FLEXIBLE_CABLE_TOTAL_DISSIPATION.value,
        )
        self.assertIn("RF Cable Analysis", self.window.thermal_heating_source_origin.text())
        assert self.window.last_result is not None
        self.assertIn(f"{self.window.last_result.flexible_cable_total_loss_w:.2f}", self.window.thermal_rf_heating_power.text())

    def test_user_entered_source_shows_what_if_label(self) -> None:
        idx = self.window.thermal_source_combo.findText(ThermalHeatingSource.USER_ENTERED_HEATING_POWER.value)
        self.window.thermal_source_combo.setCurrentIndex(idx)
        self.window.thermal_user_power_w.setText("33")
        self.window._update_thermal_analysis()
        self.app.processEvents()
        self.assertIn("USER INPUT / WHAT-IF ANALYSIS", self.window.thermal_heating_source_origin.text())
        self.assertIn("USER-ENTERED HEATING POWER", self.window.thermal_rf_heating_power.text())

    def test_connector_status_shows_not_modeled_when_unavailable(self) -> None:
        self.window.include_connector.setChecked(False)
        self.window.recalculate()
        self.app.processEvents()
        self.assertIn("NOT MODELED", self.window.thermal_connector_status.text())

    def test_connector_heating_option_appears_when_connector_loss_modeled(self) -> None:
        self.window.include_connector.setChecked(True)
        self.window.connector_mode.setCurrentText("TRANSMISSION-LINE SECTION")
        self.window.connector_length.setText("0.10")
        self.window.connector_z0.setText("50")
        self.window.connector_vf.setText("0.70")
        self.window.connector_ideal_lossless.setChecked(False)
        self.window.connector_att.setText("0.4")
        self.window.connector_att_unit.setCurrentText("dB/m")
        self.window.recalculate()
        self.app.processEvents()
        options = [self.window.thermal_source_combo.itemText(i) for i in range(self.window.thermal_source_combo.count())]
        self.assertIn(ThermalHeatingSource.CONNECTOR_TOTAL_DISSIPATION.value, options)


if __name__ == "__main__":
    unittest.main()
