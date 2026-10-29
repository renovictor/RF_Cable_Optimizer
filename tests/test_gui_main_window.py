from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from models.system_parameters import MatchOutputPowerSource


class MainWindowGuiTests(unittest.TestCase):
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

    def test_file_exit_triggers_clean_shutdown(self) -> None:
        self.assertTrue(self.window.isVisible())
        self.window.exit_action.trigger()
        self.app.processEvents()
        self.assertFalse(self.window.isVisible())

    def test_help_menu_opens_non_modal_help_window(self) -> None:
        self.window.help_action.trigger()
        self.app.processEvents()
        self.assertIsNotNone(self.window.help_window)
        assert self.window.help_window is not None
        self.assertTrue(self.window.help_window.isVisible())
        self.assertFalse(self.window.help_window.isModal())

    def test_main_window_remains_operable_with_help_open(self) -> None:
        self.window.help_action.trigger()
        self.app.processEvents()
        self.assertTrue(self.window.run_btn.isEnabled())
        self.assertTrue(self.window.isEnabled())

    def test_help_window_single_instance(self) -> None:
        self.window.help_action.trigger()
        self.app.processEvents()
        first_ref = self.window.help_window
        self.window.help_action.trigger()
        self.app.processEvents()
        self.assertIs(self.window.help_window, first_ref)

    def test_operating_point_updates_match_input_power(self) -> None:
        self.window.operating_point_combo.setCurrentText("1200 W")
        self.app.processEvents()
        self.assertEqual(self.window.p_match_in_w.text(), "1200")

    def test_aa11459_preset_loads_expected_electrical_defaults(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.app.processEvents()
        self.assertEqual(self.window.z0.text(), "50")
        self.assertEqual(self.window.vf.text(), "0.76")
        self.assertEqual(self.window.length.text(), "1.03")

    def test_aa11459_preset_does_not_load_vf_088(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.app.processEvents()
        self.assertNotEqual(self.window.vf.text(), "0.88")

    def test_switching_to_aa11459_recalculates_electrical_length(self) -> None:
        self.window.cable_preset.setCurrentText("Custom")
        self.window.length.setText("2.0")
        self.window.vf.setText("0.88")
        self.window.recalculate()
        self.app.processEvents()
        assert self.window.last_result is not None
        initial_length_deg = self.window.last_result.electrical_length_deg

        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.app.processEvents()
        assert self.window.last_result is not None
        updated_length_deg = self.window.last_result.electrical_length_deg
        self.assertNotAlmostEqual(initial_length_deg, updated_length_deg, places=6)

    def test_changing_cable_length_recalculates_distribution(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.window.recalculate()
        self.app.processEvents()
        assert self.window.last_result is not None
        first = self.window.last_result
        first_end = float(first.z_m[-1])
        first_v = first.v_rms.copy()

        self.window.length.setText("1.50")
        self.window.recalculate()
        self.app.processEvents()
        assert self.window.last_result is not None
        second = self.window.last_result
        self.assertNotAlmostEqual(first_end, float(second.z_m[-1]), places=6)
        self.assertNotEqual(float(abs(first_v[100])), float(abs(second.v_rms[100])))

    def test_unknown_geometry_d_is_not_fabricated(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.app.processEvents()
        self.assertEqual(self.window.spec_d_inner.text(), "UNKNOWN / INPUT REQUIRED")

    def test_power_ratings_not_labeled_as_current_limits(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.app.processEvents()
        text = self.window.power_rating_panel.text()
        self.assertIn("Equivalent Matched-Line RMS Current", text)
        self.assertNotIn("Maximum Cable Current", text)
        self.assertNotIn("Current Limit", text)

    def test_matched_line_current_label_is_derived(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.window.recalculate()
        self.app.processEvents()
        self.assertIn("[DERIVED]", self.window.current_comparison_label.text())
        self.assertNotIn("MANUFACTURER SPEC", self.window.current_comparison_label.text())

    def test_interpolated_60mhz_attenuation_is_labeled_interpolated(self) -> None:
        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.window.att_mode.setCurrentText("INTERPOLATE BETWEEN PUBLISHED POINTS")
        self.app.processEvents()
        self.assertIn("INTERPOLATED", self.window.att_info.text())

    def test_historical_damage_overlay_independent_of_preset(self) -> None:
        self.window.cable_preset.setCurrentText("DX7 Existing Cable")
        self.window.recalculate()
        self.app.processEvents()
        axes_dx7 = self.window.dist_canvas.figure.axes
        dx7_patch_count = sum(len(ax.patches) for ax in axes_dx7)

        self.window.cable_preset.setCurrentText("COMCRAFT AA-11459")
        self.window.recalculate()
        self.app.processEvents()
        axes_aa = self.window.dist_canvas.figure.axes
        aa_patch_count = sum(len(ax.patches) for ax in axes_aa)
        self.assertEqual(dx7_patch_count, aa_patch_count)

    def test_match_output_power_reflects_matcher_efficiency(self) -> None:
        self.window.power_source.setCurrentText(MatchOutputPowerSource.CALCULATED.value)
        self.window.match_eff.setText("90")
        self.window.operating_point_combo.setCurrentText("1300 W")
        self.window.recalculate()
        assert self.window.last_result is not None
        self.assertAlmostEqual(self.window.last_result.match_output_power_w, 1170.0, places=6)

    def test_power_balance_equation_holds(self) -> None:
        self.window.power_source.setCurrentText(MatchOutputPowerSource.CALCULATED.value)
        self.window.match_eff.setText("95")
        self.window.recalculate()
        assert self.window.last_result is not None
        r = self.window.last_result
        self.assertAlmostEqual(r.match_input_power_w, r.match_loss_w + r.cable_loss_w + r.chamber_power_w, places=6)


if __name__ == "__main__":
    unittest.main()
