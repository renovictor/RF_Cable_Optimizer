from __future__ import annotations

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


class SmithGuiTests(unittest.TestCase):
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

    def test_smith_chart_is_available_on_cable_tab(self) -> None:
        self.assertIsNotNone(self.window.smith_canvas)
        self.assertGreaterEqual(len(self.window.smith_canvas.figure.axes), 1)

    def test_probe_slider_updates_smith_probe_marker(self) -> None:
        self.window.probe_slider.setValue(300)
        self.app.processEvents()
        artist = self.window.smith_canvas._probe_artist
        self.assertIsNotNone(artist)
        self.assertAlmostEqual(float(artist.get_xdata()[0]), self.window._probe_state_at(float(self.window.probe_pos.text()))["gamma"].real)

    def test_probe_text_updates_smith_probe_marker(self) -> None:
        self.window.probe_pos.setText("0.600")
        self.window._on_probe_text()
        self.app.processEvents()
        artist = self.window.smith_canvas._probe_artist
        self.assertIsNotNone(artist)
        self.assertAlmostEqual(float(self.window.probe_pos.text()), 0.6, places=2)

    def test_smith_selection_signal_moves_probe(self) -> None:
        self.window.smith_canvas.probePositionSelected.emit(0.42)
        self.app.processEvents()
        self.assertAlmostEqual(float(self.window.probe_pos.text()), 0.42, places=2)
        self.assertIn("@ z = ", self.window.smith_probe_label.text())

    def test_measurement_table_selection_moves_probe(self) -> None:
        self.window.multi_positions.setText("0.05,0.10,0.15")
        self.window._refresh_probe_table()
        self.window.probe_table.selectRow(1)
        self.app.processEvents()
        self.assertAlmostEqual(float(self.window.probe_pos.text()), 0.10, places=2)

    def test_cable_length_change_redraws_smith_trajectory(self) -> None:
        self.window.recalculate()
        self.app.processEvents()
        initial_gamma = self.window.last_result.gamma_local.copy()  # type: ignore[union-attr]
        self.window.length.setText("1.50")
        self.window.recalculate()
        self.app.processEvents()
        updated_gamma = self.window.last_result.gamma_local  # type: ignore[union-attr]
        self.assertNotEqual(float(np.abs(initial_gamma[100])), float(np.abs(updated_gamma[100])))

    def test_z0_change_updates_smith_reference_label(self) -> None:
        self.window.z0.setText("30")
        self.window.recalculate()
        self.app.processEvents()
        self.assertIn("30.000", self.window.smith_ref_label.text())

    def test_historical_segment_visibility_depends_on_preset(self) -> None:
        self.window.cable_preset.setCurrentText("DX7 Existing Cable")
        self.window.recalculate()
        self.app.processEvents()
        self.assertIn("Historical Damage Distance", self.window.smith_note_label.text())

        self.window.cable_preset.setCurrentText("Custom")
        self.window.recalculate()
        self.app.processEvents()
        self.assertEqual(self.window.smith_note_label.text(), "")

    def test_help_contains_smith_section_and_is_non_modal(self) -> None:
        self.window.help_action.trigger()
        self.app.processEvents()
        assert self.window.help_window is not None
        self.assertFalse(self.window.help_window.isModal())
        self.assertIn("Smith Chart", self.window.help_window.browser.toPlainText())

    def test_profile_csv_includes_smith_columns(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            temp_path = tmp.name
        try:
            with patch("gui.main_window.QFileDialog.getSaveFileName", return_value=(temp_path, "CSV (*.csv)")):
                self.window.export_profile_csv()
            with open(temp_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            header = next(row for row in rows if row and row[0] == "global_position_from_match_m")
            self.assertIn("section_name", header)
            self.assertIn("position_from_flexible_cable_start_m", header)
            self.assertIn("Gamma_imag", header)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
