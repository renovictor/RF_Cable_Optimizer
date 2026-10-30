from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QScrollArea

from gui.main_window import MainWindow


class ScrollableSpecPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.resize(1366, 768)
        self.window.show()
        for idx in range(self.window.tabs.count()):
            if self.window.tabs.tabText(idx) == "Cable Analysis":
                self.window.tabs.setCurrentIndex(idx)
                break
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_cable_spec_panel_uses_vertical_scroll_area(self) -> None:
        self.assertIsInstance(self.window.cable_spec_scroll_area, QScrollArea)
        self.assertFalse(self.window.cable_spec_scroll_area.widgetResizable())
        self.assertTrue(self.window.cable_spec_scroll_area.verticalScrollBar().maximum() >= 0)
        self.assertTrue(self.window.cable_spec_scroll_area.horizontalScrollBar().maximum() >= 0)

    def test_scrolling_does_not_trigger_reanalysis(self) -> None:
        initial_calls = self.window._analysis_call_count
        bar = self.window.cable_spec_scroll_area.verticalScrollBar()
        bar.setValue(min(bar.maximum(), bar.value() + 120))
        self.app.processEvents()
        self.assertEqual(initial_calls, self.window._analysis_call_count)

    def test_summary_cards_and_smith_remain_visible_while_scrolling(self) -> None:
        bar = self.window.cable_spec_scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())
        self.app.processEvents()
        self.assertTrue(self.window.smith_canvas.isVisible())
        self.assertTrue(self.window.card_labels["match_output"].isVisible())

    def test_invalid_offscreen_connector_field_is_scrolled_into_view(self) -> None:
        self.window.include_connector.setChecked(True)
        self.window.connector_mode.setCurrentText("TRANSMISSION-LINE SECTION")
        self.window.connector_length.setText("0.10")
        self.window.connector_ideal_lossless.setChecked(True)
        self.window.connector_vf.setText("")
        self.window.recalculate()
        self.app.processEvents()
        self.assertIs(self.window._last_invalid_widget, self.window.connector_vf)


if __name__ == "__main__":
    unittest.main()
