from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QVBoxLayout, QWidget

from gui.focus_widgets import FocusWheelComboBox
from models.cable_parameters import AttenuationUnit
from models.connector_parameters import ConnectorModelMode


class ConnectorSpecificationPanel(QWidget):
    connectorChanged = Signal()

    def __init__(self, line_edit_factory, text_edit_factory, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        group = QGroupBox("MATCH-SIDE CONNECTOR / TRANSITION")
        form = QFormLayout(group)
        form.setVerticalSpacing(8)

        self.include_connector = QCheckBox("Include Connector")
        self.include_connector.setChecked(True)
        self.connector_mode = FocusWheelComboBox()
        self.connector_mode.addItems(
            [
                ConnectorModelMode.COORDINATE_OFFSET_ONLY.value,
                ConnectorModelMode.TRANSMISSION_LINE.value,
                "S-PARAMETER / TOUCHSTONE (FUTURE)",
            ]
        )
        self.connector_mode.setCurrentText(ConnectorModelMode.TRANSMISSION_LINE.value)
        self.connector_model_name = text_edit_factory("Unknown / Input Required", "Connector model / description.")
        self.connector_length = line_edit_factory("0.1", "Connector physical length [m].")
        self.connector_z0 = line_edit_factory("50", "Connector characteristic impedance [ohm].")
        self.connector_vf = line_edit_factory("0.76", "Connector velocity factor.")
        self.connector_att = line_edit_factory("0.06", "Connector attenuation value.")
        self.connector_att_unit = FocusWheelComboBox()
        self.connector_att_unit.addItems([u.value for u in AttenuationUnit])
        self.connector_att_unit.setCurrentText(AttenuationUnit.DB_PER_M.value)
        self.connector_ideal_lossless = QCheckBox("Ideal Lossless Connector")
        self.connector_source_status = text_edit_factory("USER INPUT", "Connector source status.")
        self.connector_notes = text_edit_factory("", "Connector notes.")
        self.connector_mode_note = QLabel("COORDINATE OFFSET ONLY\nCONNECTOR RF BEHAVIOR NOT MODELED")
        self.connector_mode_note.setStyleSheet("color:#c4b5fd;")
        self.connector_mode_note.setContentsMargins(0, 6, 0, 8)
        self.connector_mode_note.setMinimumHeight(52)
        self.connector_metrics = QLabel("Connector Electrical Length: -\nConnector Delay: -")
        self.connector_metrics.setStyleSheet("color:#93c5fd;")
        self.connector_mode_note.setWordWrap(True)
        self.connector_metrics.setWordWrap(True)

        form.addRow(self.include_connector)
        form.addRow("Connector Mode", self.connector_mode)
        form.addRow("Connector Model / Description", self.connector_model_name)
        form.addRow("Connector Physical Length [m]", self.connector_length)
        form.addRow("Connector Z0 [ohm]", self.connector_z0)
        form.addRow("Connector Velocity Factor", self.connector_vf)
        form.addRow("Connector Attenuation", self.connector_att)
        form.addRow("Connector Attenuation Unit", self.connector_att_unit)
        form.addRow(self.connector_ideal_lossless)
        form.addRow("Connector Source Status", self.connector_source_status)
        form.addRow("Notes", self.connector_notes)
        form.addRow(self.connector_mode_note)
        form.addRow(self.connector_metrics)
        layout.addWidget(group)

        for widget in (
            self.include_connector,
            self.connector_mode,
            self.connector_model_name,
            self.connector_length,
            self.connector_z0,
            self.connector_vf,
            self.connector_att,
            self.connector_att_unit,
            self.connector_ideal_lossless,
            self.connector_source_status,
            self.connector_notes,
        ):
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(lambda *_: self.connectorChanged.emit())
            elif isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self.connectorChanged.emit)
            else:
                widget.currentTextChanged.connect(lambda *_: self.connectorChanged.emit())

        self.include_connector.toggled.connect(self._sync_enabled_state)
        self.connector_mode.currentTextChanged.connect(self._sync_enabled_state)
        self.connector_ideal_lossless.toggled.connect(self._sync_enabled_state)
        self._sync_enabled_state()

    def _sync_enabled_state(self) -> None:
        enabled = self.include_connector.isChecked()
        mode = self.connector_mode.currentText()
        tl_mode = enabled and mode == ConnectorModelMode.TRANSMISSION_LINE.value
        for widget in (
            self.connector_mode,
            self.connector_model_name,
            self.connector_length,
            self.connector_source_status,
            self.connector_notes,
        ):
            widget.setEnabled(enabled)
        for widget in (self.connector_z0, self.connector_vf, self.connector_att, self.connector_att_unit, self.connector_ideal_lossless):
            widget.setEnabled(tl_mode)
        self.connector_att.setEnabled(tl_mode and (not self.connector_ideal_lossless.isChecked()))
        self.connector_att_unit.setEnabled(tl_mode and (not self.connector_ideal_lossless.isChecked()))
        self.connector_mode_note.setVisible(enabled and mode == ConnectorModelMode.COORDINATE_OFFSET_ONLY.value)
