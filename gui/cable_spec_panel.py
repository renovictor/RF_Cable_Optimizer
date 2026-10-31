from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QLineEdit, QVBoxLayout, QWidget

from gui.connector_spec_panel import ConnectorSpecificationPanel
from gui.focus_widgets import FocusWheelComboBox
from models.cable_parameters import AttenuationUnit, CableModelMode
from models.connector_parameters import HistoricalCoordinateReference


class CableSpecificationPanel(QWidget):
    parameterChanged = Signal()
    connectorChanged = Signal()

    def __init__(self, line_edit_factory, text_edit_factory, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)

        basic_group = QGroupBox("Basic Cable Electrical Parameters")
        basic_form = QFormLayout(basic_group)
        self.cable_mode = FocusWheelComboBox()
        self.cable_mode.addItems([m.value for m in CableModelMode])
        self.z0 = line_edit_factory("50", "Cable characteristic impedance Z0 [ohm].")
        self.length = line_edit_factory("1.03", "Flexible cable physical length [m].")
        self.vf = line_edit_factory("0.76", "Cable velocity factor [0..1].")
        basic_form.addRow("Cable Model", self.cable_mode)
        basic_form.addRow("Cable Z0 [ohm]", self.z0)
        basic_form.addRow("Cable Length [m]", self.length)
        basic_form.addRow("Cable Velocity Factor", self.vf)
        root.addWidget(basic_group)

        loss_group = QGroupBox("Attenuation / Loss Model")
        loss_form = QFormLayout(loss_group)
        self.att = line_edit_factory("0.6", "Datasheet attenuation value.")
        self.att_unit = FocusWheelComboBox()
        self.att_unit.addItems([u.value for u in AttenuationUnit])
        self.att_unit.setCurrentText(AttenuationUnit.DB_PER_100FT.value)
        self.att_mode = FocusWheelComboBox()
        self.att_mode.addItems(
            [
                "MANUAL ENTRY",
                "MANUFACTURER TABLE ONLY",
                "INTERPOLATE BETWEEN PUBLISHED POINTS",
            ]
        )
        self.att_info = QLabel("Attenuation source status: USER INPUT")
        self.att_info.setWordWrap(True)
        self.att_info.setStyleSheet("color: #facc15;")
        loss_form.addRow("Cable Attenuation", self.att)
        loss_form.addRow("Attenuation Unit", self.att_unit)
        loss_form.addRow("Attenuation Source", self.att_mode)
        loss_form.addRow(self.att_info)
        root.addWidget(loss_group)

        rlgc_group = QGroupBox("RLGC Parameters")
        rlgc_form = QFormLayout(rlgc_group)
        self.rp = line_edit_factory("0.2331", "R' [ohm/m]")
        self.lp = line_edit_factory("2.1945e-7", "L' [H/m]")
        self.gp = line_edit_factory("0", "G' [S/m]")
        self.cp = line_edit_factory("8.7780e-11", "C' [F/m]")
        rlgc_form.addRow("R' [ohm/m]", self.rp)
        rlgc_form.addRow("L' [H/m]", self.lp)
        rlgc_form.addRow("G' [S/m]", self.gp)
        rlgc_form.addRow("C' [F/m]", self.cp)
        root.addWidget(rlgc_group)

        spec_group = QGroupBox("Cable Preset / Assembly Information")
        spec_form = QFormLayout(spec_group)
        self.spec_family = text_edit_factory("Unknown / Input Required", "Cable family.")
        self.spec_model = text_edit_factory("Unknown / Input Required", "Cable model.")
        self.spec_part = text_edit_factory("Unknown / Input Required", "Part number.")
        self.spec_assembly = text_edit_factory("Unknown / Input Required", "Assembly designation.")
        self.spec_outer_d = text_edit_factory("Unknown / Input Required", "Outer diameter [mm].")
        self.spec_center_d = text_edit_factory("Unknown / Input Required", "Center-conductor diameter [mm].")
        self.spec_dielectric_d = text_edit_factory("Unknown / Input Required", "Dielectric diameter [mm].")
        self.spec_d_inner = text_edit_factory("UNKNOWN / INPUT REQUIRED", "Outer conductor inner diameter D [mm].")
        self.spec_dielectric = text_edit_factory("Unknown / Input Required", "Dielectric material.")
        self.spec_center_type = text_edit_factory("Unknown / Input Required", "Center conductor.")
        self.spec_outer_shield = text_edit_factory("Unknown / Input Required", "Outer shield.")
        self.spec_jacket = text_edit_factory("Unknown / Input Required", "Jacket material.")
        self.spec_temp_range = text_edit_factory("Unknown / Input Required", "Temperature range.")
        self.spec_bend_radius = text_edit_factory("Unknown / Input Required", "Bend radius [mm].")
        self.spec_status = QLabel("Status: USER INPUT")
        self.spec_field_status = QLabel("Field status: USER INPUT")
        self.spec_field_status.setWordWrap(True)
        self.spec_field_status.setStyleSheet("color: #a5f3fc;")
        self.power_rating_panel = QLabel("")
        self.power_rating_panel.setWordWrap(True)
        self.power_rating_panel.setStyleSheet("color: #c4b5fd;")
        spec_form.addRow("Cable Family", self.spec_family)
        spec_form.addRow("Cable Model", self.spec_model)
        spec_form.addRow("Part Number", self.spec_part)
        spec_form.addRow("Assembly Designation", self.spec_assembly)
        spec_form.addRow("Outer Diameter [mm]", self.spec_outer_d)
        spec_form.addRow("Center-Conductor Diameter [mm]", self.spec_center_d)
        spec_form.addRow("Dielectric Diameter [mm]", self.spec_dielectric_d)
        spec_form.addRow("D [mm]", self.spec_d_inner)
        spec_form.addRow("Dielectric Material", self.spec_dielectric)
        spec_form.addRow("Center Conductor", self.spec_center_type)
        spec_form.addRow("Outer Shield", self.spec_outer_shield)
        spec_form.addRow("Jacket", self.spec_jacket)
        spec_form.addRow("Temperature Range", self.spec_temp_range)
        spec_form.addRow("Bend Radius [mm]", self.spec_bend_radius)
        spec_form.addRow(self.spec_status)
        spec_form.addRow(self.spec_field_status)
        spec_form.addRow("Manufacturer Power Points", self.power_rating_panel)
        root.addWidget(spec_group)

        reference_group = QGroupBox("Reference Planes / Coordinates")
        reference_form = QFormLayout(reference_group)
        self.historical_damage_start = line_edit_factory("0.10", "Historical damage start [m].")
        self.historical_damage_end = line_edit_factory("0.20", "Historical damage end [m].")
        self.historical_coordinate_reference = FocusWheelComboBox()
        self.historical_coordinate_reference.addItems([r.value for r in HistoricalCoordinateReference])
        self.historical_coordinate_reference.setCurrentText(HistoricalCoordinateReference.UNKNOWN_UNVERIFIED.value)
        reference_form.addRow("Historical Damage Start [m]", self.historical_damage_start)
        reference_form.addRow("Historical Damage End [m]", self.historical_damage_end)
        reference_form.addRow("Historical Coordinate Reference", self.historical_coordinate_reference)
        root.addWidget(reference_group)

        self.connector_panel = ConnectorSpecificationPanel(line_edit_factory, text_edit_factory, self)
        root.addWidget(self.connector_panel)
        root.addStretch(1)

        self.connector_panel.connectorChanged.connect(self.connectorChanged.emit)

        for widget in (
            self.cable_mode,
            self.z0,
            self.length,
            self.vf,
            self.att,
            self.att_unit,
            self.att_mode,
            self.rp,
            self.lp,
            self.gp,
            self.cp,
            self.spec_family,
            self.spec_model,
            self.spec_part,
            self.spec_assembly,
            self.spec_outer_d,
            self.spec_center_d,
            self.spec_dielectric_d,
            self.spec_d_inner,
            self.spec_dielectric,
            self.spec_center_type,
            self.spec_outer_shield,
            self.spec_jacket,
            self.spec_temp_range,
            self.spec_bend_radius,
            self.historical_damage_start,
            self.historical_damage_end,
            self.historical_coordinate_reference,
        ):
            if isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self.parameterChanged.emit)
            else:
                widget.currentTextChanged.connect(lambda *_: self.parameterChanged.emit())
