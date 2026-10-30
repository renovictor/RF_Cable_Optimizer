from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDoubleValidator, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from analysis.cable_analysis import CableAnalysisResult, analyze_cable
from analysis.optimization import OptimizationWeights, SweepResult, sweep_z0
from gui.help_window import HelpWindow
from gui.plots import MplCanvas
from gui.smith_chart import SmithChartCanvas
from models.cable_presets import AA11459_ATTENUATION_DB_PER_100FT, AA11459_POWER_RATING_KW, interpolate_log_frequency
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters, CableSpecification
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.operating_points import find_operating_point
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, MatcherModelMode, SystemParameters
from models.validation import validate_inputs
from rf.coax import inner_diameter_for_target_z0, z0_from_geometry
from rf.smith_math import vswr_from_gamma
from version import APP_VERSION


def _line_edit(value: str, tooltip: str) -> QLineEdit:
    w = QLineEdit(value)
    w.setToolTip(tooltip)
    w.setValidator(QDoubleValidator(bottom=-1e12, top=1e12, decimals=8))
    return w


def _text_line_edit(value: str, tooltip: str) -> QLineEdit:
    w = QLineEdit(value)
    w.setToolTip(tooltip)
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"RF Match-to-Chamber Cable Engineering Analyzer - {APP_VERSION}")
        icon_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        icon_path = icon_base / "smithchart.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1500, 980)
        self._apply_theme()
        self.last_result: CableAnalysisResult | None = None
        self.last_sweep: SweepResult | None = None
        self.help_window: HelpWindow | None = None
        self._measurement_row_positions: list[float] = []

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self._build_system_tab()
        self._build_cable_tab()
        self._build_distribution_tab()
        self._build_optimization_tab()
        self._build_coax_tab()
        self._build_summary_tab()
        self._create_menu_bar()

        self.recalculate()

    def _create_menu_bar(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        help_menu = menu.addMenu("Help")

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setShortcutContext(Qt.ApplicationShortcut)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        self.exit_action = exit_action

        help_action = QAction("RF Cable Optimizer Help", self)
        help_action.triggered.connect(self._show_help_window)
        help_menu.addAction(help_action)
        self.help_action = help_action

    def _show_help_window(self) -> None:
        if self.help_window is None:
            self.help_window = HelpWindow(self)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #0f172a; color: #e2e8f0; }
            QTabBar::tab { background: #1e293b; color: #cbd5e1; padding: 8px 14px; border: 1px solid #334155; }
            QTabBar::tab:selected { background: #334155; color: #f8fafc; }
            QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 8px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 6px; color: #93c5fd; }
            QLineEdit, QComboBox, QTableWidget { background: #1e293b; border: 1px solid #475569; padding: 5px; }
            QPushButton { background-color: #2563eb; color: white; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background-color: #1d4ed8; }
            QSlider::groove:horizontal { border: 1px solid #9a3412; height: 8px; background: #1f2937; border-radius: 4px; }
            QSlider::sub-page:horizontal { background: #ea580c; border-radius: 4px; }
            QSlider::handle:horizontal { background: #f97316; border: 1px solid #9a3412; width: 18px; margin: -5px 0; border-radius: 9px; }
            """
        )

    def _build_system_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "System Input")
        layout = QGridLayout(tab)

        op_box = QGroupBox("Operating Point (Match Input Power)")
        op_layout = QFormLayout(op_box)
        self.operating_point_combo = QComboBox()
        for power_w in range(100, 1301, 100):
            self.operating_point_combo.addItem(f"{power_w} W")
        self.operating_point_combo.setCurrentText("1300 W")
        self.operating_point_combo.currentTextChanged.connect(self._on_operating_point_changed)
        op_layout.addRow("Select Match Input Power:", self.operating_point_combo)
        self.measured_load = QLabel("Load: -")
        self.measured_tune = QLabel("Tune: -")
        self.measured_iout = QLabel("Measured Iout - Untrusted: -")
        self.measured_iout.setStyleSheet("color: #fb7185;")
        op_layout.addRow(self.measured_load)
        op_layout.addRow(self.measured_tune)
        op_layout.addRow(self.measured_iout)

        sys_box = QGroupBox("Match Input & Matcher")
        sys = QFormLayout(sys_box)
        self.freq_mhz = _line_edit("60", "System RF frequency in MHz.")
        self.p_match_in_w = _line_edit("1300", "Real RF power at matcher input reference plane [W].")
        self.z_gen = _line_edit("50", "Generator reference impedance [ohm].")
        self.p_match_out = _line_edit("1300", "Match Output real power at match-output reference plane [W].")
        self.match_eff = _line_edit("70", "Match efficiency [%] when using calculated mode.")
        self.power_source = QComboBox()
        self.power_source.addItems([s.value for s in MatchOutputPowerSource])
        self.zm_r = _line_edit("3", "Match Output impedance real part [ohm].")
        self.zm_x = _line_edit("19", "Match Output impedance imaginary part [ohm].")
        self.vpp = _line_edit("", "Measured Vpp at Match Output [V]. Optional.")
        self.s_mode_lbl = QLabel("Matcher Model: IDEAL")
        self.matcher_mode = QComboBox()
        self.matcher_mode.addItems([m.value for m in MatcherModelMode])
        self.matcher_mode.currentTextChanged.connect(self._on_matcher_mode_changed)
        self.s11r = _line_edit("0", "S11 real")
        self.s11i = _line_edit("0", "S11 imag")
        self.s12r = _line_edit("1", "S12 real")
        self.s12i = _line_edit("0", "S12 imag")
        self.s21r = _line_edit("1", "S21 real")
        self.s21i = _line_edit("0", "S21 imag")
        self.s22r = _line_edit("0", "S22 real")
        self.s22i = _line_edit("0", "S22 imag")
        self.s_zref = _line_edit("50", "S-parameter reference impedance [ohm]")
        self.s_freq = _line_edit("60", "S-parameter frequency [MHz]")
        sys.addRow("Frequency [MHz]", self.freq_mhz)
        sys.addRow("Match Input Power [W]", self.p_match_in_w)
        sys.addRow("Generator Z [ohm]", self.z_gen)
        sys.addRow("Match Output Power [W]", self.p_match_out)
        sys.addRow("Match Output Power Mode", self.power_source)
        sys.addRow("Match Efficiency [%]", self.match_eff)
        sys.addRow("Match Output R [ohm]", self.zm_r)
        sys.addRow("Match Output X [ohm]", self.zm_x)
        sys.addRow("Measured Vpp [V]", self.vpp)
        sys.addRow("Matcher Model", self.matcher_mode)
        sys.addRow("S11 (Re, Im)", self._pair_widget(self.s11r, self.s11i))
        sys.addRow("S12 (Re, Im)", self._pair_widget(self.s12r, self.s12i))
        sys.addRow("S21 (Re, Im)", self._pair_widget(self.s21r, self.s21i))
        sys.addRow("S22 (Re, Im)", self._pair_widget(self.s22r, self.s22i))
        sys.addRow("S-Param Zref [ohm]", self.s_zref)
        sys.addRow("S-Param Freq [MHz]", self.s_freq)
        sys.addRow(self.s_mode_lbl)

        chamber_box = QGroupBox("Chamber Impedance")
        chamber = QFormLayout(chamber_box)
        self.chamber_mode = QComboBox()
        self.chamber_mode.addItems([m.value for m in ChamberImpedanceMode])
        self.chamber_mode.setCurrentText(ChamberImpedanceMode.DERIVED_FROM_MATCH.value)
        self.chamber_mode.currentTextChanged.connect(self._toggle_chamber_fields)
        self.zl_r = _line_edit("", "Chamber impedance real part [ohm].")
        self.zl_x = _line_edit("", "Chamber impedance imaginary part [ohm].")
        chamber.addRow("Mode", self.chamber_mode)
        chamber.addRow("Chamber R [ohm]", self.zl_r)
        chamber.addRow("Chamber X [ohm]", self.zl_x)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.recalculate)
        btns.addWidget(self.run_btn)
        self.export_profile_btn = QPushButton("Export Profile CSV")
        self.export_profile_btn.clicked.connect(self.export_profile_csv)
        btns.addWidget(self.export_profile_btn)
        btns.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(op_box)
        left.addWidget(sys_box)
        left.addWidget(chamber_box)
        left.addLayout(btns)
        left.addStretch(1)
        layout.addLayout(left, 0, 0)

        self.system_diag = QLabel("")
        self.system_diag.setWordWrap(True)
        self.system_diag.setStyleSheet("color: #a7f3d0;")
        layout.addWidget(self.system_diag, 0, 1)
        self._toggle_chamber_fields()

    def _build_cable_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Cable Analysis")
        layout = QVBoxLayout(tab)

        cable_box = QGroupBox("Cable Parameters")
        f = QFormLayout(cable_box)
        self.cable_mode = QComboBox()
        self.cable_mode.addItems([m.value for m in CableModelMode])
        self.cable_mode.currentTextChanged.connect(self._toggle_rlgc_fields)
        self.z0 = _line_edit("50", "Cable characteristic impedance Z0 [ohm], user configurable/sweepable.")
        self.length = _line_edit("1.03", "Cable physical length [m], z=0 at Match Output.")
        self.vf = _line_edit("0.76", "Velocity factor [0..1].")
        self.att = _line_edit("0.6", "Datasheet attenuation value.")
        self.att_unit = QComboBox()
        self.att_unit.addItems([u.value for u in AttenuationUnit])
        self.att_unit.setCurrentText(AttenuationUnit.DB_PER_100FT.value)
        self.att_mode = QComboBox()
        self.att_mode.addItems(
            [
                "MANUAL ENTRY",
                "MANUFACTURER TABLE ONLY",
                "INTERPOLATE BETWEEN PUBLISHED POINTS",
            ]
        )
        self.att_mode.currentTextChanged.connect(self._on_attenuation_mode_changed)
        self.rp = _line_edit("", "R' [ohm/m]")
        self.lp = _line_edit("", "L' [H/m]")
        self.gp = _line_edit("", "G' [S/m]")
        self.cp = _line_edit("", "C' [F/m]")
        f.addRow("Cable Model", self.cable_mode)
        f.addRow("Z0 [ohm]", self.z0)
        f.addRow("Length [m]", self.length)
        f.addRow("Velocity Factor", self.vf)
        f.addRow("Attenuation", self.att)
        f.addRow("Attenuation Unit", self.att_unit)
        f.addRow("Attenuation Source", self.att_mode)
        f.addRow("R' [ohm/m]", self.rp)
        f.addRow("L' [H/m]", self.lp)
        f.addRow("G' [S/m]", self.gp)
        f.addRow("C' [F/m]", self.cp)
        self.att_info = QLabel("Attenuation source status: USER INPUT")
        self.att_info.setWordWrap(True)
        self.att_info.setStyleSheet("color: #facc15;")
        f.addRow(self.att_info)

        spec_box = QGroupBox("Cable Specification / Preset")
        s = QFormLayout(spec_box)
        self.cable_preset = QComboBox()
        self.cable_preset.addItems(["DX7 Existing Cable", "COMCRAFT AA-11459", "Custom"])
        self.cable_preset.currentTextChanged.connect(self._on_cable_preset_changed)
        self.spec_family = _text_line_edit("Unknown / Input Required", "Cable family.")
        self.spec_model = _text_line_edit("Unknown / Input Required", "Cable model / assembly designation.")
        self.spec_part = _text_line_edit("Unknown / Input Required", "ASM part number.")
        self.spec_assembly = _text_line_edit("Unknown / Input Required", "Assembly designation.")
        self.spec_outer_d = _text_line_edit("Unknown / Input Required", "Cable outer diameter [mm].")
        self.spec_center_d = _text_line_edit("Unknown / Input Required", "Center conductor diameter [mm].")
        self.spec_dielectric_d = _text_line_edit("Unknown / Input Required", "Dielectric diameter [mm].")
        self.spec_center_type = _text_line_edit("Unknown / Input Required", "Center conductor construction.")
        self.spec_dielectric = _text_line_edit("Unknown / Input Required", "Dielectric material.")
        self.spec_outer_shield = _text_line_edit("Unknown / Input Required", "Outer shield construction.")
        self.spec_jacket = _text_line_edit("Unknown / Input Required", "Jacket material.")
        self.spec_temp_range = _text_line_edit("Unknown / Input Required", "Cable published temperature range.")
        self.spec_bend_radius = _text_line_edit("Unknown / Input Required", "Minimum bend radius [mm].")
        self.spec_d_inner = _text_line_edit(
            "UNKNOWN / INPUT REQUIRED",
            "D for coax equation is inner diameter of outer conductor; do not fabricate from overall dimensions.",
        )
        self.spec_status = QLabel("Status: USER INPUT")
        self.spec_field_status = QLabel("Field status: USER INPUT")
        self.spec_field_status.setWordWrap(True)
        self.spec_field_status.setStyleSheet("color: #a5f3fc;")
        self.power_rating_panel = QLabel("")
        self.power_rating_panel.setWordWrap(True)
        self.power_rating_panel.setStyleSheet("color: #c4b5fd;")
        self.power_rating_panel.setToolTip(
            "Equivalent Matched-Line RMS Current is a calculated engineering reference and is not a manufacturer-specified maximum RF current."
        )
        s.addRow("Preset", self.cable_preset)
        s.addRow("Cable Family", self.spec_family)
        s.addRow("Model", self.spec_model)
        s.addRow("ASM Part Number", self.spec_part)
        s.addRow("Assembly", self.spec_assembly)
        s.addRow("Outer Diameter [mm]", self.spec_outer_d)
        s.addRow("Center Conductor Diameter [mm]", self.spec_center_d)
        s.addRow("Dielectric Diameter [mm]", self.spec_dielectric_d)
        s.addRow("Center Conductor", self.spec_center_type)
        s.addRow("Dielectric", self.spec_dielectric)
        s.addRow("Outer Shield", self.spec_outer_shield)
        s.addRow("Jacket", self.spec_jacket)
        s.addRow("Cable Published Temperature Range", self.spec_temp_range)
        s.addRow("Minimum Bend Radius [mm]", self.spec_bend_radius)
        s.addRow("Outer-Conductor Inner Diameter D [mm]", self.spec_d_inner)
        s.addRow(self.spec_status)
        s.addRow(self.spec_field_status)
        s.addRow("AA-11459 Power Rating", self.power_rating_panel)

        cards = QGroupBox("Power Flow")
        cg = QGridLayout(cards)
        self.card_labels: dict[str, QLabel] = {}
        for idx, (title, target, color) in enumerate(
            [
                ("MATCH INPUT POWER", "match_input", "#3b82f6"),
                ("MATCH LOSS", "match_loss", "#f59e0b"),
                ("MATCH OUTPUT POWER", "match_output", "#22d3ee"),
                ("CABLE DISSIPATION", "cable_loss", "#ef4444"),
                ("CABLE END DELIVERED POWER", "chamber", "#22c55e"),
            ]
        ):
            t = QLabel(f"{title}\n-")
            t.setStyleSheet(f"color:{color}; border:1px solid #334155; border-radius:8px; padding:10px;")
            self.card_labels[target] = t
            cg.addWidget(t, idx // 3, idx % 3, 1, 1)

        self.efficiency_label = QLabel("Matcher Efficiency: -\nCable Efficiency: -\nOverall Delivery Efficiency: -")
        self.efficiency_label.setStyleSheet("color:#93c5fd;")
        cg.addWidget(self.efficiency_label, 2, 0, 1, 3)

        self.stress_metrics_label = QLabel("Run analysis to view max voltage/current/local dissipation.")
        self.stress_metrics_label.setStyleSheet("color:#cbd5e1;")
        cg.addWidget(self.stress_metrics_label, 3, 0, 1, 3)
        self.current_comparison_label = QLabel(
            "Matched-Line Current: - Arms\n"
            "Simulated Maximum: - Arms\n"
            "Current Enhancement: - x"
        )
        self.current_comparison_label.setStyleSheet("color:#e9d5ff;")
        cg.addWidget(self.current_comparison_label, 4, 0, 1, 3)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color:#f87171;")
        self.live_metrics_label = QLabel(self._format_live_metrics(None, None))
        self.live_metrics_label.setStyleSheet("color:#93c5fd;")

        action_row = QHBoxLayout()
        self.run_btn_cable = QPushButton("Run Analysis")
        self.run_btn_cable.clicked.connect(self.recalculate)
        action_row.addWidget(self.run_btn_cable)
        self.export_smith_btn = QPushButton("Export Smith Chart Image")
        self.export_smith_btn.clicked.connect(self.export_smith_chart_image)
        action_row.addWidget(self.export_smith_btn)
        action_row.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(cable_box)
        left_layout.addWidget(spec_box)
        left_layout.addWidget(cards)
        left_layout.addWidget(self.warning_label)
        left_layout.addLayout(action_row)
        left_layout.addWidget(self.live_metrics_label)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.smith_canvas = SmithChartCanvas(width=7.0, height=7.0, dpi=100)
        self.smith_canvas.probePositionSelected.connect(self._on_smith_probe_selected)
        self.smith_canvas.set_hover_formatter(self._format_smith_hover)
        right_layout.addWidget(self.smith_canvas, 4)

        smith_cards = QGroupBox("Smith Chart Summary")
        sg = QVBoxLayout(smith_cards)
        self.smith_match_label = QLabel("MATCH OUTPUT IMPEDANCE\n-")
        self.smith_probe_label = QLabel("PROBE IMPEDANCE\n-")
        self.smith_end_label = QLabel("CABLE END IMPEDANCE\n-")
        self.smith_ref_label = QLabel("SMITH REFERENCE\nZ0 = -")
        self.smith_probe_gamma_label = QLabel("PROBE |GAMMA|\n-")
        self.smith_note_label = QLabel("")
        self.smith_note_label.setStyleSheet("color:#facc15;")
        for widget in (
            self.smith_match_label,
            self.smith_probe_label,
            self.smith_end_label,
            self.smith_ref_label,
            self.smith_probe_gamma_label,
        ):
            widget.setStyleSheet("border:1px solid #334155; border-radius:8px; padding:8px;")
            sg.addWidget(widget)
        sg.addWidget(self.smith_note_label)
        right_layout.addWidget(smith_cards, 2)

        probe_box = QGroupBox("Probe Impedance Details")
        probe_layout = QVBoxLayout(probe_box)
        self.probe_impedance_label = QLabel("Probe impedance details are available after analysis.")
        self.probe_impedance_label.setWordWrap(True)
        self.probe_impedance_label.setStyleSheet("color:#f8fafc;")
        probe_layout.addWidget(self.probe_impedance_label)
        right_layout.addWidget(probe_box, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        self._toggle_rlgc_fields()
        self.cable_preset.setCurrentText("DX7 Existing Cable")

    @staticmethod
    def _pair_widget(left: QWidget, right: QWidget) -> QWidget:
        wrapper = QWidget()
        l = QHBoxLayout(wrapper)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(left)
        l.addWidget(right)
        return wrapper

    def _build_distribution_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Voltage / Current Distribution")
        layout = QVBoxLayout(tab)
        probe_row = QHBoxLayout()
        self.probe_pos = _line_edit("0.15", "Probe position from Match Output [m].")
        self.probe_pos.setMaximumWidth(90)
        self.probe_slider = QSlider(Qt.Horizontal)
        self.probe_slider.setRange(0, 1000)
        self.probe_slider.setMinimumWidth(380)
        self.probe_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { border: 1px solid #9a3412; height: 10px; background: #111827; border-radius: 5px; }
            QSlider::sub-page:horizontal { background: #ea580c; border-radius: 5px; }
            QSlider::add-page:horizontal { background: #374151; border-radius: 5px; }
            QSlider::handle:horizontal { background: #fb923c; border: 1px solid #c2410c; width: 20px; margin: -6px 0; border-radius: 10px; }
            """
        )
        self.probe_slider.valueChanged.connect(self._on_probe_slider)
        self.probe_pos.editingFinished.connect(self._on_probe_text)
        probe_row.addWidget(QLabel("Probe Position [m]"))
        probe_row.addWidget(self.probe_pos)
        probe_row.addWidget(self.probe_slider, 3)
        self.probe_values = QLabel("Probe: Input Required")
        probe_row.addWidget(self.probe_values, 2)
        layout.addLayout(probe_row)

        positions_row = QHBoxLayout()
        self.multi_positions = QLineEdit("0.05,0.10,0.15,0.20,0.50,1.00")
        self.multi_positions.setToolTip("Comma-separated positions in meters.")
        self.multi_positions.editingFinished.connect(self._refresh_probe_table)
        positions_row.addWidget(QLabel("Measurement Positions [m]"))
        positions_row.addWidget(self.multi_positions, 1)
        layout.addLayout(positions_row)

        self.probe_table = QTableWidget(0, 8)
        self.probe_table.setHorizontalHeaderLabels(
            [
                "Position [m]",
                "Vrms [V]",
                "Vpp [V]",
                "Irms [A]",
                "P(z) [W]",
                "Cond [W/m]",
                "Diel [W/m]",
                "CumLoss [W]",
            ]
        )
        self.probe_table.itemSelectionChanged.connect(self._on_probe_table_selection_changed)
        layout.addWidget(self.probe_table, 1)

        self.dist_canvas = MplCanvas(width=12, height=9, dpi=100)
        layout.addWidget(self.dist_canvas, 4)

    def _build_optimization_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Z0 Optimization")
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.z0_min = _line_edit("10", "Sweep minimum Z0 [ohm].")
        self.z0_max = _line_edit("60", "Sweep maximum Z0 [ohm].")
        self.z0_step = _line_edit("1", "Sweep step [ohm].")
        self.w_loss = _line_edit("1", "Weight for cable loss.")
        self.w_v = _line_edit("1", "Weight for max voltage.")
        self.w_i = _line_edit("1", "Weight for max current.")
        for title, widget in [
            ("Z0 min", self.z0_min),
            ("Z0 max", self.z0_max),
            ("Step", self.z0_step),
            ("w_loss", self.w_loss),
            ("w_voltage", self.w_v),
            ("w_current", self.w_i),
        ]:
            row.addWidget(QLabel(title))
            row.addWidget(widget)
        run = QPushButton("Run Z0 Sweep")
        run.clicked.connect(self.run_sweep)
        row.addWidget(run)
        exp = QPushButton("Export Sweep CSV")
        exp.clicked.connect(self.export_sweep_csv)
        row.addWidget(exp)
        layout.addLayout(row)
        self.sweep_note = QLabel("Score is user-defined engineering criterion, not a physical law.")
        self.sweep_note.setStyleSheet("color:#93c5fd;")
        layout.addWidget(self.sweep_note)
        self.sweep_canvas = MplCanvas(width=12, height=8, dpi=100)
        layout.addWidget(self.sweep_canvas, 1)

    def _build_coax_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Coax Geometry")
        layout = QFormLayout(tab)
        self.coax_mode = QComboBox()
        self.coax_mode.addItems(
            ["MODE 1: Direct Z0 Input", "MODE 2: Calculate Z0 from Geometry", "MODE 3: Design Geometry for Target Z0"]
        )
        self.coax_D = _line_edit("", "D = inner diameter of outer conductor [mm].")
        self.coax_d = _line_edit("", "d = outer diameter of inner conductor [mm].")
        self.coax_er = _line_edit("2.1", "Relative dielectric constant.")
        self.coax_target_z0 = _line_edit("30", "Target Z0 [ohm] for Mode 3.")
        self.coax_result = QLabel("Ideal TEM Coax Approximation")
        run = QPushButton("Calculate Geometry")
        run.clicked.connect(self.calculate_coax)
        layout.addRow("Mode", self.coax_mode)
        layout.addRow("D [mm]", self.coax_D)
        layout.addRow("d [mm]", self.coax_d)
        layout.addRow("epsilon_r", self.coax_er)
        layout.addRow("Target Z0 [ohm]", self.coax_target_z0)
        layout.addRow(run)
        layout.addRow(self.coax_result)

    def _build_summary_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Engineering Summary")
        layout = QVBoxLayout(tab)
        self.summary_label = QLabel("Run analysis to view summary.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

    def _toggle_rlgc_fields(self) -> None:
        use_rlgc = self.cable_mode.currentText() == CableModelMode.RLGC.value
        for w in (self.rp, self.lp, self.gp, self.cp):
            w.setEnabled(use_rlgc)
        self.att.setEnabled(not use_rlgc)
        self.att_unit.setEnabled(not use_rlgc)
        self.att_mode.setEnabled(not use_rlgc)

    def _toggle_chamber_fields(self) -> None:
        direct = self.chamber_mode.currentText() != ChamberImpedanceMode.DERIVED_FROM_MATCH.value
        self.zl_r.setEnabled(direct)
        self.zl_x.setEnabled(direct)

    @staticmethod
    def _as_float_or_none(text: str) -> float | None:
        value = text.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _active_frequency_mhz(self) -> float:
        try:
            f_mhz = float(self.freq_mhz.text())
        except ValueError:
            return 60.0
        return f_mhz if f_mhz > 0 else 60.0

    def _update_power_rating_panel(self) -> None:
        lines = ["Manufacturer specification points:"]
        for freq_mhz, p_kw in AA11459_POWER_RATING_KW:
            i_eq = math.sqrt(p_kw * 1000.0 / 50.0)
            lines.append(f"{freq_mhz:g} MHz    {p_kw:.2f} kW")
            lines.append(
                f"  Equivalent Matched-Line RMS Current = {i_eq:.2f} Arms "
                f"[CALCULATED ENGINEERING REFERENCE]"
            )
        self.power_rating_panel.setText("\n".join(lines))

    def _sync_attenuation_from_mode(self) -> None:
        if self.cable_preset.currentText() != "COMCRAFT AA-11459":
            if self.att_mode.currentText() == "MANUAL ENTRY":
                self.att_info.setText("Attenuation source status: USER INPUT")
            return

        mode = self.att_mode.currentText()
        if mode == "MANUFACTURER TABLE ONLY":
            self.att.setText("")
            self.att_info.setText(
                "60 MHz Attenuation: Not directly specified in current datasheet points.\n"
                "Status: MANUFACTURER SPEC"
            )
            return

        if mode == "INTERPOLATE BETWEEN PUBLISHED POINTS":
            target_f = self._active_frequency_mhz()
            try:
                interp = interpolate_log_frequency(AA11459_ATTENUATION_DB_PER_100FT, target_f)
                self.att.setText(f"{interp.value:.6g}")
                self.att_info.setText(
                    f"{target_f:.3f} MHz Attenuation: {interp.value:.6g} dB/100ft [INTERPOLATED VALUE]\n"
                    f"Source points: {interp.lower_point[0]:g} MHz ({interp.lower_point[1]:.2f} dB/100ft) "
                    f"and {interp.upper_point[0]:g} MHz ({interp.upper_point[1]:.2f} dB/100ft)\n"
                    "Status: INTERPOLATED"
                )
            except ValueError as exc:
                self.att_info.setText(f"Interpolation unavailable: {exc}")
            return

        self.att_info.setText("Attenuation source status: USER INPUT")

    def _on_attenuation_mode_changed(self, _text: str) -> None:
        self._sync_attenuation_from_mode()
        self.recalculate()

    def _on_cable_preset_changed(self, text: str) -> None:
        if text == "DX7 Existing Cable":
            self.length.setText("1.03")
            self.z0.setText("50")
            self.vf.setText("0.76")
            self.att.setText("0.6")
            self.spec_family.setText("AA-11459")
            self.spec_model.setText("AA11459-LCMR/LMCR-158M-01.03M")
            self.spec_part.setText("1238-727-01")
            self.spec_assembly.setText("AA11459-LCMR/LMCR-158M-01.03M")
            self.spec_outer_d.setText("21.46")
            self.spec_center_d.setText("5.97")
            self.spec_dielectric_d.setText("17.27")
            self.spec_center_type.setText("Stranded")
            self.spec_dielectric.setText("Taped PTFE")
            self.spec_outer_shield.setText("Double Braid")
            self.spec_jacket.setText("PUR")
            self.spec_temp_range.setText("-55 C to +105 C")
            self.spec_bend_radius.setText("107.95")
            self.spec_d_inner.setText("UNKNOWN / INPUT REQUIRED")
            self.att_mode.setCurrentText("MANUAL ENTRY")
            self.spec_status.setText("Status: INTERNAL ASSEMBLY DATA + MANUFACTURER SPEC")
            self.spec_field_status.setText(
                "Cable Family: MANUFACTURER SPEC | Velocity Factor: MANUFACTURER SPEC\n"
                "ASM P/N: INTERNAL ASSEMBLY DATA | Assembly Length: INTERNAL ASSEMBLY DATA\n"
                "Outer-Conductor Inner Diameter D: UNKNOWN / INPUT REQUIRED"
            )
            self._update_power_rating_panel()
        elif text == "COMCRAFT AA-11459":
            self.length.setText("1.03")
            self.z0.setText("50")
            self.vf.setText("0.76")
            self.spec_family.setText("AA-11459")
            self.spec_model.setText("AA-11459")
            self.spec_part.setText("1238-727-01")
            self.spec_assembly.setText("AA11459-LCMR/LMCR-158M-01.03M")
            self.spec_outer_d.setText("21.46")
            self.spec_center_d.setText("5.97")
            self.spec_dielectric_d.setText("17.27")
            self.spec_center_type.setText("Stranded")
            self.spec_dielectric.setText("Taped PTFE")
            self.spec_outer_shield.setText("Double Braid")
            self.spec_jacket.setText("PUR")
            self.spec_temp_range.setText("-55 C to +105 C")
            self.spec_bend_radius.setText("107.95")
            self.spec_d_inner.setText("UNKNOWN / INPUT REQUIRED")
            self.att_mode.setCurrentText("INTERPOLATE BETWEEN PUBLISHED POINTS")
            self.spec_status.setText("Status: MANUFACTURER SPEC + INTERNAL ASSEMBLY DATA")
            self.spec_field_status.setText(
                "Cable Family: MANUFACTURER SPEC | Velocity Factor: MANUFACTURER SPEC\n"
                "ASM P/N: INTERNAL ASSEMBLY DATA | Assembly Length: INTERNAL ASSEMBLY DATA\n"
                "60 MHz Attenuation: INTERPOLATED (when interpolation mode is selected)\n"
                "Outer-Conductor Inner Diameter D: UNKNOWN / INPUT REQUIRED"
            )
            self._update_power_rating_panel()
        else:
            self.spec_status.setText("Status: USER INPUT")
            self.spec_family.setText("Unknown / Input Required")
            self.spec_model.setText("Unknown / Input Required")
            self.spec_part.setText("Unknown / Input Required")
            self.spec_assembly.setText("Unknown / Input Required")
            self.spec_outer_d.setText("Unknown / Input Required")
            self.spec_center_d.setText("Unknown / Input Required")
            self.spec_dielectric_d.setText("Unknown / Input Required")
            self.spec_center_type.setText("Unknown / Input Required")
            self.spec_dielectric.setText("Unknown / Input Required")
            self.spec_outer_shield.setText("Unknown / Input Required")
            self.spec_jacket.setText("Unknown / Input Required")
            self.spec_temp_range.setText("Unknown / Input Required")
            self.spec_bend_radius.setText("Unknown / Input Required")
            self.spec_d_inner.setText("UNKNOWN / INPUT REQUIRED")
            self.spec_field_status.setText("Field status: USER INPUT")
            self.power_rating_panel.setText("")
            if self.att_mode.currentText() != "MANUAL ENTRY":
                self.att_mode.setCurrentText("MANUAL ENTRY")
        self._sync_attenuation_from_mode()
        self.recalculate()

    def _on_matcher_mode_changed(self, text: str) -> None:
        self.s_mode_lbl.setText(f"Matcher Model: {text}")
        self.recalculate()

    def _on_operating_point_changed(self, text: str) -> None:
        if "W" in text:
            selected_power_w = float(text.replace("W", "").strip())
            self.p_match_in_w.setText(f"{selected_power_w:g}")
            p = find_operating_point(selected_power_w)
            if p is not None:
                self.zm_r.setText(str(p.zreal_ohm))
                self.zm_x.setText(str(p.zimag_ohm))
                self.vpp.setText(str(p.vpp_v))
                self.measured_load.setText(f"Load: {p.load_setting}")
                self.measured_tune.setText(f"Tune: {p.tune_setting}")
                self.measured_iout.setText(f"Measured Iout - Untrusted: {p.iout_a_untrusted} A")
            else:
                self.measured_load.setText("Load: -")
                self.measured_tune.setText("Tune: -")
                self.measured_iout.setText("Measured Iout - Untrusted: -")
        self.recalculate()

    def _collect_models(self) -> tuple[SystemParameters, CableParameters, ChamberParameters]:
        self._sync_attenuation_from_mode()
        source = MatchOutputPowerSource(self.power_source.currentText())
        matcher_mode = MatcherModelMode(self.matcher_mode.currentText())
        system = SystemParameters(
            frequency_mhz=float(self.freq_mhz.text()),
            match_input_power_w=float(self.p_match_in_w.text()),
            generator_impedance_ohm=float(self.z_gen.text()),
            match_efficiency_percent=float(self.match_eff.text()),
            match_output_power_source=source,
            matcher_model_mode=matcher_mode,
            match_output_power_w=float(self.p_match_out.text()) if self.p_match_out.text().strip() else None,
            match_output_impedance=ComplexImpedance(float(self.zm_r.text()), float(self.zm_x.text())),
            measured_vpp_v=float(self.vpp.text()) if self.vpp.text().strip() else None,
            measured_iout_untrusted_a=None,
            s11=complex(float(self.s11r.text()), float(self.s11i.text())),
            s12=complex(float(self.s12r.text()), float(self.s12i.text())),
            s21=complex(float(self.s21r.text()), float(self.s21i.text())),
            s22=complex(float(self.s22r.text()), float(self.s22i.text())),
            sparam_ref_impedance_ohm=float(self.s_zref.text()),
            sparam_frequency_mhz=float(self.s_freq.text()) if self.s_freq.text().strip() else None,
        )
        c_mode = CableModelMode(self.cable_mode.currentText())
        cable = CableParameters(
            z0_ohm=float(self.z0.text()),
            length_m=float(self.length.text()) if self.length.text().strip() else None,
            velocity_factor=float(self.vf.text()),
            attenuation_value=float(self.att.text()) if self.att.text().strip() else None,
            attenuation_unit=AttenuationUnit(self.att_unit.currentText()),
            model_mode=c_mode,
            r_per_m=float(self.rp.text()) if self.rp.text().strip() else None,
            l_per_m=float(self.lp.text()) if self.lp.text().strip() else None,
            g_per_m=float(self.gp.text()) if self.gp.text().strip() else None,
            c_per_m=float(self.cp.text()) if self.cp.text().strip() else None,
            specification=CableSpecification(
                preset_name=self.cable_preset.currentText(),
                cable_family=self.spec_family.text(),
                cable_model=self.spec_model.text(),
                part_number=self.spec_part.text(),
                assembly_designation=self.spec_assembly.text(),
                assembly_length_m=self._as_float_or_none(self.length.text()),
                outer_diameter_mm=self._as_float_or_none(self.spec_outer_d.text()),
                center_conductor_diameter_mm=self._as_float_or_none(self.spec_center_d.text()),
                dielectric_diameter_mm=self._as_float_or_none(self.spec_dielectric_d.text()),
                center_conductor_type=self.spec_center_type.text(),
                dielectric_material=self.spec_dielectric.text(),
                outer_shield=self.spec_outer_shield.text(),
                jacket=self.spec_jacket.text(),
                temperature_range_c=self.spec_temp_range.text(),
                min_bend_radius_mm=self._as_float_or_none(self.spec_bend_radius.text()),
                outer_conductor_inner_d_mm=self._as_float_or_none(self.spec_d_inner.text()),
                attenuation_status=self.att_info.text(),
                source_status=self.spec_status.text().replace("Status: ", ""),
            ),
        )
        ch_mode = ChamberImpedanceMode(self.chamber_mode.currentText())
        chamber = ChamberParameters(
            mode=ch_mode,
            impedance_real_ohm=float(self.zl_r.text()) if self.zl_r.text().strip() else None,
            impedance_imag_ohm=float(self.zl_x.text()) if self.zl_x.text().strip() else None,
        )
        return system, cable, chamber

    def recalculate(self) -> None:
        try:
            system, cable, chamber = self._collect_models()
            valid = validate_inputs(system, cable, chamber)
            if not valid.valid:
                self.system_diag.setText("Input Required:\n" + "\n".join(valid.errors))
                self.last_result = None
                self.live_metrics_label.setText(self._format_live_metrics(None, system))
                return
            result = analyze_cable(system, cable, chamber)
            self.last_result = result
            self.system_diag.setText(self._format_consistency_diagnostic(system))
            if system.match_output_power_source == MatchOutputPowerSource.USER and result.match_input_power_w > 0:
                derived_eff = 100.0 * result.match_output_power_w / result.match_input_power_w
                self.match_eff.setText(f"{derived_eff:.6g}")
            self._update_cards(result, system)
            all_warnings = valid.warnings + result.warnings
            if "IDEAL MATCHER - NO MATCH LOSS ASSUMED" in result.match_output_power_source_label:
                all_warnings.append("IDEAL MATCHER - NO MATCH LOSS ASSUMED")
            if system.sparam_frequency_mhz is not None and abs(system.sparam_frequency_mhz - system.frequency_mhz) > 0.5:
                all_warnings.append("S-parameter frequency differs from simulation frequency.")
            self.warning_label.setText("\n".join(all_warnings))
            try:
                probe_requested = float(self.probe_pos.text())
            except ValueError:
                probe_requested = 0.0
            probe_clamped = min(max(probe_requested, 0.0), float(result.z_m[-1]))
            self.probe_pos.setText(f"{probe_clamped:.5f}")
            self.probe_slider.setValue(int(1000.0 * probe_clamped / result.z_m[-1]))
            self._draw_distribution()
            self._update_smith_chart_full()
            self._update_summary(result, system)
            self._refresh_probe_table()
        except Exception as exc:
            self.last_result = None
            self.system_diag.setText(f"Input Required / Error: {exc}")
            self.live_metrics_label.setText(self._format_live_metrics(None, None))
            if hasattr(self, "smith_canvas"):
                self.smith_canvas.clear_chart()

    def _format_consistency_diagnostic(self, system: SystemParameters) -> str:
        if system.measured_vpp_v is None:
            return "Reference-plane / measurement consistency check: Vpp not provided."
        z = system.match_output_impedance.value
        zmag = abs(z)
        vrms = system.measured_vpp_v / (2.0 * np.sqrt(2.0))
        icalc = vrms / zmag if zmag else np.nan
        return (
            "Reference-plane / measurement consistency check\n"
            f"Zmag={zmag:.4f} ohm, Zphase={np.degrees(np.angle(z)):.2f} deg, "
            f"Vrms(Vpp)={vrms:.2f} V, Icalc={icalc:.2f} A\n"
            "Do not infer Match Input Power directly from Match Output V/I without a matcher model."
        )

    def _update_cards(self, r: CableAnalysisResult, system: SystemParameters) -> None:
        vmax_idx = int(np.argmax(np.abs(r.v_rms)))
        imax_idx = int(np.argmax(np.abs(r.i_rms)))
        wmax_idx = int(np.argmax(r.p_total_loss_w_per_m))
        eta_match = (r.match_output_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        eta_cable = (r.chamber_power_w / r.match_output_power_w * 100.0) if r.match_output_power_w > 0 else 0.0
        eta_total = (r.chamber_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        self.card_labels["match_input"].setText(f"MATCH INPUT POWER\n{r.match_input_power_w:.2f} W")
        self.card_labels["match_loss"].setText(f"MATCH LOSS\n{r.match_loss_w:.2f} W")
        self.card_labels["match_output"].setText(f"MATCH OUTPUT POWER\n{r.match_output_power_w:.2f} W")
        self.card_labels["cable_loss"].setText(f"CABLE DISSIPATION\n{r.cable_loss_w:.2f} W ({r.cable_loss_percent:.2f}%)")
        self.card_labels["chamber"].setText(f"CABLE END DELIVERED POWER\n{r.chamber_power_w:.2f} W")
        self.efficiency_label.setText(
            f"Matcher Efficiency [%]: {eta_match:.2f}\n"
            f"Cable Efficiency [%]: {eta_cable:.2f}\n"
            f"Overall Delivery Efficiency [%]: {eta_total:.2f}"
        )
        self.stress_metrics_label.setText(
            f"MAX VOLTAGE: {np.abs(r.v_rms[vmax_idx]):.2f} Vrms @ {r.z_m[vmax_idx]:.3f} m   |   "
            f"MAX CURRENT: {np.abs(r.i_rms[imax_idx]):.2f} Arms @ {r.z_m[imax_idx]:.3f} m   |   "
            f"MAX LOCAL DISSIPATION: {r.p_total_loss_w_per_m[wmax_idx]:.3f} W/m @ {r.z_m[wmax_idx]:.3f} m"
        )
        z0_ohm = self._as_float_or_none(self.z0.text())
        i_matched = 0.0
        if z0_ohm is not None and z0_ohm > 0 and r.match_output_power_w >= 0:
            i_matched = math.sqrt(r.match_output_power_w / z0_ohm)
        i_sim_max = float(np.abs(r.i_rms[imax_idx]))
        k_i = (i_sim_max / i_matched) if i_matched > 0 else float("nan")
        self.current_comparison_label.setText(
            f"SIMULATED MAXIMUM CURRENT\n{i_sim_max:.2f} Arms @ {r.z_m[imax_idx]:.3f} m\n"
            f"MATCHED-LINE CURRENT AT MATCH OUTPUT POWER\n{i_matched:.2f} Arms [DERIVED]\n"
            f"Current Enhancement: {k_i:.2f} x"
        )
        self.live_metrics_label.setText(self._format_live_metrics(r, system))

    def _format_live_metrics(self, r: CableAnalysisResult | None, system: SystemParameters | None) -> str:
        thresholds = (system.thresholds if system is not None else SystemParameters().thresholds)
        if r is None:
            return (
                f"Current VSWR - (limit at {thresholds.vswr:g})    "
                f"Current Cable Loss - (limit at {thresholds.cable_loss_percent:g}%)"
            )
        return (
            f"Current VSWR {r.vswr:.3f} (limit at {thresholds.vswr:g})    "
            f"Current Cable Loss: {r.cable_loss_percent:.2f}% ({r.cable_loss_w:.2f} W) "
            f"(limit at {thresholds.cable_loss_percent:g}%)"
        )

    def _draw_distribution(self) -> None:
        if self.last_result is None:
            return
        r = self.last_result
        z = r.z_m
        v = np.abs(r.v_rms)
        i = np.abs(r.i_rms)
        p = r.p_real_w
        pl = r.p_total_loss_w_per_m
        cl = r.p_cumulative_loss_w

        fig = self.dist_canvas.figure
        fig.clear()
        axes = fig.subplots(5, 1, sharex=True)
        axes[0].plot(z, v, color="#22d3ee", label="Voltage RMS")
        axes[1].plot(z, i, color="#fb923c", label="Current RMS")
        axes[2].plot(z, p, color="#22c55e", label="Real Power")
        axes[3].plot(z, pl, color="#ef4444", label="Local Dissipation")
        axes[4].plot(z, cl, color="#d946ef", label="Cumulative Dissipation")
        show_hist = self._historical_damage_metadata_available()
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            if show_hist:
                ax.axvspan(0.10, 0.20, color="#7f1d1d", alpha=0.15)
        axes[0].set_ylabel("Vrms [V]")
        axes[1].set_ylabel("Irms [A]")
        axes[2].set_ylabel("P(z) [W]")
        axes[3].set_ylabel("W/m")
        axes[4].set_ylabel("W")
        axes[4].set_xlabel("Position from Match Output z [m]")
        self._draw_probe_markers(axes)
        fig.tight_layout()
        self.dist_canvas.draw_idle()

    def _draw_probe_markers(self, axes) -> None:
        if self.last_result is None:
            return
        try:
            z_probe = float(self.probe_pos.text())
        except ValueError:
            return
        for ax in axes:
            ax.axvline(z_probe, color="#facc15", linestyle="--", linewidth=1.2)

    def _position_index(self, pos_m: float) -> int:
        r = self.last_result
        assert r is not None
        return int(np.argmin(np.abs(r.z_m - pos_m)))

    def _on_probe_slider(self, value: int) -> None:
        if self.last_result is None:
            return
        z = self.last_result.z_m[-1] * value / 1000.0
        self.probe_pos.setText(f"{z:.5f}")
        self._update_probe_readout()
        self._update_smith_probe_only()

    def _on_probe_text(self) -> None:
        if self.last_result is None:
            return
        try:
            z = float(self.probe_pos.text())
        except ValueError:
            return
        z = min(max(z, 0.0), float(self.last_result.z_m[-1]))
        self.probe_pos.setText(f"{z:.5f}")
        self.probe_slider.setValue(int(1000.0 * z / self.last_result.z_m[-1]))
        self._update_probe_readout()
        self._update_smith_probe_only()

    def _update_probe_readout(self) -> None:
        if self.last_result is None:
            return
        z = float(self.probe_pos.text())
        probe = self._probe_state_at(z)
        r = self.last_result
        self.probe_values.setText(
            f"Probe {probe['z']:.3f} m | Vrms={probe['vrms']:.2f} V | Irms={probe['irms']:.2f} A | "
            f"P={probe['p']:.2f} W | Local={probe['local']:.3f} W/m | Cum={probe['cum']:.3f} W"
        )
        z_ref = r.z0_reference
        z_ref_txt = (
            f"Z0 = {z_ref.real:.3f}+j{z_ref.imag:.3f} Ω"
            if abs(z_ref.imag) > 1e-9
            else f"Z0 = {z_ref.real:.3f} Ω"
        )
        gamma = probe["gamma"]
        z_local = probe["z_local"]
        z_norm = probe["z_norm"]
        probe_warning = ""
        if not probe["valid_impedance"]:
            probe_warning = "\nWARNING: Probe impedance invalid because |I(z)| is below numerical tolerance."
        elif np.isfinite(gamma.real) and np.isfinite(gamma.imag) and abs(gamma) >= 1.0:
            probe_warning = "\nWARNING: Local |Gamma| >= 1, VSWR is not finite."
        vswr_text = "N/A"
        if np.isfinite(gamma.real) and np.isfinite(gamma.imag):
            local_vswr = vswr_from_gamma(gamma)
            if np.isfinite(local_vswr):
                vswr_text = f"{local_vswr:.4f}"
            elif local_vswr == float("inf"):
                vswr_text = "Infinity"

        self.probe_impedance_label.setText(
            "DERIVED FROM V(z)/I(z)\n"
            f"Probe Position from Match Output [m]: {probe['z']:.3f}\n"
            f"Probe Position from Cable End [m]: {float(r.z_m[-1]) - probe['z']:.3f}\n"
            f"Local Impedance Z(z): {z_local.real:+.3f}{z_local.imag:+.3f}j ohm\n"
            f"Resistance R [ohm]: {z_local.real:.3f}\n"
            f"Reactance X [ohm]: {z_local.imag:.3f}\n"
            f"Impedance Magnitude |Z| [ohm]: {abs(z_local):.3f}\n"
            f"Impedance Phase [deg]: {np.degrees(np.angle(z_local)):.2f}\n"
            f"Normalized z: {z_norm.real:+.4f}{z_norm.imag:+.4f}j\n"
            f"Gamma: {gamma.real:+.4f}{gamma.imag:+.4f}j\n"
            f"|Gamma|: {abs(gamma):.4f}\n"
            f"Gamma Phase [deg]: {np.degrees(np.angle(gamma)):.2f}\n"
            f"Local VSWR: {vswr_text}\n"
            f"Vrms={probe['vrms']:.2f} V | Vpp={2.0 * np.sqrt(2.0) * probe['vrms']:.2f} V | Irms={probe['irms']:.2f} A\n"
            f"Real RF Power P(z)={probe['p']:.2f} W | Local Dissipation={probe['local']:.3f} W/m | Cumulative Loss={probe['cum']:.3f} W\n"
            f"Smith Reference {z_ref_txt}"
            f"{probe_warning}"
        )
        self.smith_probe_label.setText(
            f"PROBE IMPEDANCE\n{z_local.real:+.3f}{z_local.imag:+.3f}j ohm\n@ z = {probe['z']:.3f} m"
        )
        self.smith_probe_gamma_label.setText(f"PROBE |GAMMA|\n{abs(gamma):.4f}")
        self._draw_distribution()

    def _refresh_probe_table(self) -> None:
        if self.last_result is None:
            return
        self._update_probe_readout()
        r = self.last_result
        values = []
        self._measurement_row_positions = []
        for token in self.multi_positions.text().split(","):
            t = token.strip()
            if not t:
                continue
            try:
                z = float(t)
            except ValueError:
                continue
            if z < 0 or z > float(r.z_m[-1]):
                continue
            idx = self._position_index(z)
            vrms = np.abs(r.v_rms[idx])
            self._measurement_row_positions.append(z)
            values.append(
                (
                    z,
                    vrms,
                    2.0 * np.sqrt(2.0) * vrms,
                    np.abs(r.i_rms[idx]),
                    r.p_real_w[idx],
                    r.p_cond_w_per_m[idx],
                    r.p_diel_w_per_m[idx],
                    r.p_cumulative_loss_w[idx],
                )
            )
        self.probe_table.setRowCount(len(values))
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                self.probe_table.setItem(i, j, QTableWidgetItem(f"{v:.6g}"))
        self._update_smith_measurement_markers()

    def _historical_damage_metadata_available(self) -> bool:
        return self.cable_preset.currentText().strip() == "DX7 Existing Cable"

    @staticmethod
    def _interpolate_complex(x: np.ndarray, y: np.ndarray, x0: float) -> complex:
        real = float(np.interp(x0, x, y.real))
        imag = float(np.interp(x0, x, y.imag))
        return complex(real, imag)

    def _probe_state_at(self, z_probe: float) -> dict:
        if self.last_result is None:
            return {}
        r = self.last_result
        z_probe = min(max(z_probe, 0.0), float(r.z_m[-1]))
        z_local = self._interpolate_complex(r.z_m, r.z_local_ohm, z_probe)
        gamma = self._interpolate_complex(r.z_m, r.gamma_local, z_probe)
        z_norm = self._interpolate_complex(r.z_m, r.z_normalized, z_probe)
        vrms = float(np.interp(z_probe, r.z_m, np.abs(r.v_rms)))
        irms = float(np.interp(z_probe, r.z_m, np.abs(r.i_rms)))
        pz = float(np.interp(z_probe, r.z_m, r.p_real_w))
        local = float(np.interp(z_probe, r.z_m, r.p_total_loss_w_per_m))
        cum = float(np.interp(z_probe, r.z_m, r.p_cumulative_loss_w))
        valid_impedance = np.isfinite(z_local.real) and np.isfinite(z_local.imag)
        return {
            "z": z_probe,
            "z_local": z_local,
            "gamma": gamma,
            "z_norm": z_norm,
            "vrms": vrms,
            "irms": irms,
            "p": pz,
            "local": local,
            "cum": cum,
            "valid_impedance": valid_impedance,
        }

    def _format_smith_hover(self, idx: int) -> str:
        if self.last_result is None:
            return ""
        r = self.last_result
        z = float(r.z_m[idx])
        z_local = complex(r.z_local_ohm[idx])
        z_norm = complex(r.z_normalized[idx])
        gamma = complex(r.gamma_local[idx])
        return (
            f"z={z:.3f} m\n"
            f"Z={z_local.real:+.3f}{z_local.imag:+.3f}j Ω\n"
            f"z_norm={z_norm.real:+.3f}{z_norm.imag:+.3f}j\n"
            f"Γ={gamma.real:+.4f}{gamma.imag:+.4f}j\n"
            f"Vrms={abs(r.v_rms[idx]):.2f} V | Irms={abs(r.i_rms[idx]):.2f} A"
        )

    def _update_smith_measurement_markers(self) -> None:
        if self.last_result is None:
            return
        r = self.last_result
        gamma_values: list[complex] = []
        for z in self._measurement_row_positions:
            gamma_values.append(self._interpolate_complex(r.z_m, r.gamma_local, z))
        self.smith_canvas.set_measurement_markers(self._measurement_row_positions, gamma_values)
        self.smith_canvas.refresh_chart()

    def _update_smith_probe_only(self) -> None:
        if self.last_result is None:
            return
        probe = self._probe_state_at(float(self.probe_pos.text()))
        self.smith_canvas.set_probe(
            probe["z"],
            probe["gamma"],
            probe["z_local"],
            invalid=(not probe["valid_impedance"]),
        )
        self.smith_canvas.refresh_chart()

    def _update_smith_chart_full(self) -> None:
        if self.last_result is None:
            self.smith_canvas.clear_chart()
            return
        r = self.last_result
        title = f"Cable Position Trajectory at {self._active_frequency_mhz():.3f} MHz"
        self.smith_canvas.set_trajectory(r.gamma_local, r.z_m, title=title)
        self.smith_canvas.set_endpoints(complex(r.gamma_local[0]), complex(r.gamma_local[-1]))

        if self._historical_damage_metadata_available():
            damage_mask = (r.z_m >= 0.10) & (r.z_m <= 0.20)
            damage_gamma = np.full(r.gamma_local.shape, np.nan + 1j * np.nan, dtype=np.complex128)
            damage_gamma[damage_mask] = r.gamma_local[damage_mask]
            self.smith_canvas.set_historical_region(damage_gamma, enabled=True)
            self.smith_note_label.setText("Historical Damage Distance: 0.10 m to 0.20 m (Reference Direction Unverified)")
        else:
            self.smith_canvas.set_historical_region(np.array([], dtype=np.complex128), enabled=False)
            self.smith_note_label.setText("")

        z0 = r.z0_reference
        if abs(z0.imag) > 1e-9:
            self.smith_ref_label.setText(f"SMITH REFERENCE\nComplex Z0 = {z0.real:.3f}+j{z0.imag:.3f} ohm")
            self.smith_note_label.setText(
                self.smith_note_label.text()
                + "\nSmith Chart uses active complex reference impedance. Real-Z0 grid is visual reference only."
            )
        else:
            self.smith_ref_label.setText(f"SMITH REFERENCE\nZ0 = {z0.real:.3f} ohm")

        z_match = complex(r.z_local_ohm[0])
        z_end = complex(r.z_local_ohm[-1])
        self.smith_match_label.setText(f"MATCH OUTPUT IMPEDANCE\n{z_match.real:+.3f}{z_match.imag:+.3f}j ohm")
        self.smith_end_label.setText(f"CABLE END IMPEDANCE\n{z_end.real:+.3f}{z_end.imag:+.3f}j ohm")
        self._update_smith_measurement_markers()
        self._update_smith_probe_only()

    def _on_smith_probe_selected(self, z_m: float) -> None:
        if self.last_result is None:
            return
        z_clamped = min(max(float(z_m), 0.0), float(self.last_result.z_m[-1]))
        self.probe_pos.setText(f"{z_clamped:.5f}")
        self.probe_slider.setValue(int(1000.0 * z_clamped / self.last_result.z_m[-1]))
        self._update_probe_readout()

    def _on_probe_table_selection_changed(self) -> None:
        row = self.probe_table.currentRow()
        if row < 0 or row >= len(self._measurement_row_positions) or self.last_result is None:
            return
        z = self._measurement_row_positions[row]
        self.probe_pos.setText(f"{z:.5f}")
        self.probe_slider.setValue(int(1000.0 * z / self.last_result.z_m[-1]))
        self._update_probe_readout()

    def run_sweep(self) -> None:
        try:
            system, cable, chamber = self._collect_models()
            w = OptimizationWeights(float(self.w_loss.text()), float(self.w_v.text()), float(self.w_i.text()))
            sweep = sweep_z0(
                system,
                cable,
                chamber,
                float(self.z0_min.text()),
                float(self.z0_max.text()),
                float(self.z0_step.text()),
                w,
            )
            self.last_sweep = sweep
            fig = self.sweep_canvas.figure
            fig.clear()
            axs = fig.subplots(2, 2)
            axs = axs.flatten()
            axs[0].plot(sweep.z0_ohm, sweep.cable_loss_w, color="#ef4444")
            axs[0].set_title("Z0 vs Cable Loss [W]")
            axs[1].plot(sweep.z0_ohm, sweep.max_voltage_rms, color="#22d3ee")
            axs[1].set_title("Z0 vs Max Voltage [Vrms]")
            axs[2].plot(sweep.z0_ohm, sweep.max_current_rms, color="#fb923c")
            axs[2].set_title("Z0 vs Max Current [Arms]")
            axs[3].plot(sweep.z0_ohm, sweep.chamber_power_w, color="#22c55e")
            axs[3].set_title("Z0 vs Chamber Power [W]")
            for ax in axs:
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Z0 [ohm]")
            fig.tight_layout()
            self.sweep_canvas.draw_idle()
        except Exception as exc:
            QMessageBox.warning(self, "Sweep Error", str(exc))

    def calculate_coax(self) -> None:
        try:
            mode = self.coax_mode.currentText().strip().lower()
            er = float(self.coax_er.text())
            if mode.startswith("mode 2"):
                z0 = z0_from_geometry(float(self.coax_D.text()), float(self.coax_d.text()), er)
                self.coax_result.setText(f"Ideal TEM Coax Approximation: Z0 = {z0:.4f} ohm")
                self.z0.setText(f"{z0:.6g}")
            elif mode.startswith("mode 3"):
                d_req = inner_diameter_for_target_z0(float(self.coax_target_z0.text()), float(self.coax_D.text()), er)
                self.coax_result.setText(f"Ideal TEM Coax Approximation: required d = {d_req:.4f} mm")
                self.coax_d.setText(f"{d_req:.6g}")
            else:
                self.coax_result.setText("Mode 1 uses direct user-entered Z0.")
            self.recalculate()
        except Exception as exc:
            QMessageBox.warning(self, "Coax Geometry Error", str(exc))

    def export_profile_csv(self) -> None:
        if self.last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Cable Profile CSV", "cable_profile.csv", "CSV (*.csv)")
        if not path:
            return
        r = self.last_result
        system, cable, chamber = self._collect_models()
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["SECTION", "key", "value"])
            for k, v in asdict(system).items():
                w.writerow(["system", k, v])
            for k, v in asdict(cable).items():
                w.writerow(["cable", k, v])
            for k, v in asdict(chamber).items():
                w.writerow(["chamber", k, v])
            w.writerow([])
            w.writerow(
                [
                    "position_m",
                    "V_rms",
                    "I_rms",
                    "P_real_W",
                    "P_cond_W_per_m",
                    "P_diel_W_per_m",
                    "P_total_loss_W_per_m",
                    "P_cumulative_loss_W",
                    "z_local_real_ohm",
                    "z_local_imag_ohm",
                    "z_local_magnitude_ohm",
                    "z_local_phase_deg",
                    "z_normalized_real",
                    "z_normalized_imag",
                    "gamma_real",
                    "gamma_imag",
                    "gamma_magnitude",
                    "gamma_phase_deg",
                    "smith_reference_z0_real_ohm",
                    "smith_reference_z0_imag_ohm",
                ]
            )
            for i in range(len(r.z_m)):
                z_local = complex(r.z_local_ohm[i])
                z_norm = complex(r.z_normalized[i])
                gamma = complex(r.gamma_local[i])
                w.writerow(
                    [
                        r.z_m[i],
                        float(np.abs(r.v_rms[i])),
                        float(np.abs(r.i_rms[i])),
                        r.p_real_w[i],
                        r.p_cond_w_per_m[i],
                        r.p_diel_w_per_m[i],
                        r.p_total_loss_w_per_m[i],
                        r.p_cumulative_loss_w[i],
                        z_local.real,
                        z_local.imag,
                        abs(z_local),
                        np.degrees(np.angle(z_local)),
                        z_norm.real,
                        z_norm.imag,
                        gamma.real,
                        gamma.imag,
                        abs(gamma),
                        np.degrees(np.angle(gamma)),
                        r.z0_reference.real,
                        r.z0_reference.imag,
                    ]
                )

    def export_smith_chart_image(self) -> None:
        if self.last_result is None:
            return
        freq = self._active_frequency_mhz()
        z0 = self.last_result.z0_reference
        default_name = f"smith_chart_{freq:.3f}MHz_{float(self.last_result.z_m[-1]):.3f}m_{z0.real:.2f}ohm.png"
        safe_name = (
            default_name.replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )
        path, _ = QFileDialog.getSaveFileName(self, "Export Smith Chart Image", safe_name, "PNG (*.png)")
        if not path:
            return
        self.smith_canvas.export_png(path)

    def export_sweep_csv(self) -> None:
        if self.last_sweep is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Sweep CSV", "z0_sweep.csv", "CSV (*.csv)")
        if not path:
            return
        s = self.last_sweep
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "Z0_ohm",
                    "CableLoss_W",
                    "CableLoss_pct",
                    "MaxVoltage_Vrms",
                    "MaxCurrent_Arms",
                    "VSWR",
                    "ChamberPower_W",
                    "Score",
                ]
            )
            for i in range(len(s.z0_ohm)):
                w.writerow(
                    [
                        s.z0_ohm[i],
                        s.cable_loss_w[i],
                        s.cable_loss_percent[i],
                        s.max_voltage_rms[i],
                        s.max_current_rms[i],
                        s.vswr[i],
                        s.chamber_power_w[i],
                        s.score[i],
                    ]
                )

    def _update_summary(self, r: CableAnalysisResult, system: SystemParameters) -> None:
        eta_match = (r.match_output_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        eta_cable = (r.chamber_power_w / r.match_output_power_w * 100.0) if r.match_output_power_w > 0 else 0.0
        eta_total = (r.chamber_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        ideal_note = ""
        if "IDEAL MATCHER" in r.match_output_power_source_label:
            ideal_note = "IDEAL MATCHER - NO MATCH LOSS ASSUMED\n\n"
        self.summary_label.setText(
            f"Reference planes:\n"
            f"Match Input -> Matcher -> Match Output/Cable Input (z=0) -> Chamber (z=L)\n\n"
            f"{ideal_note}"
            f"Match Input Power = {r.match_input_power_w:.3f} W\n"
            f"Match Output Power = {r.match_output_power_w:.3f} W [{r.match_output_power_source_label}]\n"
            f"Match Loss = {r.match_loss_w:.3f} W\n"
            f"Cable Dissipation Power = {r.cable_loss_w:.3f} W\n"
            f"Cable End Delivered Power = {r.chamber_power_w:.3f} W\n"
            f"Power Balance Error = {r.power_balance_error_w:.6f} W ({r.power_balance_error_percent:.4f}%)\n\n"
            f"Match Efficiency = {eta_match:.3f}%\n"
            f"Cable Efficiency = {eta_cable:.3f}%\n"
            f"Total Delivery Efficiency = {eta_total:.3f}%\n\n"
            f"Gamma_L = {abs(r.gamma_l):.4f} ∠ {r.gamma_phase_deg:.2f} deg, "
            f"VSWR = {r.vswr:.4f}, Return Loss = {r.return_loss_db:.2f} dB\n"
            f"Electrical length = {r.electrical_length_deg:.2f} deg\n\n"
            f"Endpoint consistency (from V/I)\n"
            f"z=0 Delta_Z = {r.endpoint_match_delta_ohm.real:+.4g}{r.endpoint_match_delta_ohm.imag:+.4g}j ohm "
            f"(rel err {r.endpoint_match_rel_error:.3e})\n"
            f"z=L Delta_Z = {r.endpoint_end_delta_ohm.real:+.4g}{r.endpoint_end_delta_ohm.imag:+.4g}j ohm "
            f"(rel err {r.endpoint_end_rel_error:.3e})"
        )
