from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDoubleValidator, QIcon, QKeySequence, QPixmap
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
    QScrollArea,
    QSplitter,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from analysis.cable_analysis import CURRENT_EPSILON_A, CableAnalysisResult, analyze_cable
from analysis.optimization import OptimizationWeights, SweepResult, sweep_z0
from analysis.stress_comparison import RFStressComparisonResult, build_stress_comparison_result
from gui.thermal_plot import draw_lumped_temperature_plot
from gui.help_window import HelpWindow
from gui.plots import MplCanvas
from gui.cable_spec_panel import CableSpecificationPanel
from gui.focus_widgets import FocusWheelComboBox
from gui.smith_chart import SmithChartCanvas
from models.cable_presets import AA11459_ATTENUATION_DB_PER_100FT, AA11459_POWER_RATING_KW, interpolate_log_frequency
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters, CableSpecification
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.connector_parameters import ConnectorModelMode, ConnectorParameters, HistoricalCoordinateReference
from models.operating_points import find_operating_point
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, MatcherModelMode, SystemParameters
from models.validation import validate_inputs
from rf.coax import inner_diameter_for_target_z0, z0_from_geometry
from rf.smith_math import vswr_from_gamma
from thermal.engine import available_heating_sources, run_thermal_analysis
from thermal.models import ThermalHeatingSource, ThermalInputs, ThermalModel, ThermalRegionSelection, ThermalRequest
from thermal.validation import validate_thermal_request
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
        self._ui_ready = False
        self.setWindowTitle(f"RF Match-to-Chamber Cable Engineering Analyzer - {APP_VERSION}")
        icon_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        icon_path = icon_base / "smithchart.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1500, 980)
        self._apply_theme()
        self.last_result: CableAnalysisResult | None = None
        self.last_stress_result: RFStressComparisonResult | None = None
        self.last_thermal_result = None
        self.last_sweep: SweepResult | None = None
        self.help_window: HelpWindow | None = None
        self._measurement_row_positions: list[float] = []
        self._analysis_call_count = 0
        self._last_invalid_widget: QWidget | None = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self._build_system_tab()
        self._build_cable_tab()
        self._build_distribution_tab()
        self._build_thermal_tab()
        self._build_optimization_tab()
        self._build_coax_tab()
        self._build_summary_tab()
        self._build_feed_image_tab("Top feed", "Top_feed.png", "top_feed_image_label")
        self._build_feed_image_tab("Bottom feed", "Bottom_feed.png", "bottom_feed_image_label")
        self._create_menu_bar()

        self._ui_ready = True
        self.recalculate()

    def _create_menu_bar(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        analysis_menu = menu.addMenu("Analysis")
        window_menu = menu.addMenu("Window")
        help_menu = menu.addMenu("Help")

        export_profile_action = QAction("Export Profile CSV", self)
        export_profile_action.triggered.connect(self.export_profile_csv)
        file_menu.addAction(export_profile_action)
        self.export_profile_action = export_profile_action

        export_smith_action = QAction("Export Smith Chart Image", self)
        export_smith_action.triggered.connect(self.export_smith_chart_image)
        file_menu.addAction(export_smith_action)
        self.export_smith_action = export_smith_action

        export_stress_action = QAction("Export RF Stress Dashboard Image", self)
        export_stress_action.triggered.connect(self.export_rf_stress_dashboard_image)
        file_menu.addAction(export_stress_action)
        self.export_stress_action = export_stress_action

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setShortcutContext(Qt.ApplicationShortcut)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        self.exit_action = exit_action

        run_analysis_action = QAction("Run Analysis", self)
        run_analysis_action.triggered.connect(self.recalculate)
        analysis_menu.addAction(run_analysis_action)
        self.run_analysis_action = run_analysis_action

        self.window_tab_actions: list[QAction] = []
        for tab_index in range(self.tabs.count()):
            tab_name = self.tabs.tabText(tab_index)
            tab_action = QAction(tab_name, self)
            tab_action.triggered.connect(lambda _checked=False, idx=tab_index: self.tabs.setCurrentIndex(idx))
            window_menu.addAction(tab_action)
            self.window_tab_actions.append(tab_action)

        help_action = QAction("RF Cable Optimizer Help", self)
        help_action.triggered.connect(self._show_help_window)
        help_menu.addAction(help_action)
        self.help_action = help_action

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
        self.about_action = about_action

    def _show_help_window(self) -> None:
        if self.help_window is None:
            self.help_window = HelpWindow(self)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About RF Cable Optimizer",
            "RF Match-to-Chamber Cable Engineering Analyzer\n"
            f"Version: {APP_VERSION}\n"
            "Author: Victor Huang\n"
            "Date: 2026-10-31\n"
            "Company: ASM International\n\n"
            "PySide6 desktop tool for RF cable analysis, Smith Chart visualization, "
            "distributed voltage/current/power profiling, and engineering diagnostics.",
        )

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
        self.operating_point_combo.setCurrentText("1200 W")
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
        self.p_match_in_w = _line_edit("1200", "Real RF power at matcher input reference plane [W].")
        self.z_gen = _line_edit("50", "Source impedance ZS real part [ohm].")
        self.z_gen_x = _line_edit("0", "Source impedance ZS imaginary part [ohm].")
        self.p_match_out = _line_edit("", "Calculated Match Output real power at match-output reference plane [W].")
        self.p_match_out.setReadOnly(True)
        self.match_eff = _line_edit("70", "Match efficiency [%]. Not used in USER S-PARAMETERS calculated mode.")
        self.power_source = QComboBox()
        self.power_source.addItems([s.value for s in MatchOutputPowerSource])
        self.power_source.currentTextChanged.connect(self._on_matcher_mode_changed)
        self.zm_r = _line_edit("3", "Historical / operating-point load resistance RL [ohm].")
        self.zm_x = _line_edit("19", "Historical / operating-point load reactance XL [ohm].")
        self.vpp = _line_edit("", "Measured Vpp at Match Output [V]. Optional.")
        self.s_mode_lbl = QLabel("Matcher Model: IDEAL")
        self.matcher_mode = QComboBox()
        self.matcher_mode.addItems([m.value for m in MatcherModelMode])
        self.matcher_mode.currentTextChanged.connect(self._on_matcher_mode_changed)
        self.s11r = _line_edit("-0.817", "S11 real")
        self.s11i = _line_edit("0.414", "S11 imag")
        self.s12r = _line_edit("-0.0561", "S12 real")
        self.s12i = _line_edit("0.343", "S12 imag")
        self.s21r = _line_edit("-0.0559", "S21 real")
        self.s21i = _line_edit("0.343", "S21 imag")
        self.s22r = _line_edit("-0.670", "S22 real")
        self.s22i = _line_edit("-0.625", "S22 imag")
        self.s_zref = _line_edit("50", "S-parameter reference impedance [ohm]")
        self.s_freq = _line_edit("60", "S-parameter frequency [MHz]")
        sys.addRow("Frequency [MHz]", self.freq_mhz)
        sys.addRow("Match Input Power [W]", self.p_match_in_w)
        sys.addRow("Source Impedance ZS [ohm] (R, X)", self._pair_widget(self.z_gen, self.z_gen_x))
        sys.addRow("Calculated Match Output Power [W]", self.p_match_out)
        sys.addRow("Match Output Power Mode", self.power_source)
        sys.addRow("Match Efficiency [%]", self.match_eff)
        sys.addRow("Load R, RL [ohm]", self.zm_r)
        sys.addRow("Load X, XL [ohm]", self.zm_x)
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
        self.chamber_mode.setCurrentText(ChamberImpedanceMode.MEASURED_OR_USER_LOAD.value)
        self.chamber_mode.currentTextChanged.connect(self._toggle_chamber_fields)
        self.zl_r = _line_edit("", "Chamber impedance real part [ohm].")
        self.zl_x = _line_edit("", "Chamber impedance imaginary part [ohm].")
        chamber.addRow("Load Impedance Mode", self.chamber_mode)
        chamber.addRow("Downstream Chamber R [ohm]", self.zl_r)
        chamber.addRow("Downstream Chamber X [ohm]", self.zl_x)

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
        self._on_matcher_mode_changed(self.matcher_mode.currentText())

    def _build_cable_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Cable Analysis")
        layout = QVBoxLayout(tab)
        cards = QGroupBox("Power Flow")
        cg = QGridLayout(cards)
        self.card_labels: dict[str, QLabel] = {}
        for idx, (title, target, color) in enumerate(
            [
                ("MATCH INPUT POWER", "match_input", "#3b82f6"),
                ("MATCH OUTPUT POWER", "match_output", "#22d3ee"),
                ("CONNECTOR DISSIPATION", "connector_loss", "#a855f7"),
                ("FLEXIBLE CABLE INPUT POWER", "flex_input", "#60a5fa"),
                ("FLEXIBLE CABLE DISSIPATION", "cable_loss", "#ef4444"),
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
        self.current_comparison_label = QLabel(
            "Matched-Line Current: - Arms\n"
            "Simulated Maximum: - Arms\n"
            "Current Enhancement: - x"
        )
        self.current_comparison_label.setStyleSheet("color:#e9d5ff;")
        self.min_impedance_point_label = QLabel("LOWEST-IMPEDANCE POINT\nNot Available")
        self.min_impedance_point_label.setStyleSheet("color:#fdba74; border:1px solid #334155; border-radius:8px; padding:8px;")
        self.max_current_point_label = QLabel("MAXIMUM-CURRENT POINT\nNot Available")
        self.max_current_point_label.setStyleSheet("color:#c4b5fd; border:1px solid #334155; border-radius:8px; padding:8px;")

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color:#f87171;")
        self.live_metrics_label = QLabel(self._format_live_metrics(None, None))
        self.live_metrics_label.setStyleSheet("color:#93c5fd;")

        self.power_flow_scroll_area = QScrollArea()
        self.power_flow_scroll_area.setWidgetResizable(True)
        self.power_flow_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.power_flow_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.power_flow_scroll_area.setWidget(cards)
        self.power_flow_scroll_area.setMinimumHeight(220)
        layout.addWidget(self.power_flow_scroll_area, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        sticky_header = QGroupBox("Cable Preset / Run Analysis")
        sticky_form = QFormLayout(sticky_header)
        self.cable_preset = FocusWheelComboBox()
        self.cable_preset.addItems(["DX7 Existing Cable", "COMCRAFT AA-11459", "Custom"])
        self.cable_preset.currentTextChanged.connect(self._on_cable_preset_changed)
        self.run_btn_cable = QPushButton("Run Analysis")
        self.run_btn_cable.clicked.connect(self.recalculate)
        sticky_form.addRow("Cable Preset", self.cable_preset)
        sticky_form.addRow(self.run_btn_cable)
        left_layout.addWidget(sticky_header)

        self.cable_spec_scroll_area = QScrollArea()
        self.cable_spec_scroll_area.setWidgetResizable(False)
        self.cable_spec_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cable_spec_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cable_spec_panel = CableSpecificationPanel(_line_edit, _text_line_edit, self)
        self.cable_spec_panel.setMinimumWidth(500)
        self.cable_spec_scroll_area.setWidget(self.cable_spec_panel)
        left_layout.addWidget(self.cable_spec_scroll_area, 1)
        left_layout.addWidget(self.warning_label)
        left_layout.addWidget(self.live_metrics_label)

        # Backward-compatible widget attributes used across existing logic/tests.
        self.cable_mode = self.cable_spec_panel.cable_mode
        self.z0 = self.cable_spec_panel.z0
        self.length = self.cable_spec_panel.length
        self.vf = self.cable_spec_panel.vf
        self.att = self.cable_spec_panel.att
        self.att_unit = self.cable_spec_panel.att_unit
        self.att_mode = self.cable_spec_panel.att_mode
        self.att_info = self.cable_spec_panel.att_info
        self.rp = self.cable_spec_panel.rp
        self.lp = self.cable_spec_panel.lp
        self.gp = self.cable_spec_panel.gp
        self.cp = self.cable_spec_panel.cp
        self.spec_family = self.cable_spec_panel.spec_family
        self.spec_model = self.cable_spec_panel.spec_model
        self.spec_part = self.cable_spec_panel.spec_part
        self.spec_assembly = self.cable_spec_panel.spec_assembly
        self.spec_outer_d = self.cable_spec_panel.spec_outer_d
        self.spec_center_d = self.cable_spec_panel.spec_center_d
        self.spec_dielectric_d = self.cable_spec_panel.spec_dielectric_d
        self.spec_center_type = self.cable_spec_panel.spec_center_type
        self.spec_dielectric = self.cable_spec_panel.spec_dielectric
        self.spec_outer_shield = self.cable_spec_panel.spec_outer_shield
        self.spec_jacket = self.cable_spec_panel.spec_jacket
        self.spec_temp_range = self.cable_spec_panel.spec_temp_range
        self.spec_bend_radius = self.cable_spec_panel.spec_bend_radius
        self.spec_d_inner = self.cable_spec_panel.spec_d_inner
        self.spec_status = self.cable_spec_panel.spec_status
        self.spec_field_status = self.cable_spec_panel.spec_field_status
        self.power_rating_panel = self.cable_spec_panel.power_rating_panel

        connector_panel = self.cable_spec_panel.connector_panel
        self.include_connector = connector_panel.include_connector
        self.connector_mode = connector_panel.connector_mode
        self.connector_model_name = connector_panel.connector_model_name
        self.connector_length = connector_panel.connector_length
        self.connector_z0 = connector_panel.connector_z0
        self.connector_vf = connector_panel.connector_vf
        self.connector_att = connector_panel.connector_att
        self.connector_att_unit = connector_panel.connector_att_unit
        self.connector_ideal_lossless = connector_panel.connector_ideal_lossless
        self.connector_source_status = connector_panel.connector_source_status
        self.connector_notes = connector_panel.connector_notes
        self.connector_metrics = connector_panel.connector_metrics

        self.historical_damage_start = self.cable_spec_panel.historical_damage_start
        self.historical_damage_end = self.cable_spec_panel.historical_damage_end
        self.historical_coordinate_reference = self.cable_spec_panel.historical_coordinate_reference

        self.cable_mode.currentTextChanged.connect(self._toggle_rlgc_fields)
        self.att_mode.currentTextChanged.connect(self._on_attenuation_mode_changed)
        self.cable_spec_panel.parameterChanged.connect(self.recalculate)
        self.cable_spec_panel.connectorChanged.connect(self.recalculate)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.smith_canvas = SmithChartCanvas(width=7.0, height=7.0, dpi=100)
        self.smith_canvas.setMinimumHeight(320)
        self.smith_canvas.probePositionSelected.connect(self._on_smith_probe_selected)
        self.smith_canvas.set_hover_formatter(self._format_smith_hover)

        critical_cards = QGroupBox("Critical-Point Cards")
        critical_layout = QVBoxLayout(critical_cards)
        critical_layout.addWidget(self.stress_metrics_label)
        critical_layout.addWidget(self.current_comparison_label)
        critical_layout.addWidget(self.min_impedance_point_label)
        critical_layout.addWidget(self.max_current_point_label)
        self.critical_cards_scroll_area = QScrollArea()
        self.critical_cards_scroll_area.setWidgetResizable(True)
        self.critical_cards_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.critical_cards_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.critical_cards_scroll_area.setWidget(critical_cards)

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
        self.smith_summary_scroll_area = QScrollArea()
        self.smith_summary_scroll_area.setWidgetResizable(True)
        self.smith_summary_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.smith_summary_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.smith_summary_scroll_area.setWidget(smith_cards)

        probe_box = QGroupBox("Probe Impedance Details")
        probe_layout = QVBoxLayout(probe_box)
        self.probe_impedance_label = QLabel("Probe impedance details are available after analysis.")
        self.probe_impedance_label.setWordWrap(True)
        self.probe_impedance_label.setStyleSheet("color:#f8fafc;")
        probe_layout.addWidget(self.probe_impedance_label)
        self.probe_impedance_scroll_area = QScrollArea()
        self.probe_impedance_scroll_area.setWidgetResizable(True)
        self.probe_impedance_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.probe_impedance_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.probe_impedance_scroll_area.setWidget(probe_box)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.addWidget(self.critical_cards_scroll_area, 1)
        overview_layout.addWidget(self.probe_impedance_scroll_area, 1)

        smith_tab = QWidget()
        smith_layout = QVBoxLayout(smith_tab)
        smith_layout.setContentsMargins(0, 0, 0, 0)
        right_inner_splitter = QSplitter(Qt.Horizontal)
        right_inner_splitter.addWidget(self.smith_summary_scroll_area)
        right_inner_splitter.addWidget(self.smith_canvas)
        right_inner_splitter.setStretchFactor(0, 1)
        right_inner_splitter.setStretchFactor(1, 1)
        right_inner_splitter.setSizes([420, 520])
        smith_layout.addWidget(right_inner_splitter, 1)

        self.stress_tab = self._build_rf_stress_tab()
        self.cable_analysis_subtabs = QTabWidget()
        self.cable_analysis_subtabs.addTab(overview_tab, "Overview")
        self.cable_analysis_subtabs.addTab(self.stress_tab, "RF Stress Comparison")
        self.cable_analysis_subtabs.addTab(smith_tab, "Smith Chart")
        animation_corner = QWidget()
        animation_corner_layout = QHBoxLayout(animation_corner)
        animation_corner_layout.setContentsMargins(0, 0, 0, 0)
        animation_corner_layout.addWidget(QLabel("Enable Animation"))
        self.enable_stress_animation = QComboBox()
        self.enable_stress_animation.addItems(["ON", "OFF"])
        animation_corner_layout.addWidget(self.enable_stress_animation)
        self.cable_analysis_subtabs.setCornerWidget(animation_corner, Qt.TopRightCorner)
        self.cable_analysis_subtabs.setCurrentIndex(2)
        right_layout.addWidget(self.cable_analysis_subtabs, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 35)
        splitter.setStretchFactor(1, 65)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(420)
        splitter.setSizes([380, 920])
        layout.addWidget(splitter, 2)
        export_row = QHBoxLayout()
        self.export_smith_btn = QPushButton("Export Smith Chart Image")
        self.export_smith_btn.clicked.connect(self.export_smith_chart_image)
        self.export_stress_btn = QPushButton("Export RF Stress Dashboard Image")
        self.export_stress_btn.clicked.connect(self.export_rf_stress_dashboard_image)
        export_row.addWidget(self.export_smith_btn)
        export_row.addWidget(self.export_stress_btn)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        self._toggle_rlgc_fields()
        self.cable_preset.setCurrentText("COMCRAFT AA-11459")

    @staticmethod
    def _pair_widget(left: QWidget, right: QWidget) -> QWidget:
        wrapper = QWidget()
        l = QHBoxLayout(wrapper)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(left)
        l.addWidget(right)
        return wrapper

    def _build_rf_stress_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        stress_scroll = QScrollArea()
        stress_scroll.setWidgetResizable(True)
        stress_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        stress_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel("RF STRESS COMPARISON"))

        columns = QGroupBox("Reference Comparison")
        columns_layout = QGridLayout(columns)
        self.manufacturer_ref_label = QLabel("MANUFACTURER REFERENCE\n-")
        self.matched_ref_label = QLabel("IDEAL MATCHED REFERENCE\n-")
        self.actual_ref_label = QLabel("ACTUAL DISTRIBUTED RESULT\n-")
        self.manufacturer_ref_label.setStyleSheet("color:#93c5fd; border:1px solid #334155; border-radius:8px; padding:10px;")
        self.matched_ref_label.setStyleSheet("color:#4ade80; border:1px solid #334155; border-radius:8px; padding:10px;")
        self.actual_ref_label.setStyleSheet("color:#fb923c; border:1px solid #334155; border-radius:8px; padding:10px;")
        columns_layout.addWidget(self.manufacturer_ref_label, 0, 0)
        columns_layout.addWidget(self.matched_ref_label, 0, 1)
        columns_layout.addWidget(self.actual_ref_label, 0, 2)
        layout.addWidget(columns)

        cards = QGroupBox("Enhancement Summary")
        cards_layout = QGridLayout(cards)
        self.current_enhancement_card = QPushButton("CURRENT ENHANCEMENT\nNOT AVAILABLE")
        self.current_enhancement_card.clicked.connect(self._jump_probe_to_max_current)
        self.current_enhancement_card.setStyleSheet(
            "color:#fb923c; border:1px solid #7c2d12; border-radius:10px; padding:10px; font-weight:bold;"
        )
        self.current_enhancement_card.setToolTip(
            "Ratio of the maximum distributed RMS current to the ideal matched RMS current at the same Match Output Power."
        )
        self.i2_stress_card = QLabel("LOCAL I-SQUARED STRESS INDEX\nNOT AVAILABLE")
        self.i2_stress_card.setStyleSheet(
            "color:#f87171; border:1px solid #7f1d1d; border-radius:10px; padding:10px; font-weight:bold;"
        )
        self.i2_stress_card.setToolTip(
            "Square of the current-enhancement ratio. This is a relative local conductor-current stress indicator, not a temperature prediction or manufacturer derating factor."
        )
        self.voltage_enhancement_card = QLabel("VOLTAGE ENHANCEMENT\nNOT AVAILABLE")
        self.voltage_enhancement_card.setStyleSheet(
            "color:#22d3ee; border:1px solid #155e75; border-radius:10px; padding:10px; font-weight:bold;"
        )
        self.voltage_enhancement_card.setToolTip(
            "Ratio of maximum distributed RMS voltage to ideal matched RMS voltage at the same power."
        )
        self.interpretation_label = QLabel("Run analysis to populate RF stress comparison.")
        self.interpretation_label.setWordWrap(True)
        self.interpretation_label.setStyleSheet("color:#e2e8f0;")
        cards_layout.addWidget(self.current_enhancement_card, 0, 0)
        cards_layout.addWidget(self.i2_stress_card, 0, 1)
        cards_layout.addWidget(self.voltage_enhancement_card, 0, 2)
        cards_layout.addWidget(self.interpretation_label, 1, 0, 1, 3)
        layout.addWidget(cards)

        actions_row = QHBoxLayout()
        self.open_thermal_hotspot_btn = QPushButton("OPEN THERMAL ANALYSIS AT HOT SPOT")
        self.open_thermal_hotspot_btn.clicked.connect(self._open_thermal_analysis_at_hot_spot)
        actions_row.addWidget(self.open_thermal_hotspot_btn)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.stress_bar_canvas = MplCanvas(width=12, height=4.6, dpi=100)
        self.stress_map_canvas = MplCanvas(width=12, height=9, dpi=100)
        self.stress_bar_canvas.setMinimumHeight(260)
        self.stress_map_canvas.setMinimumHeight(720)
        self._stress_bar_click_cid = self.stress_bar_canvas.mpl_connect("button_press_event", self._on_stress_bar_click)
        self._stress_map_click_cid = self.stress_map_canvas.mpl_connect("button_press_event", self._on_stress_map_click)
        layout.addWidget(self.stress_bar_canvas, 1)
        layout.addWidget(self.stress_map_canvas, 2)

        details = QGroupBox("Critical Point Details")
        details_layout = QHBoxLayout(details)
        self.max_current_detail_label = QLabel("MAXIMUM CURRENT POINT\nNot Available")
        self.min_impedance_detail_label = QLabel("LOWEST-IMPEDANCE POINT\nNot Available")
        self.max_current_detail_label.setStyleSheet("color:#fdba74; border:1px solid #334155; border-radius:8px; padding:8px;")
        self.min_impedance_detail_label.setStyleSheet("color:#fda4af; border:1px solid #334155; border-radius:8px; padding:8px;")
        details_layout.addWidget(self.max_current_detail_label, 1)
        details_layout.addWidget(self.min_impedance_detail_label, 1)
        layout.addWidget(details)
        layout.addStretch(1)
        stress_scroll.setWidget(content)
        tab_layout.addWidget(stress_scroll)
        return tab

    def _build_distribution_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Voltage / Current Distribution")
        layout = QVBoxLayout(tab)
        probe_row = QHBoxLayout()
        self.probe_pos = _line_edit("0.15", "Probe position from Match Output [m].")
        self.probe_pos.setMaximumWidth(90)
        self.probe_reference = FocusWheelComboBox()
        self.probe_reference.addItems(["From Match Output", "From Flexible Cable Start"])
        self.probe_reference.currentTextChanged.connect(self._on_probe_reference_changed)
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
        probe_row.addWidget(self.probe_reference)
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

    def _build_thermal_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Thermal Analysis")
        layout = QVBoxLayout(tab)

        top_splitter = QSplitter(Qt.Horizontal)

        source_box = QGroupBox("Thermal Heating Source")
        source_form = QFormLayout(source_box)
        self.thermal_model = QComboBox()
        self.thermal_model.addItem(ThermalModel.LUMPED_RC.value, ThermalModel.LUMPED_RC)
        self.thermal_model.addItem(ThermalModel.DISTRIBUTED_1D.value, ThermalModel.DISTRIBUTED_1D)
        self.thermal_model.setCurrentIndex(0)
        self.thermal_source_combo = QComboBox()
        self.thermal_rf_heating_power = QLabel("RF-DERIVED HEATING POWER\n- W")
        self.thermal_rf_heating_power.setStyleSheet("color:#f97316; border:1px solid #334155; border-radius:8px; padding:8px;")
        self.thermal_heating_source_origin = QLabel("Source: RF Cable Analysis")
        self.thermal_source_detail = QLabel("Thermal source: -")
        self.thermal_connector_status = QLabel("Connector Dissipation: NOT MODELED")
        self.thermal_connector_status.setStyleSheet("color:#facc15;")
        source_form.addRow("Thermal Model", self.thermal_model)
        source_form.addRow("Thermal Heating Source", self.thermal_source_combo)
        source_form.addRow(self.thermal_rf_heating_power)
        source_form.addRow(self.thermal_heating_source_origin)
        source_form.addRow(self.thermal_source_detail)
        source_form.addRow(self.thermal_connector_status)
        distributed_note = QLabel("Distributed 1D thermal model is prepared for future release and not yet implemented.")
        distributed_note.setWordWrap(True)
        distributed_note.setStyleSheet("color:#93c5fd;")
        source_form.addRow(distributed_note)
        self.thermal_source_scroll_area = QScrollArea()
        self.thermal_source_scroll_area.setWidgetResizable(True)
        self.thermal_source_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thermal_source_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thermal_source_scroll_area.setWidget(source_box)
        top_splitter.addWidget(self.thermal_source_scroll_area)

        params_box = QGroupBox("Lumped Thermal RC Parameters")
        params_form = QFormLayout(params_box)
        self.thermal_rth = _line_edit("1.5", "Thermal resistance to ambient [K/W].")
        self.thermal_cth = _line_edit("300", "Thermal capacitance [J/K].")
        self.thermal_ambient_temp = _line_edit("20", "Ambient temperature [deg C].")
        self.thermal_initial_temp = _line_edit("20", "Initial temperature [deg C].")
        self.thermal_target_temp = _line_edit("80", "Exceed temperature limit [deg C].")
        self.thermal_duration_s = _line_edit("900", "Thermal simulation duration [s].")
        self.thermal_time_step_s = _line_edit("1", "Thermal simulation time step [s].")
        self.thermal_hotspot_start_m = _line_edit("0.10", "Hot-spot start position from Match Output [m].")
        self.thermal_hotspot_end_m = _line_edit("0.20", "Hot-spot end position from Match Output [m].")
        self.thermal_section_combo = QComboBox()
        self.thermal_user_power_w = _line_edit("50", "User-entered heating power for what-if analysis [W].")

        self.thermal_rth_status = QLabel("ASSUMED / INITIAL ESTIMATE")
        self.thermal_cth_status = QLabel("ASSUMED / INITIAL ESTIMATE")
        self.thermal_rth_status.setStyleSheet("color:#facc15;")
        self.thermal_cth_status.setStyleSheet("color:#facc15;")

        params_form.addRow("Thermal Resistance R_th [K/W]", self._pair_widget(self.thermal_rth, self.thermal_rth_status))
        params_form.addRow("Thermal Capacitance C_th [J/K]", self._pair_widget(self.thermal_cth, self.thermal_cth_status))
        params_form.addRow("Ambient Temperature [deg C]", self.thermal_ambient_temp)
        params_form.addRow("Initial Temperature [deg C]", self.thermal_initial_temp)
        params_form.addRow("Exceed Temp limit [deg C]", self.thermal_target_temp)
        params_form.addRow("Simulation Duration [s]", self.thermal_duration_s)
        params_form.addRow("Simulation Time Step [s]", self.thermal_time_step_s)
        params_form.addRow("Hot-Spot Start [m]", self.thermal_hotspot_start_m)
        params_form.addRow("Hot-Spot End [m]", self.thermal_hotspot_end_m)
        params_form.addRow("Selected RF Section", self.thermal_section_combo)
        params_form.addRow("User-Entered Heating Power [W]", self.thermal_user_power_w)
        self.thermal_params_scroll_area = QScrollArea()
        self.thermal_params_scroll_area.setWidgetResizable(True)
        self.thermal_params_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thermal_params_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thermal_params_scroll_area.setWidget(params_box)
        top_splitter.addWidget(self.thermal_params_scroll_area)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([700, 900])
        layout.addWidget(top_splitter, 0)

        self.thermal_estimate_warning = QLabel(
            "Initial Rth and Cth values are engineering estimates for exploratory simulation. "
            "Replace or calibrate them with measured thermal data for quantitative temperature prediction."
        )
        self.thermal_estimate_warning.setWordWrap(True)
        self.thermal_estimate_warning.setStyleSheet("color:#f87171;")
        layout.addWidget(self.thermal_estimate_warning)

        self.thermal_summary_label = QLabel("Run RF analysis to compute thermal response.")
        self.thermal_summary_label.setStyleSheet("color:#93c5fd;")
        self.thermal_summary_label.setWordWrap(True)
        layout.addWidget(self.thermal_summary_label)

        self.thermal_canvas = MplCanvas(width=12, height=5, dpi=100)
        layout.addWidget(self.thermal_canvas, 1)

        self.thermal_model.currentIndexChanged.connect(self._update_thermal_analysis)
        self.thermal_source_combo.currentIndexChanged.connect(self._on_thermal_source_changed)
        self.thermal_section_combo.currentIndexChanged.connect(self._update_thermal_analysis)
        for widget in (
            self.thermal_rth,
            self.thermal_cth,
            self.thermal_ambient_temp,
            self.thermal_initial_temp,
            self.thermal_target_temp,
            self.thermal_duration_s,
            self.thermal_time_step_s,
            self.thermal_hotspot_start_m,
            self.thermal_hotspot_end_m,
            self.thermal_user_power_w,
        ):
            widget.editingFinished.connect(self._update_thermal_analysis)

    def _refresh_thermal_source_options(self, result: CableAnalysisResult) -> None:
        previous_text = self.thermal_source_combo.currentText()
        available = [item for item in available_heating_sources(result) if item.enabled]
        self.thermal_source_combo.blockSignals(True)
        self.thermal_source_combo.clear()
        for item in available:
            self.thermal_source_combo.addItem(item.source.value, item.source)
        if previous_text:
            index = self.thermal_source_combo.findText(previous_text)
            if index >= 0:
                self.thermal_source_combo.setCurrentIndex(index)
        if self.thermal_source_combo.currentIndex() < 0:
            default_text = ThermalHeatingSource.FLEXIBLE_CABLE_TOTAL_DISSIPATION.value
            default_index = self.thermal_source_combo.findText(default_text)
            self.thermal_source_combo.setCurrentIndex(default_index if default_index >= 0 else 0)
        self.thermal_source_combo.blockSignals(False)
        if result.connector_loss_w is None:
            self.thermal_connector_status.setText("Connector Dissipation: NOT MODELED")
        else:
            self.thermal_connector_status.setText(f"Connector Dissipation: {result.connector_loss_w:.2f} W (RF-derived)")

        section_names = sorted({str(name) for name in result.section_name})
        previous_section = self.thermal_section_combo.currentText()
        self.thermal_section_combo.blockSignals(True)
        self.thermal_section_combo.clear()
        self.thermal_section_combo.addItems(section_names)
        section_idx = self.thermal_section_combo.findText(previous_section)
        if section_idx >= 0:
            self.thermal_section_combo.setCurrentIndex(section_idx)
        self.thermal_section_combo.blockSignals(False)

    def _on_thermal_source_changed(self) -> None:
        source = self._selected_thermal_source()
        user_input = source == ThermalHeatingSource.USER_ENTERED_HEATING_POWER
        hotspot = source == ThermalHeatingSource.SELECTED_HOT_SPOT_REGION
        section = source == ThermalHeatingSource.SELECTED_RF_SECTION
        self.thermal_user_power_w.setEnabled(user_input)
        self.thermal_hotspot_start_m.setEnabled(hotspot)
        self.thermal_hotspot_end_m.setEnabled(hotspot)
        self.thermal_section_combo.setEnabled(section)
        self._update_thermal_analysis()

    def _selected_thermal_source(self) -> ThermalHeatingSource:
        data = self.thermal_source_combo.currentData()
        if isinstance(data, ThermalHeatingSource):
            return data
        return ThermalHeatingSource(self.thermal_source_combo.currentText())

    def _collect_thermal_request(self) -> ThermalRequest:
        model_data = self.thermal_model.currentData()
        model = model_data if isinstance(model_data, ThermalModel) else ThermalModel.LUMPED_RC
        source = self._selected_thermal_source()
        thermal_inputs = ThermalInputs(
            r_th_k_per_w=float(self.thermal_rth.text()),
            c_th_j_per_k=float(self.thermal_cth.text()),
            ambient_temp_c=float(self.thermal_ambient_temp.text()),
            initial_temp_c=float(self.thermal_initial_temp.text()),
            target_temp_c=float(self.thermal_target_temp.text()),
            duration_s=float(self.thermal_duration_s.text()),
            time_step_s=float(self.thermal_time_step_s.text()),
        )
        region = ThermalRegionSelection(
            start_m=float(self.thermal_hotspot_start_m.text()),
            end_m=float(self.thermal_hotspot_end_m.text()),
            section_name=self.thermal_section_combo.currentText().strip() or None,
        )
        user_power: float | None = None
        if self.thermal_user_power_w.text().strip():
            user_power = float(self.thermal_user_power_w.text())
        return ThermalRequest(
            model=model,
            heating_source=source,
            thermal_inputs=thermal_inputs,
            region=region,
            user_entered_heating_power_w=user_power,
        )

    def _reset_thermal_output(self, message: str) -> None:
        self.last_thermal_result = None
        if not hasattr(self, "thermal_summary_label"):
            return
        self.thermal_summary_label.setText(message)
        self.thermal_rf_heating_power.setText("RF-DERIVED HEATING POWER\n- W")
        self.thermal_heating_source_origin.setText("Source: RF Cable Analysis")
        self.thermal_source_detail.setText("Thermal source: -")
        self.thermal_canvas.figure.clear()
        self.thermal_canvas.draw_idle()

    def _update_thermal_analysis(self) -> None:
        if not getattr(self, "_ui_ready", False):
            return
        if self.last_result is None:
            self._reset_thermal_output("Run RF analysis to compute thermal response.")
            return
        result = self.last_result
        self._refresh_thermal_source_options(result)
        source = self._selected_thermal_source()
        self.thermal_user_power_w.setEnabled(source == ThermalHeatingSource.USER_ENTERED_HEATING_POWER)
        self.thermal_hotspot_start_m.setEnabled(source == ThermalHeatingSource.SELECTED_HOT_SPOT_REGION)
        self.thermal_hotspot_end_m.setEnabled(source == ThermalHeatingSource.SELECTED_HOT_SPOT_REGION)
        self.thermal_section_combo.setEnabled(source == ThermalHeatingSource.SELECTED_RF_SECTION)
        try:
            request = self._collect_thermal_request()
            validation = validate_thermal_request(request, result)
            if not validation.valid:
                self._reset_thermal_output("Thermal Input Required:\n" + "\n".join(validation.errors))
                return
            thermal_result = run_thermal_analysis(request, result)
            self.last_thermal_result = thermal_result
            if thermal_result.heating.rf_derived:
                self.thermal_rf_heating_power.setText(f"RF-DERIVED HEATING POWER\n{thermal_result.heating.power_w:.2f} W")
            else:
                self.thermal_rf_heating_power.setText(f"USER-ENTERED HEATING POWER\n{thermal_result.heating.power_w:.2f} W")
            self.thermal_heating_source_origin.setText(f"Source: {thermal_result.heating.source_origin_label}")
            self.thermal_source_detail.setText(f"Thermal source: {thermal_result.heating.source_label}")
            crossing = (
                f"{thermal_result.target_crossing_time_s:.1f} s"
                if thermal_result.target_crossing_time_s is not None
                else "Not reached in simulation window"
            )
            warning_text = ("\n" + "\n".join(thermal_result.warnings)) if thermal_result.warnings else ""
            self.thermal_summary_label.setText(
                f"Model: {thermal_result.model.value}\n"
                f"Thermal time constant tau = {thermal_result.thermal_time_constant_s:.2f} s\n"
                f"Steady-state temperature = {thermal_result.steady_state_temp_c:.2f} deg C\n"
                f"Target crossing time = {crossing}{warning_text}"
            )
            draw_lumped_temperature_plot(self.thermal_canvas, thermal_result, request.thermal_inputs.target_temp_c)
        except NotImplementedError as exc:
            self._reset_thermal_output(str(exc))
        except Exception as exc:
            self._reset_thermal_output(f"Thermal Input Required / Error: {exc}")

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

    def _build_feed_image_tab(self, tab_title: str, image_name: str, label_attr_name: str) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, tab_title)
        layout = QVBoxLayout(tab)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border:1px solid #334155;")

        image_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        image_path = image_base / image_name
        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                image_label.setPixmap(pixmap)
            else:
                image_label.setText(f"Unable to load image: {image_name}")
        else:
            image_label.setText(f"Image not found: {image_name}")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(image_label)
        layout.addWidget(scroll, 1)

        setattr(self, label_attr_name, image_label)

    def _toggle_rlgc_fields(self) -> None:
        use_rlgc = self.cable_mode.currentText() == CableModelMode.RLGC.value
        for w in (self.rp, self.lp, self.gp, self.cp):
            w.setEnabled(use_rlgc)
        self.att.setEnabled(not use_rlgc)
        self.att_unit.setEnabled(not use_rlgc)
        self.att_mode.setEnabled(not use_rlgc)

    def _toggle_chamber_fields(self) -> None:
        mode = self.chamber_mode.currentText()
        manual_load_mode = mode == ChamberImpedanceMode.MEASURED_OR_USER_LOAD.value
        downstream_mode = mode == ChamberImpedanceMode.DERIVED_FROM_DOWNSTREAM_PATH.value
        conjugate_mode = mode == ChamberImpedanceMode.CONJUGATE_MATCH_TO_ZOUT.value
        self.zm_r.setEnabled(manual_load_mode)
        self.zm_x.setEnabled(manual_load_mode)
        self.zl_r.setEnabled(downstream_mode)
        self.zl_x.setEnabled(downstream_mode)
        if conjugate_mode:
            self.zm_r.setEnabled(False)
            self.zm_x.setEnabled(False)

    @staticmethod
    def _as_float_or_none(text: str) -> float | None:
        value = text.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_required_float(text: str, field_name: str) -> float:
        value = text.strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid number.") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{field_name} must be finite.")
        return parsed

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
        self.s_mode_lbl.setText(f"Matcher Model: {self.matcher_mode.currentText()}")
        s_mode_active = self.matcher_mode.currentText() == MatcherModelMode.USER_S.value
        calculated_mode = self.power_source.currentText() == MatchOutputPowerSource.CALCULATED.value
        self.match_eff.setEnabled(not (s_mode_active and calculated_mode))
        self.p_match_out.setReadOnly(calculated_mode)
        if s_mode_active and calculated_mode:
            self.match_eff.setText("0")
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

    def _collect_models(self) -> tuple[SystemParameters, CableParameters, ChamberParameters, ConnectorParameters]:
        self._sync_attenuation_from_mode()
        source = MatchOutputPowerSource(self.power_source.currentText())
        matcher_mode = MatcherModelMode(self.matcher_mode.currentText())
        s_mode_calculated = matcher_mode == MatcherModelMode.USER_S and source == MatchOutputPowerSource.CALCULATED
        match_efficiency_percent = 0.0 if s_mode_calculated else self._parse_required_float(self.match_eff.text(), "Match Efficiency")
        system = SystemParameters(
            frequency_mhz=self._parse_required_float(self.freq_mhz.text(), "Frequency"),
            match_input_power_w=self._parse_required_float(self.p_match_in_w.text(), "Match Input Power"),
            generator_impedance_ohm=self._parse_required_float(self.z_gen.text(), "Source Impedance ZS (R)"),
            generator_impedance_imag_ohm=self._parse_required_float(self.z_gen_x.text(), "Source Impedance ZS (X)"),
            match_efficiency_percent=match_efficiency_percent,
            match_output_power_source=source,
            matcher_model_mode=matcher_mode,
            match_output_power_w=self._as_float_or_none(self.p_match_out.text()),
            match_output_impedance=ComplexImpedance(
                self._parse_required_float(self.zm_r.text(), "Historical Load R, RL"),
                self._parse_required_float(self.zm_x.text(), "Historical Load X, XL"),
            ),
            measured_vpp_v=self._as_float_or_none(self.vpp.text()),
            measured_iout_untrusted_a=None,
            s11=complex(
                self._parse_required_float(self.s11r.text(), "S11 real"),
                self._parse_required_float(self.s11i.text(), "S11 imaginary"),
            ),
            s12=complex(
                self._parse_required_float(self.s12r.text(), "S12 real"),
                self._parse_required_float(self.s12i.text(), "S12 imaginary"),
            ),
            s21=complex(
                self._parse_required_float(self.s21r.text(), "S21 real"),
                self._parse_required_float(self.s21i.text(), "S21 imaginary"),
            ),
            s22=complex(
                self._parse_required_float(self.s22r.text(), "S22 real"),
                self._parse_required_float(self.s22i.text(), "S22 imaginary"),
            ),
            sparam_ref_impedance_ohm=self._parse_required_float(self.s_zref.text(), "S-parameter Reference Impedance"),
            sparam_frequency_mhz=self._as_float_or_none(self.s_freq.text()),
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
        connector_mode_text = self.connector_mode.currentText()
        if not self.include_connector.isChecked():
            connector_mode = ConnectorModelMode.DISABLED
        elif connector_mode_text == ConnectorModelMode.TRANSMISSION_LINE.value:
            connector_mode = ConnectorModelMode.TRANSMISSION_LINE
        else:
            connector_mode = ConnectorModelMode.COORDINATE_OFFSET_ONLY
        connector = ConnectorParameters(
            enabled=self.include_connector.isChecked(),
            model_mode=connector_mode,
            model_name=self.connector_model_name.text(),
            physical_length_m=float(self.connector_length.text()) if self.connector_length.text().strip() else None,
            z0_ohm=float(self.connector_z0.text()) if self.connector_z0.text().strip() else None,
            velocity_factor=float(self.connector_vf.text()) if self.connector_vf.text().strip() else None,
            attenuation_value=float(self.connector_att.text()) if self.connector_att.text().strip() else None,
            attenuation_unit=AttenuationUnit(self.connector_att_unit.currentText()),
            ideal_lossless=self.connector_ideal_lossless.isChecked(),
            source_status=self.connector_source_status.text(),
            notes=self.connector_notes.text(),
        )
        return system, cable, chamber, connector

    @staticmethod
    def _safe_windows_filename(name: str) -> str:
        return (
            name.replace(":", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )

    def _resolve_manufacturer_power_rating(self, frequency_mhz: float) -> tuple[float | None, float | None, bool, str]:
        if self.cable_preset.currentText().strip() not in ("COMCRAFT AA-11459", "DX7 Existing Cable"):
            return None, None, False, "NOT DIRECTLY SPECIFIED"
        points = AA11459_POWER_RATING_KW
        for point_f_mhz, p_kw in points:
            if abs(point_f_mhz - frequency_mhz) < 1e-9:
                return p_kw * 1000.0, point_f_mhz * 1e6, False, "MANUFACTURER SPEC"
        if self.att_mode.currentText() == "INTERPOLATE BETWEEN PUBLISHED POINTS":
            try:
                interp = interpolate_log_frequency(points, frequency_mhz)
                return interp.value * 1000.0, frequency_mhz * 1e6, True, "INTERPOLATED"
            except ValueError:
                return None, None, False, "NOT DIRECTLY SPECIFIED"
        return None, None, False, "NOT DIRECTLY SPECIFIED"

    def _compute_stress_result(self, analysis_result: CableAnalysisResult) -> RFStressComparisonResult:
        manufacturer_power, manufacturer_freq_hz, interpolated, rating_status = self._resolve_manufacturer_power_rating(
            self._active_frequency_mhz()
        )
        return build_stress_comparison_result(
            analysis_result=analysis_result,
            manufacturer_power_rating_w=manufacturer_power,
            manufacturer_rating_frequency_hz=manufacturer_freq_hz,
            manufacturer_value_interpolated=interpolated,
            manufacturer_rating_status=rating_status,
        )

    def recalculate(self) -> None:
        if not getattr(self, "_ui_ready", False):
            return
        try:
            self._analysis_call_count += 1
            system, cable, chamber, connector = self._collect_models()
            valid = validate_inputs(system, cable, chamber, connector)
            if not valid.valid:
                self.system_diag.setText("Input Required:\n" + "\n".join(valid.errors))
                self.last_result = None
                self.last_stress_result = None
                self.live_metrics_label.setText(self._format_live_metrics(None, system))
                self._reset_thermal_output("Run RF analysis to compute thermal response.")
                self._focus_invalid_input(valid.errors[0] if valid.errors else "")
                return
            self._clear_invalid_highlight()
            result = analyze_cable(system, cable, chamber, connector=connector)
            self.last_result = result
            self.last_stress_result = self._compute_stress_result(result)
            self.system_diag.setText(self._format_consistency_diagnostic(system, result))
            self.p_match_out.setText(f"{result.match_output_power_w:.6g}")
            if (
                system.matcher_model_mode == MatcherModelMode.USER_S
                and system.match_output_power_source == MatchOutputPowerSource.CALCULATED
                and result.match_input_power_w > 0
            ):
                derived_eff = 100.0 * result.match_output_power_w / result.match_input_power_w
                self.match_eff.setText(f"{derived_eff:.6g}")
            elif system.match_output_power_source == MatchOutputPowerSource.USER and result.match_input_power_w > 0:
                derived_eff = 100.0 * result.match_output_power_w / result.match_input_power_w
                self.match_eff.setText(f"{derived_eff:.6g}")

            if (
                system.matcher_model_mode == MatcherModelMode.USER_S
                and result.load_impedance_zl is not None
                and chamber.mode != ChamberImpedanceMode.MEASURED_OR_USER_LOAD
            ):
                self.zm_r.setText(f"{result.load_impedance_zl.real:.6g}")
                self.zm_x.setText(f"{result.load_impedance_zl.imag:.6g}")
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
            probe_global = self._probe_input_to_global(probe_requested)
            probe_clamped = min(max(probe_global, 0.0), float(result.z_m[-1]))
            self._set_probe_display_from_global(probe_clamped)
            self._draw_distribution()
            self._update_smith_chart_full()
            self._update_stress_dashboard()
            self._update_summary(result, system)
            self._refresh_probe_table()
            self._update_thermal_analysis()
        except Exception as exc:
            self.last_result = None
            self.last_stress_result = None
            self.system_diag.setText(f"Input Required / Error: {exc}")
            self.live_metrics_label.setText(self._format_live_metrics(None, None))
            self._reset_thermal_output("Run RF analysis to compute thermal response.")
            if hasattr(self, "smith_canvas"):
                self.smith_canvas.clear_chart()

    def _format_consistency_diagnostic(self, system: SystemParameters, result: CableAnalysisResult | None = None) -> str:
        if system.measured_vpp_v is None:
            return "Reference-plane / measurement consistency check: Vpp not provided."
        measured_vrms = system.measured_vpp_v / (2.0 * np.sqrt(2.0))
        if result is None:
            return (
                "Reference-plane / measurement consistency check\n"
                f"Measured Vpp={system.measured_vpp_v:.2f} V, Measured Vrms={measured_vrms:.2f} V\n"
                "Loaded-model V2 is unavailable."
            )
        loaded_vrms = result.loaded_model_v2_rms_v
        loaded_vpp = result.loaded_model_v2_vpp_v
        residual = result.measured_vpp_residual_v
        loaded_text = f"{loaded_vrms:.2f} V" if loaded_vrms is not None else "N/A"
        loaded_vpp_text = f"{loaded_vpp:.2f} V" if loaded_vpp is not None else "N/A"
        residual_text = f"{residual:+.2f} V" if residual is not None else "N/A"
        return (
            "Reference-plane / measurement consistency check\n"
            f"Measured Vpp={system.measured_vpp_v:.2f} V, Measured Vrms={measured_vrms:.2f} V\n"
            f"Loaded-model V2 RMS={loaded_text}, Loaded-model V2 Vpp={loaded_vpp_text}\n"
            f"Vpp residual (Measured - Loaded)={residual_text}\n"
            "Do not calculate operating load current as Vpp/|Zout| or Vpp/|Z22|."
        )

    def _update_cards(self, r: CableAnalysisResult, system: SystemParameters) -> None:
        vmax_idx = int(np.argmax(np.abs(r.v_rms)))
        wmax_idx = int(np.argmax(r.p_total_loss_w_per_m))
        eta_match = (r.match_output_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        cable_ref_power = r.flexible_cable_input_power_w if r.flexible_cable_input_power_w > 0 else r.match_output_power_w
        eta_cable = (r.chamber_power_w / cable_ref_power * 100.0) if cable_ref_power > 0 else 0.0
        eta_total = (r.chamber_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        self.card_labels["match_input"].setText(f"MATCH INPUT POWER\n{r.match_input_power_w:.2f} W")
        self.card_labels["match_output"].setText(f"MATCH OUTPUT POWER\n{r.match_output_power_w:.2f} W")
        if r.connector_enabled and r.connector_mode == ConnectorModelMode.TRANSMISSION_LINE.value and r.connector_loss_w is not None:
            self.card_labels["connector_loss"].setText(f"CONNECTOR DISSIPATION\n{r.connector_loss_w:.2f} W")
            self.card_labels["flex_input"].setText(f"FLEXIBLE CABLE INPUT POWER\n{r.flexible_cable_input_power_w:.2f} W")
        else:
            self.card_labels["connector_loss"].setText("CONNECTOR DISSIPATION\nN/A")
            self.card_labels["flex_input"].setText("FLEXIBLE CABLE INPUT POWER\nN/A")
        self.card_labels["cable_loss"].setText(
            f"FLEXIBLE CABLE DISSIPATION\n{r.flexible_cable_total_loss_w:.2f} W ({r.cable_loss_percent:.2f}%)"
        )
        self.card_labels["chamber"].setText(f"CABLE END DELIVERED POWER\n{r.chamber_power_w:.2f} W")
        self.efficiency_label.setText(
            f"Matcher Efficiency [%]: {eta_match:.2f}\n"
            f"Cable Efficiency [%]: {eta_cable:.2f}\n"
            f"Overall Delivery Efficiency [%]: {eta_total:.2f}"
        )
        max_current_text = "Not Available"
        if r.max_current_index is not None and r.max_current_irms_a is not None and r.max_current_z_m is not None:
            max_current_text = f"{r.max_current_irms_a:.2f} Arms @ {r.max_current_z_m:.3f} m"
        self.stress_metrics_label.setText(
            f"MAX VOLTAGE: {np.abs(r.v_rms[vmax_idx]):.2f} Vrms @ {r.z_m[vmax_idx]:.3f} m   |   "
            f"MAX CURRENT: {max_current_text}   |   "
            f"MAX LOCAL DISSIPATION: {r.p_total_loss_w_per_m[wmax_idx]:.3f} W/m @ {r.z_m[wmax_idx]:.3f} m"
        )
        z0_ohm = self._as_float_or_none(self.z0.text())
        i_matched = 0.0
        if z0_ohm is not None and z0_ohm > 0 and r.match_output_power_w >= 0:
            i_matched = math.sqrt(r.match_output_power_w / z0_ohm)
        i_sim_max = r.max_current_irms_a if r.max_current_irms_a is not None else float("nan")
        k_i = (i_sim_max / i_matched) if i_matched > 0 else float("nan")
        max_current_pos = f"{r.max_current_z_m:.3f} m" if r.max_current_z_m is not None else "Not Available"
        self.current_comparison_label.setText(
            f"SIMULATED MAXIMUM CURRENT\n{i_sim_max:.2f} Arms @ {max_current_pos}\n"
            f"MATCHED-LINE CURRENT AT MATCH OUTPUT POWER\n{i_matched:.2f} Arms [DERIVED]\n"
            f"Current Enhancement: {k_i:.2f} x"
        )
        self.min_impedance_point_label.setText(self._format_engineering_point_card("LOWEST-IMPEDANCE POINT", r, "min"))
        self.max_current_point_label.setText(self._format_engineering_point_card("MAXIMUM-CURRENT POINT", r, "max"))
        self.live_metrics_label.setText(self._format_live_metrics(r, system))
        self._update_connector_metrics(r)

    def _jump_probe_to_max_current(self) -> None:
        if self.last_result is None or self.last_result.max_current_z_m is None:
            return
        self._set_probe_display_from_global(float(self.last_result.max_current_z_m))
        self._update_probe_readout()
        self._update_smith_probe_only()
        self._draw_distribution()
        self._draw_stress_dashboard()

    def _open_thermal_analysis_at_hot_spot(self) -> None:
        if self.last_result is None:
            return
        r = self.last_result
        hot_idx = int(np.argmax(r.p_total_loss_w_per_m))
        center = float(r.z_m[hot_idx])
        span = max(0.01, 0.03 * float(r.z_m[-1]))
        start = max(0.0, center - span)
        end = min(float(r.z_m[-1]), center + span)
        self.thermal_hotspot_start_m.setText(f"{start:.6g}")
        self.thermal_hotspot_end_m.setText(f"{end:.6g}")
        target_source = ThermalHeatingSource.SELECTED_HOT_SPOT_REGION.value
        idx = self.thermal_source_combo.findText(target_source)
        if idx >= 0:
            self.thermal_source_combo.setCurrentIndex(idx)
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Thermal Analysis":
                self.tabs.setCurrentIndex(i)
                break
        self._update_thermal_analysis()

    def _on_stress_bar_click(self, event) -> None:
        if self.last_result is None or event.inaxes is None:
            return
        if event.inaxes.get_title() == "MATCHED VS ACTUAL CURRENT":
            self._jump_probe_to_max_current()

    def _on_stress_map_click(self, event) -> None:
        if self.last_result is None or event.inaxes is None or event.xdata is None:
            return
        title = str(event.inaxes.get_title())
        if title in (
            "PHYSICAL CABLE STRESS MAP (Local Current / Matched Current)",
            "CURRENT PROFILE OVERLAY",
            "VOLTAGE PROFILE OVERLAY",
        ):
            z = float(event.xdata)
            self._set_probe_display_from_global(z)
            self._update_probe_readout()
            self._update_smith_probe_only()
            self._draw_distribution()
            self._draw_stress_dashboard()

    def _update_stress_dashboard(self) -> None:
        if self.last_result is None or self.last_stress_result is None:
            return
        r = self.last_result
        s = self.last_stress_result
        max_i_idx = int(np.argmax(np.abs(r.i_rms)))
        max_v_idx = int(np.argmax(np.abs(r.v_rms)))
        max_loss_idx = int(np.argmax(r.p_total_loss_w_per_m))
        max_i_section = str(r.section_name[max_i_idx])
        max_i_cable_pos = r.position_from_flexible_cable_start_m[max_i_idx]
        max_i_cable_text = (
            f"{max_i_cable_pos:.3f} m"
            if np.isfinite(max_i_cable_pos)
            else "INSIDE CONNECTOR / BEFORE FLEXIBLE CABLE START"
        )
        manufacturer_power_text = "NOT DIRECTLY SPECIFIED"
        manufacturer_eq_text = "NOT AVAILABLE"
        manufacturer_freq_text = "NOT DIRECTLY SPECIFIED"
        if s.manufacturer_power_rating_w is not None:
            manufacturer_power_text = f"{s.manufacturer_power_rating_w:.2f} W ({s.manufacturer_rating_status})"
        if s.manufacturer_rating_frequency_hz is not None:
            manufacturer_freq_text = f"{s.manufacturer_rating_frequency_hz / 1e6:.3f} MHz"
        if s.manufacturer_equivalent_current_rms_a is not None:
            manufacturer_eq_text = (
                f"{s.manufacturer_equivalent_current_rms_a:.2f} Arms [DERIVED FROM MANUFACTURER MATCHED-POWER RATING]"
            )
        self.manufacturer_ref_label.setText(
            "MANUFACTURER REFERENCE\n"
            f"Cable Model: {self.spec_model.text()}\n"
            f"Cable Z0: {s.reference_z0_ohm:.3f} ohm\n"
            f"Published Frequency Point: {manufacturer_freq_text}\n"
            f"Published Power Rating: {manufacturer_power_text}\n"
            f"Published Attenuation: {self.att.text() or 'NOT DIRECTLY SPECIFIED'} {self.att_unit.currentText()}\n"
            f"Published Temperature Range: {self.spec_temp_range.text()}\n"
            f"Equivalent Matched-Line Rated Current: {manufacturer_eq_text}\n"
            "Equivalent current is derived from matched-power rating. It is not an explicit manufacturer current limit."
        )
        self.manufacturer_ref_label.setToolTip(
            "This is the RMS traveling-wave current corresponding to the stated manufacturer power rating under a matched-line reference. It is not an explicit manufacturer-specified current limit."
        )
        self.matched_ref_label.setText(
            "IDEAL MATCHED REFERENCE\n"
            f"Match Output Power: {s.match_output_power_w:.2f} W\n"
            f"Cable Z0: {s.reference_z0_ohm:.3f} ohm\n"
            "VSWR: 1.00\n"
            f"Matched RMS Current: {s.matched_current_rms_a:.3f} Arms\n"
            f"Matched RMS Voltage: {s.matched_voltage_rms_v:.3f} Vrms\n"
            "Reflection Coefficient: 0\n"
            "Cable Position Dependence: Uniform reference\n"
            "This is a reference condition, not the actual transmission-line solution."
        )
        self.matched_ref_label.setToolTip("RMS current for an ideal matched line: sqrt(P_match_output / Re(Z0)).")
        self.actual_ref_label.setText(
            "ACTUAL DISTRIBUTED RESULT\n"
            f"Actual Match Output Power: {r.match_output_power_w:.2f} W\n"
            f"Actual VSWR: {r.vswr:.3f}\n"
            f"Maximum RMS Current: {s.actual_max_current_rms_a:.3f} A @ {s.actual_max_current_position_m:.3f} m\n"
            f"Section: {max_i_section}\n"
            f"Position from Flexible Cable Start: {max_i_cable_text}\n"
            f"Maximum RMS Voltage: {s.actual_max_voltage_rms_v:.3f} V @ {s.actual_max_voltage_position_m:.3f} m\n"
            f"Minimum |Z|: {abs(s.minimum_impedance_ohm):.3f} ohm @ {s.minimum_impedance_position_m:.3f} m\n"
            f"Maximum Local Dissipation: {s.maximum_local_loss_w_per_m:.4f} W/m @ {s.maximum_local_loss_position_m:.3f} m\n"
            f"Total Cable Dissipation: {r.flexible_cable_total_loss_w:.3f} W\n"
            f"Cable End Delivered Power: {r.cable_end_delivered_power_w:.3f} W"
        )
        self.actual_ref_label.setToolTip("Maximum RMS current calculated along the modeled RF path.")
        enhancement_text = f"{s.current_enhancement_factor:.2f} x" if np.isfinite(s.current_enhancement_factor) else "NOT AVAILABLE"
        i2_text = f"{s.current_squared_stress_index:.1f} x" if np.isfinite(s.current_squared_stress_index) else "NOT AVAILABLE"
        voltage_enh_text = f"{s.voltage_enhancement_factor:.2f} x" if np.isfinite(s.voltage_enhancement_factor) else "NOT AVAILABLE"
        diff_i = s.actual_max_current_rms_a - s.matched_current_rms_a if np.isfinite(s.matched_current_rms_a) else float("nan")
        diff_i_text = f"{diff_i:.2f} Arms" if np.isfinite(diff_i) else "NOT AVAILABLE"
        self.current_enhancement_card.setText(
            "CURRENT ENHANCEMENT\n"
            f"{enhancement_text}\n"
            f"Matched Current: {s.matched_current_rms_a:.2f} Arms\n"
            f"Actual Maximum Current: {s.actual_max_current_rms_a:.2f} Arms\n"
            f"Difference: {diff_i_text}"
        )
        self.i2_stress_card.setText(
            "LOCAL I-SQUARED STRESS INDEX\n"
            f"{i2_text}\n"
            "Relative current-squared conductor-stress indicator compared with the ideal matched reference.\n"
            "NOT A MANUFACTURER DERATING FACTOR\nNOT A TEMPERATURE PREDICTION\nNOT A VERIFIED FAILURE LIMIT"
        )
        self.voltage_enhancement_card.setText(
            "VOLTAGE ENHANCEMENT\n"
            f"{voltage_enh_text}\n"
            f"Matched Voltage: {s.matched_voltage_rms_v:.2f} Vrms\n"
            f"Actual Maximum Voltage: {s.actual_max_voltage_rms_v:.2f} Vrms\n"
            f"Position: {s.actual_max_voltage_position_m:.3f} m"
        )
        self.interpretation_label.setText(
            s.interpretation_text
            + "\nThis comparison does not represent a manufacturer current limit or a temperature prediction."
            + (
                "\nSPATIAL CORRELATION: The calculated maximum-current location overlaps the aligned historical damage region."
                if self._historical_damage_metadata_available()
                and (0.10 <= s.actual_max_current_position_m <= 0.20)
                else ""
            )
        )

        max_gamma = complex(r.gamma_local[max_i_idx])
        local_vswr = vswr_from_gamma(max_gamma)
        local_vswr_text = f"{local_vswr:.4f}" if np.isfinite(local_vswr) else "N/A"
        self.max_current_detail_label.setText(
            "MAXIMUM CURRENT POINT\n"
            "DERIVED FROM DISTRIBUTED RF MODEL\n"
            f"Section: {max_i_section}\n"
            f"Position from Match Output: {s.actual_max_current_position_m:.3f} m\n"
            f"Position from Flexible Cable Start: {max_i_cable_text}\n"
            f"Irms: {s.actual_max_current_rms_a:.4f} A\n"
            f"Ipeak: {np.sqrt(2.0) * s.actual_max_current_rms_a:.4f} A\n"
            f"Vrms: {float(np.abs(r.v_rms[max_i_idx])):.4f} V\n"
            f"Local Complex Impedance: {r.z_local_ohm[max_i_idx].real:+.4f}{r.z_local_ohm[max_i_idx].imag:+.4f}j ohm\n"
            f"|Z|: {float(abs(r.z_local_ohm[max_i_idx])):.4f} ohm\n"
            f"Gamma: {max_gamma.real:+.4f}{max_gamma.imag:+.4f}j\n"
            f"Local VSWR: {local_vswr_text}\n"
            f"Net Real Power: {r.p_real_w[max_i_idx]:.4f} W\n"
            f"Local Dissipation: {r.p_total_loss_w_per_m[max_i_idx]:.6f} W/m\n"
            f"Matched Reference Current: {s.matched_current_rms_a:.4f} A\n"
            f"Current Enhancement: {s.current_enhancement_factor:.4f} x\n"
            f"I-squared Stress Index: {s.current_squared_stress_index:.4f} x"
        )

        min_idx = int(r.min_impedance_index) if r.min_impedance_index is not None else None
        if min_idx is None:
            self.min_impedance_detail_label.setText("LOWEST-IMPEDANCE POINT\nNot Available")
        else:
            min_z = complex(r.z_local_ohm[min_idx])
            dist_from_max = abs(float(r.z_m[min_idx]) - float(r.z_m[max_i_idx]))
            self.min_impedance_detail_label.setText(
                "LOWEST-IMPEDANCE POINT\n"
                f"Position: {float(r.z_m[min_idx]):.3f} m\n"
                f"Section: {str(r.section_name[min_idx])}\n"
                f"Z = {min_z.real:+.4f}{min_z.imag:+.4f}j ohm\n"
                f"|Z| = {abs(min_z):.4f} ohm\n"
                f"Impedance Phase: {np.degrees(np.angle(min_z)):.2f} deg\n"
                f"Vrms: {float(np.abs(r.v_rms[min_idx])):.4f} V\n"
                f"Irms: {float(np.abs(r.i_rms[min_idx])):.4f} A\n"
                f"Net Power: {float(r.p_real_w[min_idx]):.4f} W\n"
                f"Local Dissipation: {float(r.p_total_loss_w_per_m[min_idx]):.6f} W/m\n"
                f"Distance from Maximum-Current Point: {dist_from_max:.4f} m"
            )
        self._draw_stress_dashboard()

    def _format_engineering_point_card(self, title: str, r: CableAnalysisResult, point_kind: str) -> str:
        if point_kind == "min":
            z_pos = r.min_impedance_z_m
            z_local = r.min_impedance_z_ohm
            z_mag = r.min_impedance_magnitude_ohm
            irms = r.min_impedance_irms_a
            vrms = r.min_impedance_vrms_v
            p_real = r.min_impedance_real_power_w
        else:
            z_pos = r.max_current_z_m
            z_local = r.max_current_z_ohm
            z_mag = r.max_current_magnitude_ohm
            irms = r.max_current_irms_a
            vrms = r.max_current_vrms_v
            p_real = r.max_current_real_power_w
        if z_pos is None or z_local is None or z_mag is None or irms is None or vrms is None or p_real is None:
            return f"{title}\nNot Available"
        idx = int(np.argmin(np.abs(r.global_position_m - z_pos)))
        section = str(r.section_name[idx])
        cable_pos = r.position_from_flexible_cable_start_m[idx]
        cable_pos_txt = (
            f"From Cable Start: {cable_pos:.3f} m"
            if np.isfinite(cable_pos)
            else "From Cable Start: INSIDE CONNECTOR / BEFORE FLEXIBLE CABLE START"
        )
        return (
            f"{title}\n"
            f"From Match Output: {z_pos:.3f} m\n"
            f"Section: {section}\n"
            f"{cable_pos_txt}\n"
            f"Z = {z_local.real:+.3f}{z_local.imag:+.3f}j ohm  (|Z|={z_mag:.3f} ohm)\n"
            f"R = {z_local.real:.3f} ohm, X = {z_local.imag:.3f} ohm\n"
            f"Irms = {irms:.3f} A, Vrms = {vrms:.3f} V, P = {p_real:.3f} W"
        )

    def _update_connector_metrics(self, r: CableAnalysisResult) -> None:
        if not hasattr(self, "connector_metrics"):
            return
        if not r.connector_enabled:
            self.connector_metrics.setText("Connector disabled.")
            return
        if r.connector_mode == ConnectorModelMode.COORDINATE_OFFSET_ONLY.value:
            self.connector_metrics.setText(
                f"Connector Physical Length [m]: {r.connector_length_m:.4f}\n"
                "COORDINATE OFFSET ONLY\nCONNECTOR RF BEHAVIOR NOT MODELED"
            )
            return
        deg = r.connector_electrical_length_deg
        wl = r.connector_electrical_length_wavelength
        delay = r.connector_propagation_delay_ns
        self.connector_metrics.setText(
            f"Connector Physical Length [m]: {r.connector_length_m:.4f}\n"
            f"Connector Electrical Length [deg]: {deg:.3f}\n"
            f"Connector Electrical Length [wavelength]: {wl:.6f}\n"
            f"Connector Propagation Delay [ns]: {delay:.6f}"
        )

    def _clear_invalid_highlight(self) -> None:
        if self._last_invalid_widget is not None:
            self._last_invalid_widget.setStyleSheet("")
            self._last_invalid_widget = None

    def _focus_invalid_input(self, error_message: str) -> None:
        self._clear_invalid_highlight()
        widget_map = [
            ("Connector length", self.connector_length),
            ("connector mode requires length", self.connector_length),
            ("Connector Z0", self.connector_z0),
            ("Connector velocity factor", self.connector_vf),
            ("Connector attenuation", self.connector_att),
            ("Cable length", self.length),
            ("Cable characteristic impedance", self.z0),
            ("Velocity factor", self.vf),
            ("Datasheet attenuation", self.att),
            ("Frequency", self.freq_mhz),
        ]
        for key, widget in widget_map:
            if key.lower() in error_message.lower():
                self._last_invalid_widget = widget
                widget.setStyleSheet("border:2px solid #ef4444;")
                if hasattr(self, "cable_spec_scroll_area"):
                    self.cable_spec_scroll_area.ensureWidgetVisible(widget)
                break

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

    def _draw_stress_dashboard(self) -> None:
        if self.last_result is None or self.last_stress_result is None:
            return
        r = self.last_result
        s = self.last_stress_result
        label_box = {"facecolor": "#ffffff", "edgecolor": "#334155", "alpha": 0.88, "pad": 0.25}

        # Shared-scale bars
        bar_fig = self.stress_bar_canvas.figure
        bar_fig.clear()
        bar_axes = bar_fig.subplots(1, 3)
        current_max = max(s.actual_max_current_rms_a, s.matched_current_rms_a, 1e-9)
        voltage_max = max(s.actual_max_voltage_rms_v, s.matched_voltage_rms_v, 1e-9)
        i2_max = max(float(s.current_squared_stress_index), 1.0, 1e-9) if np.isfinite(s.current_squared_stress_index) else 1.0

        bar_axes[0].barh(["Matched RMS Current", "Actual Maximum RMS Current"], [s.matched_current_rms_a, s.actual_max_current_rms_a], color=["#22c55e", "#f97316"])
        bar_axes[0].set_xlim(0.0, current_max * 1.15)
        bar_axes[0].set_title("MATCHED VS ACTUAL CURRENT")
        bar_axes[0].text(s.matched_current_rms_a, 0, f" {s.matched_current_rms_a:.2f} A", va="center", color="#0f172a", fontweight="bold")
        bar_axes[0].text(s.actual_max_current_rms_a, 1, f" {s.actual_max_current_rms_a:.2f} A", va="center", color="#0f172a", fontweight="bold")
        bar_axes[0].text(0.01 * current_max, 1.25, f"{s.current_enhancement_factor:.2f} x", color="#9a3412", fontweight="bold")

        bar_axes[1].barh(["Matched RMS Voltage", "Actual Maximum RMS Voltage"], [s.matched_voltage_rms_v, s.actual_max_voltage_rms_v], color=["#22c55e", "#22d3ee"])
        bar_axes[1].set_xlim(0.0, voltage_max * 1.15)
        bar_axes[1].set_title("MATCHED VS ACTUAL VOLTAGE")
        bar_axes[1].text(s.matched_voltage_rms_v, 0, f" {s.matched_voltage_rms_v:.1f} V", va="center", color="#0f172a", fontweight="bold")
        bar_axes[1].text(s.actual_max_voltage_rms_v, 1, f" {s.actual_max_voltage_rms_v:.1f} V", va="center", color="#0f172a", fontweight="bold")
        bar_axes[1].text(0.01 * voltage_max, 1.25, f"Enhancement: {s.voltage_enhancement_factor:.2f} x", color="#0c4a6e", fontweight="bold")

        i2_value = s.current_squared_stress_index if np.isfinite(s.current_squared_stress_index) else 0.0
        bar_axes[2].barh(["Matched Reference", "Actual Maximum"], [1.0, i2_value], color=["#22c55e", "#ef4444"])
        bar_axes[2].set_xlim(0.0, i2_max * 1.15)
        bar_axes[2].set_title("LOCAL CURRENT-SQUARED STRESS")
        bar_axes[2].text(i2_value, 1, f" {i2_value:.1f} x MATCHED REFERENCE", va="center", color="#7f1d1d", fontweight="bold")
        bar_axes[2].text(
            0.0,
            -0.45,
            "Relative indicator only. This is not an official cable current limit,\nmanufacturer derating factor or cable temperature prediction.",
            fontsize=7,
            color="#7f1d1d",
        )
        for ax in bar_axes:
            ax.grid(True, alpha=0.25)
        bar_fig.tight_layout()
        self.stress_bar_canvas.draw_idle()

        # Stress map + overlays
        fig = self.stress_map_canvas.figure
        fig.clear()
        axes = fig.subplots(5, 1, sharex=False)
        z = r.z_m
        i_abs = np.abs(r.i_rms)
        v_abs = np.abs(r.v_rms)
        matched_i = s.matched_current_rms_a if np.isfinite(s.matched_current_rms_a) and s.matched_current_rms_a > 0 else 1e-12
        matched_v = s.matched_voltage_rms_v if np.isfinite(s.matched_voltage_rms_v) and s.matched_voltage_rms_v > 0 else 1e-12
        k_i_profile = np.abs(i_abs) / matched_i
        cmap = colormaps["turbo"]

        # Executive visualization
        axes[0].set_title("RF CURRENT-STRESS VISUALIZATION")
        axes[0].plot([z[0], z[-1]], [0.75, 0.75], color="#22c55e", linewidth=10, solid_capstyle="round")
        axes[0].text(
            z[0],
            0.86,
            f"IDEAL MATCHED | VSWR=1 | I={s.matched_current_rms_a:.2f} A | V={s.matched_voltage_rms_v:.1f} V",
            color="#14532d",
            fontweight="bold",
            bbox=label_box,
        )
        norm = (k_i_profile - np.nanmin(k_i_profile)) / max(np.nanmax(k_i_profile) - np.nanmin(k_i_profile), 1e-12)
        for i in range(len(z) - 1):
            axes[0].plot(z[i : i + 2], [0.35, 0.35], color=cmap(norm[i]), linewidth=10, solid_capstyle="round")
        max_i_idx = int(np.argmax(i_abs))
        axes[0].scatter([z[max_i_idx]], [0.35], marker="*", s=180, color="#fb923c", edgecolor="#111827", zorder=12)
        axes[0].text(
            z[0],
            0.06,
            f"ACTUAL DISTRIBUTED | VSWR={r.vswr:.3f} | Imax={s.actual_max_current_rms_a:.2f} A | Vmax={s.actual_max_voltage_rms_v:.1f} V | KI={s.current_enhancement_factor:.2f}x | KI^2={s.current_squared_stress_index:.1f}x",
            color="#9a3412",
            fontweight="bold",
            bbox=label_box,
        )
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_yticks([])
        axes[0].grid(True, alpha=0.18)

        # Physical stress map
        axes[1].set_title("PHYSICAL CABLE STRESS MAP (Local Current / Matched Current)")
        for i in range(len(z) - 1):
            axes[1].plot(z[i : i + 2], [0.5, 0.5], color=cmap(norm[i]), linewidth=14, solid_capstyle="butt")
        section_switches = np.flatnonzero(r.section_name[:-1] != r.section_name[1:])
        for idx in section_switches:
            axes[1].axvline(float(z[idx]), color="#334155", linestyle="--", linewidth=1.0, alpha=0.7)
            axes[1].text(float(z[idx]), 0.86, str(r.section_name[idx]), fontsize=8, color="#0f172a", bbox=label_box)
        min_idx = int(r.min_impedance_index) if r.min_impedance_index is not None else None
        vmax_idx = int(np.argmax(v_abs))
        loss_idx = int(np.argmax(r.p_total_loss_w_per_m))
        axes[1].scatter([z[max_i_idx]], [0.66], marker="*", s=170, color="#f97316", edgecolor="#111827", zorder=13)
        if min_idx is not None:
            axes[1].scatter([z[min_idx]], [0.61], marker="^", s=110, color="#fb923c", edgecolor="#111827", zorder=13)
        axes[1].scatter([z[vmax_idx]], [0.56], marker="D", s=95, color="#22d3ee", edgecolor="#111827", zorder=13)
        axes[1].scatter([z[loss_idx]], [0.51], marker="o", s=95, color="#ef4444", edgecolor="#111827", zorder=13)
        try:
            probe_global = self._probe_input_to_global(float(self.probe_pos.text()))
            axes[1].scatter([probe_global], [0.46], marker="o", s=80, color="#facc15", edgecolor="#111827", zorder=13)
        except ValueError:
            pass
        if self._historical_damage_metadata_available():
            axes[1].axvspan(0.10, 0.20, color="#7f1d1d", alpha=0.22)
        axes[1].set_ylim(0.38, 0.92)
        axes[1].set_yticks([])
        axes[1].set_xlabel("Position from Match Output z [m]")
        axes[1].grid(True, alpha=0.2)
        sm = np.linspace(np.nanmin(k_i_profile), np.nanmax(k_i_profile), 100)
        axes[1].text(
            z[0],
            0.40,
            f"VISUALIZATION SCALE (NOT SAFETY LIMITS)  Min={float(np.nanmin(sm)):.2f}  Max={float(np.nanmax(sm)):.2f}",
            color="#1e3a8a",
            fontsize=8,
            fontweight="bold",
            bbox=label_box,
        )

        # Current profile
        axes[2].set_title("CURRENT PROFILE OVERLAY")
        axes[2].plot(z, i_abs, color="#fb923c", label="Actual I(z)")
        axes[2].axhline(s.matched_current_rms_a, color="#22c55e", linestyle="--", label="IDEAL MATCHED CURRENT REFERENCE")
        axes[2].fill_between(z, i_abs, s.matched_current_rms_a, where=(i_abs > s.matched_current_rms_a), color="#fb923c", alpha=0.15)
        axes[2].axvline(z[max_i_idx], color="#f97316", linestyle="-.", linewidth=1.2)
        if self._historical_damage_metadata_available():
            axes[2].axvspan(0.10, 0.20, color="#7f1d1d", alpha=0.18)
        axes[2].set_ylabel("Irms [A]")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="upper right")

        # Voltage profile
        axes[3].set_title("VOLTAGE PROFILE OVERLAY")
        axes[3].plot(z, v_abs, color="#22d3ee", label="Actual V(z)")
        axes[3].axhline(s.matched_voltage_rms_v, color="#22c55e", linestyle="--", label="IDEAL MATCHED VOLTAGE REFERENCE")
        axes[3].axvline(z[vmax_idx], color="#22d3ee", linestyle="-.", linewidth=1.2)
        if min_idx is not None:
            min_v_idx = int(np.argmin(v_abs))
            axes[3].axvline(z[min_v_idx], color="#38bdf8", linestyle=":", linewidth=1.0)
        if self._historical_damage_metadata_available():
            axes[3].axvspan(0.10, 0.20, color="#7f1d1d", alpha=0.18)
        axes[3].set_ylabel("Vrms [V]")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(loc="upper right")

        # Power flow chain
        axes[4].set_title("ACTUAL DISTRIBUTED POWER FLOW")
        labels = ["Match Input", "Match Loss", "Match Output", "Connector Loss", "Flexible Loss", "Cable End"]
        values = [
            r.match_input_power_w,
            r.match_loss_w,
            r.match_output_power_w,
            float(r.connector_loss_w or 0.0),
            r.flexible_cable_total_loss_w,
            r.cable_end_delivered_power_w,
        ]
        colors = ["#3b82f6", "#fb923c", "#22d3ee", "#a855f7", "#ef4444", "#22c55e"]
        axes[4].barh(labels, values, color=colors)
        for idx, val in enumerate(values):
            pct = (val / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
            axes[4].text(val, idx, f" {val:.2f} W ({pct:.1f}%)", va="center", color="#0f172a", fontsize=8, fontweight="bold")
        axes[4].set_xlabel("Power [W]")
        axes[4].grid(True, alpha=0.25)

        fig.tight_layout()
        self.stress_map_canvas.draw_idle()

    def _draw_probe_markers(self, axes) -> None:
        if self.last_result is None:
            return
        try:
            z_probe = self._probe_input_to_global(float(self.probe_pos.text()))
        except ValueError:
            return
        for ax in axes:
            ax.axvline(z_probe, color="#facc15", linestyle="--", linewidth=1.6, zorder=20)
            if self.last_result.min_impedance_z_m is not None:
                ax.axvline(self.last_result.min_impedance_z_m, color="#f97316", linestyle=":", linewidth=1.2, zorder=18)
            if self.last_result.max_current_z_m is not None:
                ax.axvline(self.last_result.max_current_z_m, color="#a78bfa", linestyle="-.", linewidth=1.2, zorder=18)

    def _position_index(self, pos_m: float) -> int:
        r = self.last_result
        assert r is not None
        return int(np.argmin(np.abs(r.z_m - pos_m)))

    def _probe_input_to_global(self, probe_value: float) -> float:
        if self.last_result is None:
            return probe_value
        if self.probe_reference.currentText() == "From Flexible Cable Start":
            return probe_value + self.last_result.cable_start_global_position_m
        return probe_value

    def _probe_global_to_input(self, global_value: float) -> float:
        if self.last_result is None:
            return global_value
        if self.probe_reference.currentText() == "From Flexible Cable Start":
            return max(0.0, global_value - self.last_result.cable_start_global_position_m)
        return global_value

    def _set_probe_display_from_global(self, global_value: float) -> None:
        if self.last_result is None:
            return
        z_clamped = min(max(global_value, 0.0), float(self.last_result.z_m[-1]))
        probe_value = self._probe_global_to_input(z_clamped)
        self.probe_pos.setText(f"{probe_value:.5f}")
        self.probe_slider.setValue(int(1000.0 * z_clamped / self.last_result.z_m[-1]))

    def _on_probe_reference_changed(self, _text: str) -> None:
        if self.last_result is None:
            return
        z_global = float(self.last_result.z_m[-1]) * self.probe_slider.value() / 1000.0
        self._set_probe_display_from_global(z_global)
        self._update_probe_readout()
        self._update_smith_probe_only()
        self._draw_stress_dashboard()

    def _on_probe_slider(self, value: int) -> None:
        if self.last_result is None:
            return
        z = self.last_result.z_m[-1] * value / 1000.0
        self._set_probe_display_from_global(z)
        self._update_probe_readout()
        self._update_smith_probe_only()
        self._draw_distribution()
        self._draw_stress_dashboard()

    def _on_probe_text(self) -> None:
        if self.last_result is None:
            return
        try:
            z = self._probe_input_to_global(float(self.probe_pos.text()))
        except ValueError:
            return
        self._set_probe_display_from_global(z)
        self._update_probe_readout()
        self._update_smith_probe_only()
        self._draw_distribution()
        self._draw_stress_dashboard()

    def _update_probe_readout(self) -> None:
        if self.last_result is None:
            return
        z = self._probe_input_to_global(float(self.probe_pos.text()))
        probe = self._probe_state_at(z)
        r = self.last_result
        self.probe_values.setText(
            f"Probe {probe['z']:.3f} m (From Match) | Vrms={probe['vrms']:.2f} V | Irms={probe['irms']:.2f} A | "
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
            f"Section: {probe['section']}\n"
            f"Probe Position from Flexible Cable Start [m]: {probe['cable_pos_text']}\n"
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
        self._draw_stress_dashboard()

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
            z_global = self._probe_input_to_global(z)
            if z_global < 0 or z_global > float(r.z_m[-1]):
                continue
            idx = self._position_index(z_global)
            vrms = np.abs(r.v_rms[idx])
            self._measurement_row_positions.append(z_global)
            values.append(
                (
                    z_global,
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
        section_index = int(np.argmin(np.abs(r.global_position_m - z_probe)))
        cable_pos = float(r.position_from_flexible_cable_start_m[section_index])
        cable_pos_text = f"{cable_pos:.3f}" if np.isfinite(cable_pos) else "INSIDE CONNECTOR / BEFORE CABLE START"
        valid_impedance = (
            np.isfinite(z_local.real)
            and np.isfinite(z_local.imag)
            and np.isfinite(irms)
            and (irms >= CURRENT_EPSILON_A)
        )
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
            "section": str(r.section_name[section_index]),
            "cable_pos_text": cable_pos_text,
        }

    def _format_smith_hover(self, idx: int) -> str:
        if self.last_result is None:
            return ""
        r = self.last_result
        z = float(r.z_m[idx])
        z_local = complex(r.z_local_ohm[idx])
        z_norm = complex(r.z_normalized[idx])
        gamma = complex(r.gamma_local[idx])
        section = str(r.section_name[idx])
        cable_pos = r.position_from_flexible_cable_start_m[idx]
        cable_pos_txt = f"{cable_pos:.3f} m" if np.isfinite(cable_pos) else "INSIDE CONNECTOR"
        return (
            f"z={z:.3f} m\n"
            f"section={section}\n"
            f"from cable start={cable_pos_txt}\n"
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
        probe = self._probe_state_at(self._probe_input_to_global(float(self.probe_pos.text())))
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
        if r.connector_enabled and r.connector_mode == ConnectorModelMode.TRANSMISSION_LINE.value:
            self.smith_canvas.set_segmented_trajectory(r.gamma_local, r.z_m, r.connector_interface_index, title=title)
        else:
            self.smith_canvas.set_trajectory(r.gamma_local, r.z_m, title=title)
        self.smith_canvas.set_endpoints(complex(r.gamma_local[0]), complex(r.gamma_local[-1]))
        min_impedance_gamma = complex(r.gamma_local[r.min_impedance_index]) if r.min_impedance_index is not None else None
        max_current_gamma = complex(r.gamma_local[r.max_current_index]) if r.max_current_index is not None else None
        max_voltage_gamma = complex(r.gamma_local[int(np.argmax(np.abs(r.v_rms)))])
        max_loss_gamma = complex(r.gamma_local[int(np.argmax(r.p_total_loss_w_per_m))])
        section_interfaces = [int(idx) for idx in np.flatnonzero(r.section_name[:-1] != r.section_name[1:])]
        self.smith_canvas.set_engineering_markers(
            min_impedance_gamma=min_impedance_gamma,
            max_current_gamma=max_current_gamma,
            max_voltage_gamma=max_voltage_gamma,
            max_local_loss_gamma=max_loss_gamma,
            section_interface_gammas=[complex(r.gamma_local[idx]) for idx in section_interfaces],
        )

        if self._historical_damage_metadata_available():
            damage_start = self._as_float_or_none(self.historical_damage_start.text()) or 0.10
            damage_end = self._as_float_or_none(self.historical_damage_end.text()) or 0.20
            ref = self.historical_coordinate_reference.currentText()
            if ref == HistoricalCoordinateReference.FROM_FLEXIBLE_CABLE_START.value:
                damage_start = r.cable_start_global_position_m + damage_start
                damage_end = r.cable_start_global_position_m + damage_end
            damage_mask = (r.z_m >= min(damage_start, damage_end)) & (r.z_m <= max(damage_start, damage_end))
            damage_gamma = np.full(r.gamma_local.shape, np.nan + 1j * np.nan, dtype=np.complex128)
            damage_gamma[damage_mask] = r.gamma_local[damage_mask]
            self.smith_canvas.set_historical_region(damage_gamma, enabled=True)
            self.smith_note_label.setText(
                f"Historical Damage Distance: {min(damage_start, damage_end):.3f} m to {max(damage_start, damage_end):.3f} m "
                f"[{ref}]"
            )
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
        self.smith_match_label.setText(f"Load Impedance ZL\n{z_match.real:+.3f}{z_match.imag:+.3f}j ohm")
        self.smith_end_label.setText(f"Chamber/Cable-End Impedance\n{z_end.real:+.3f}{z_end.imag:+.3f}j ohm")
        z0 = complex(r.z0_reference)

        def _gamma_from_z(z_value: complex | None) -> complex | None:
            if z_value is None:
                return None
            denom = z_value + z0
            if abs(denom) < 1e-12:
                return None
            return (z_value - z0) / denom

        self.smith_canvas.set_impedance_summary_markers(
            zout_gamma=_gamma_from_z(r.matcher_output_impedance_zout),
            conjugate_zout_gamma=_gamma_from_z(r.conjugate_zout_target),
            zl_gamma=_gamma_from_z(r.load_impedance_zl),
            zin_gamma=_gamma_from_z(r.matcher_input_impedance_zin),
        )
        if r.connector_enabled and r.connector_mode == ConnectorModelMode.COORDINATE_OFFSET_ONLY.value:
            self.smith_note_label.setText(
                (self.smith_note_label.text() + "\n" if self.smith_note_label.text() else "")
                + "Connector RF behavior not modeled (coordinate offset only)."
            )
        self._update_smith_measurement_markers()
        self._update_smith_probe_only()

    def _on_smith_probe_selected(self, z_m: float) -> None:
        if self.last_result is None:
            return
        z_clamped = min(max(float(z_m), 0.0), float(self.last_result.z_m[-1]))
        self._set_probe_display_from_global(z_clamped)
        self._update_probe_readout()
        self._update_smith_probe_only()
        self._draw_distribution()
        self._draw_stress_dashboard()

    def _on_probe_table_selection_changed(self) -> None:
        row = self.probe_table.currentRow()
        if row < 0 or row >= len(self._measurement_row_positions) or self.last_result is None:
            return
        z = self._measurement_row_positions[row]
        self._set_probe_display_from_global(z)
        self._update_probe_readout()
        self._update_smith_probe_only()
        self._draw_distribution()
        self._draw_stress_dashboard()

    def run_sweep(self) -> None:
        try:
            system, cable, chamber, _connector = self._collect_models()
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
        system, cable, chamber, connector = self._collect_models()
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["SECTION", "key", "value"])
            for k, v in asdict(system).items():
                w.writerow(["system", k, v])
            for k, v in asdict(cable).items():
                w.writerow(["cable", k, v])
            for k, v in asdict(chamber).items():
                w.writerow(["chamber", k, v])
            for k, v in asdict(connector).items():
                w.writerow(["connector", k, v])
            if self.last_stress_result is not None:
                s = self.last_stress_result
                w.writerow([])
                w.writerow(["SECTION", "key", "value"])
                w.writerow(["stress_comparison", "match_output_power_w", s.match_output_power_w])
                w.writerow(["stress_comparison", "reference_z0_ohm", s.reference_z0_ohm])
                w.writerow(["stress_comparison", "matched_current_rms_a", s.matched_current_rms_a])
                w.writerow(["stress_comparison", "matched_voltage_rms_v", s.matched_voltage_rms_v])
                w.writerow(["stress_comparison", "actual_max_current_rms_a", s.actual_max_current_rms_a])
                w.writerow(["stress_comparison", "actual_max_current_position_m", s.actual_max_current_position_m])
                w.writerow(["stress_comparison", "actual_max_voltage_rms_v", s.actual_max_voltage_rms_v])
                w.writerow(["stress_comparison", "actual_max_voltage_position_m", s.actual_max_voltage_position_m])
                w.writerow(["stress_comparison", "minimum_impedance_magnitude_ohm", abs(s.minimum_impedance_ohm)])
                w.writerow(["stress_comparison", "minimum_impedance_position_m", s.minimum_impedance_position_m])
                w.writerow(["stress_comparison", "maximum_local_loss_w_per_m", s.maximum_local_loss_w_per_m])
                w.writerow(["stress_comparison", "maximum_local_loss_position_m", s.maximum_local_loss_position_m])
                w.writerow(["stress_comparison", "current_enhancement_factor", s.current_enhancement_factor])
                w.writerow(["stress_comparison", "voltage_enhancement_factor", s.voltage_enhancement_factor])
                w.writerow(["stress_comparison", "current_squared_stress_index", s.current_squared_stress_index])
                w.writerow(["stress_comparison", "manufacturer_power_rating_w", s.manufacturer_power_rating_w])
                w.writerow(["stress_comparison", "manufacturer_rating_frequency_hz", s.manufacturer_rating_frequency_hz])
                w.writerow(
                    [
                        "stress_comparison",
                        "manufacturer_equivalent_current_rms_a",
                        s.manufacturer_equivalent_current_rms_a,
                    ]
                )
                w.writerow(["stress_comparison", "manufacturer_rating_status", s.manufacturer_rating_status])
            w.writerow([])
            w.writerow(
                [
                    "global_position_from_match_m",
                    "section_name",
                    "section_local_position_m",
                    "position_from_flexible_cable_start_m",
                    "V_rms",
                    "I_rms",
                    "Net_Power_W",
                    "P_cond_W_per_m",
                    "P_diel_W_per_m",
                    "Local_Dissipation_W_per_m",
                    "Cumulative_Dissipation_W",
                    "Z_real",
                    "Z_imag",
                    "Gamma_real",
                    "Gamma_imag",
                ]
            )
            for i in range(len(r.z_m)):
                z_local = complex(r.z_local_ohm[i])
                gamma = complex(r.gamma_local[i])
                cable_pos = r.position_from_flexible_cable_start_m[i]
                w.writerow(
                    [
                        r.global_position_m[i],
                        r.section_name[i],
                        r.section_local_position_m[i],
                        "" if np.isnan(cable_pos) else cable_pos,
                        float(np.abs(r.v_rms[i])),
                        float(np.abs(r.i_rms[i])),
                        r.p_real_w[i],
                        r.p_cond_w_per_m[i],
                        r.p_diel_w_per_m[i],
                        r.p_total_loss_w_per_m[i],
                        r.p_cumulative_loss_w[i],
                        z_local.real,
                        z_local.imag,
                        gamma.real,
                        gamma.imag,
                    ]
                )

    def export_smith_chart_image(self) -> None:
        if self.last_result is None:
            return
        freq = self._active_frequency_mhz()
        z0 = self.last_result.z0_reference
        default_name = f"smith_chart_{freq:.3f}MHz_{float(self.last_result.z_m[-1]):.3f}m_{z0.real:.2f}ohm.png"
        safe_name = self._safe_windows_filename(default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Export Smith Chart Image", safe_name, "PNG (*.png)")
        if not path:
            return
        self.smith_canvas.export_png(path)

    def export_rf_stress_dashboard_image(self) -> None:
        if self.last_result is None:
            return
        freq = self._active_frequency_mhz()
        power = self.last_result.match_output_power_w
        default_name = f"rf_stress_comparison_{freq:.3f}MHz_{power:.0f}W.png"
        safe_name = self._safe_windows_filename(default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Export RF Stress Dashboard Image", safe_name, "PNG (*.png)")
        if not path:
            return
        pixmap = self.stress_tab.grab()
        pixmap.save(path, "PNG")

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
        cable_ref_power = r.flexible_cable_input_power_w if r.flexible_cable_input_power_w > 0 else r.match_output_power_w
        eta_cable = (r.chamber_power_w / cable_ref_power * 100.0) if cable_ref_power > 0 else 0.0
        eta_total = (r.chamber_power_w / r.match_input_power_w * 100.0) if r.match_input_power_w > 0 else 0.0
        ideal_note = ""
        if "IDEAL MATCHER" in r.match_output_power_source_label:
            ideal_note = "IDEAL MATCHER - NO MATCH LOSS ASSUMED\n\n"
        zout_text = (
            f"{r.matcher_output_impedance_zout.real:+.3f}{r.matcher_output_impedance_zout.imag:+.3f}j ohm"
            if r.matcher_output_impedance_zout is not None
            else "LOAD REQUIRED"
        )
        zin_text = (
            f"{r.matcher_input_impedance_zin.real:+.3f}{r.matcher_input_impedance_zin.imag:+.3f}j ohm"
            if r.matcher_input_impedance_zin is not None
            else "LOAD REQUIRED"
        )
        z22_text = (
            f"{r.z22_open_circuit_parameter_ohm.real:+.3f}{r.z22_open_circuit_parameter_ohm.imag:+.3f}j ohm"
            if r.z22_open_circuit_parameter_ohm is not None
            else "N/A"
        )
        zl_text = (
            f"{r.load_impedance_zl.real:+.3f}{r.load_impedance_zl.imag:+.3f}j ohm"
            if r.load_impedance_zl is not None
            else "LOAD REQUIRED"
        )
        conjugate_text = (
            f"{r.conjugate_zout_target.real:+.3f}{r.conjugate_zout_target.imag:+.3f}j ohm"
            if r.conjugate_zout_target is not None
            else "N/A"
        )
        delta_z_text = (
            f"{r.conjugate_match_delta_z.real:+.3f}{r.conjugate_match_delta_z.imag:+.3f}j ohm"
            if r.conjugate_match_delta_z is not None
            else "N/A"
        )
        passivity_text = (
            f"{r.passivity_check_label} (sigma_max={r.passivity_largest_singular_value:.4f})"
            if r.passivity_largest_singular_value is not None and r.passivity_check_label is not None
            else "N/A"
        )
        self.summary_label.setText(
            f"Reference planes:\n"
            f"Match Input -> Matcher -> Match Output -> Connector/Transition -> Flexible Cable Start -> Cable End/Chamber\n\n"
            f"{ideal_note}"
            f"ACTIVE LOAD MODE = {r.active_load_mode_label}\n"
            f"Source Impedance ZS = {r.matcher_source_impedance_zs.real:+.3f}{r.matcher_source_impedance_zs.imag:+.3f}j ohm\n"
            f"Matcher Input Impedance Zin = {zin_text}\n"
            f"Z22 Open-Circuit Network Parameter = {z22_text}\n"
            f"Matcher Output Impedance Zout = {zout_text}\n"
            f"Conjugate(Zout) = {conjugate_text}\n"
            f"Active Load Impedance ZL = {zl_text}\n"
            f"Conjugate Match Delta Z = {delta_z_text}\n"
            f"S-Parameter Passivity Check = {passivity_text}\n\n"
            f"Match Input Power = {r.match_input_power_w:.3f} W\n"
            f"Match Output Power = {r.match_output_power_w:.3f} W [{r.match_output_power_source_label}]\n"
            f"Match Loss = {r.match_loss_w:.3f} W\n"
            f"Connector Dissipation Power = {r.connector_loss_w if r.connector_loss_w is not None else 'N/A'} W\n"
            f"Flexible Cable Dissipation Power = {r.flexible_cable_total_loss_w:.3f} W\n"
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
