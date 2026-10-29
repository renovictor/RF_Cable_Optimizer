from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
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
from gui.plots import MplCanvas
from models.cable_parameters import AttenuationUnit, CableModelMode, CableParameters, CableSpecification
from models.chamber_parameters import ChamberImpedanceMode, ChamberParameters
from models.operating_points import find_operating_point
from models.system_parameters import ComplexImpedance, MatchOutputPowerSource, MatcherModelMode, SystemParameters
from models.validation import validate_inputs
from rf.coax import inner_diameter_for_target_z0, z0_from_geometry
from version import APP_VERSION


def _line_edit(value: str, tooltip: str) -> QLineEdit:
    w = QLineEdit(value)
    w.setToolTip(tooltip)
    w.setValidator(QDoubleValidator(bottom=-1e12, top=1e12, decimals=8))
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"RF Match-to-Chamber Cable Engineering Analyzer - {APP_VERSION}")
        self.resize(1500, 980)
        self._apply_theme()
        self.last_result: CableAnalysisResult | None = None
        self.last_sweep: SweepResult | None = None

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

        self.recalculate()

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

        op_box = QGroupBox("Operating Point")
        op_layout = QFormLayout(op_box)
        self.operating_point_combo = QComboBox()
        for power_w in range(100, 2001, 100):
            self.operating_point_combo.addItem(f"{power_w} W")
        self.operating_point_combo.setCurrentText("1300 W")
        self.operating_point_combo.currentTextChanged.connect(self._on_operating_point_changed)
        op_layout.addRow("Select:", self.operating_point_combo)
        self.measured_load = QLabel("Load: -")
        self.measured_tune = QLabel("Tune: -")
        self.measured_iout = QLabel("Measured Iout - Untrusted: -")
        self.measured_iout.setStyleSheet("color: #fb7185;")
        op_layout.addRow(self.measured_load)
        op_layout.addRow(self.measured_tune)
        op_layout.addRow(self.measured_iout)

        sys_box = QGroupBox("Generator & Matcher")
        sys = QFormLayout(sys_box)
        self.freq_mhz = _line_edit("60", "System RF frequency in MHz.")
        self.p_gen_w = _line_edit("1300", "Generator output real power at generator reference plane [W].")
        self.z_gen = _line_edit("50", "Generator reference impedance [ohm].")
        self.p_match_out = _line_edit("1300", "Match Output real power at match-output reference plane [W].")
        self.match_eff = _line_edit("100", "Match efficiency [%] when using calculated mode.")
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
        sys.addRow("Generator Power [W]", self.p_gen_w)
        sys.addRow("Generator Z [ohm]", self.z_gen)
        sys.addRow("Match Output Power [W]", self.p_match_out)
        sys.addRow("Power Source", self.power_source)
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
        layout = QGridLayout(tab)

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
        f.addRow("R' [ohm/m]", self.rp)
        f.addRow("L' [H/m]", self.lp)
        f.addRow("G' [S/m]", self.gp)
        f.addRow("C' [F/m]", self.cp)

        spec_box = QGroupBox("Cable Specification / Preset")
        s = QFormLayout(spec_box)
        self.cable_preset = QComboBox()
        self.cable_preset.addItems(["Custom", "DX7 60 MHz Existing Cable"])
        self.cable_preset.currentTextChanged.connect(self._on_cable_preset_changed)
        self.spec_model = _line_edit("Unknown / Input Required", "Cable model name.")
        self.spec_part = _line_edit("Unknown / Input Required", "Cable part number.")
        self.spec_status = QLabel("Status: USER INPUT")
        s.addRow("Preset", self.cable_preset)
        s.addRow("Model", self.spec_model)
        s.addRow("Part Number", self.spec_part)
        s.addRow(self.spec_status)

        cards = QGroupBox("Power & Stress Summary")
        cg = QGridLayout(cards)
        self.card_labels: dict[str, QLabel] = {}
        for idx, (title, target, color) in enumerate(
            [
                ("MATCH OUTPUT POWER", "match_output", "#22d3ee"),
                ("CABLE DISSIPATION", "cable_loss", "#f97316"),
                ("CABLE END DELIVERED", "chamber", "#22c55e"),
                ("MAX VOLTAGE", "vmax", "#22d3ee"),
                ("MAX CURRENT", "imax", "#fb923c"),
                ("MAX LOCAL HEATING", "wmax", "#ef4444"),
            ]
        ):
            t = QLabel(f"{title}\n-")
            t.setStyleSheet(f"color:{color}; border:1px solid #334155; border-radius:8px; padding:10px;")
            self.card_labels[target] = t
            cg.addWidget(t, idx // 3, idx % 3)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color:#f87171;")
        self.live_metrics_label = QLabel(self._format_live_metrics(None, None))
        self.live_metrics_label.setStyleSheet("color:#93c5fd;")

        action_row = QHBoxLayout()
        self.run_btn_cable = QPushButton("Run Analysis")
        self.run_btn_cable.clicked.connect(self.recalculate)
        action_row.addWidget(self.run_btn_cable)
        action_row.addStretch(1)

        layout.addWidget(cable_box, 0, 0)
        layout.addWidget(spec_box, 1, 0)
        layout.addWidget(cards, 0, 1, 2, 1)
        layout.addWidget(self.warning_label, 2, 0, 1, 2)
        layout.addLayout(action_row, 3, 0, 1, 2)
        layout.addWidget(self.live_metrics_label, 4, 0, 1, 2)
        self._toggle_rlgc_fields()

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

    def _toggle_chamber_fields(self) -> None:
        direct = self.chamber_mode.currentText() != ChamberImpedanceMode.DERIVED_FROM_MATCH.value
        self.zl_r.setEnabled(direct)
        self.zl_x.setEnabled(direct)

    def _on_cable_preset_changed(self, text: str) -> None:
        if text == "DX7 60 MHz Existing Cable":
            self.length.setText("1.03")
            self.z0.setText("50")
            self.att.setText("0.6")
            self.spec_model.setText("AA11459-LCMR-158M-01.03M")
            self.spec_part.setText("1238-727-01")
            self.spec_status.setText("Status: DATASHEET/HISTORICAL")
        else:
            self.spec_status.setText("Status: USER INPUT")
        self.recalculate()

    def _on_matcher_mode_changed(self, text: str) -> None:
        self.s_mode_lbl.setText(f"Matcher Model: {text}")
        self.recalculate()

    def _on_operating_point_changed(self, text: str) -> None:
        if "W" in text:
            selected_power_w = float(text.replace("W", "").strip())
            self.p_gen_w.setText(f"{selected_power_w:g}")
            self.p_match_out.setText(f"{selected_power_w:g}")
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
        source = MatchOutputPowerSource(self.power_source.currentText())
        matcher_mode = MatcherModelMode(self.matcher_mode.currentText())
        system = SystemParameters(
            frequency_mhz=float(self.freq_mhz.text()),
            generator_power_w=float(self.p_gen_w.text()),
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
                cable_model=self.spec_model.text(),
                part_number=self.spec_part.text(),
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
            self._update_cards(result, system)
            all_warnings = valid.warnings + result.warnings
            if system.sparam_frequency_mhz is not None and abs(system.sparam_frequency_mhz - system.frequency_mhz) > 0.5:
                all_warnings.append("S-parameter frequency differs from simulation frequency.")
            self.warning_label.setText("\n".join(all_warnings))
            self._draw_distribution()
            self._update_summary(result, system)
            self._refresh_probe_table()
        except Exception as exc:
            self.last_result = None
            self.system_diag.setText(f"Input Required / Error: {exc}")
            self.live_metrics_label.setText(self._format_live_metrics(None, None))

    def _format_consistency_diagnostic(self, system: SystemParameters) -> str:
        if system.measured_vpp_v is None:
            return "Reference-plane / measurement consistency check: Vpp not provided."
        z = system.match_output_impedance.value
        zmag = abs(z)
        vrms = system.measured_vpp_v / (2.0 * np.sqrt(2.0))
        icalc = vrms / zmag if zmag else np.nan
        pcalc = (vrms**2 * z.real / (zmag**2)) if zmag else np.nan
        return (
            "Reference-plane / measurement consistency check\n"
            f"Zmag={zmag:.4f} ohm, Zphase={np.degrees(np.angle(z)):.2f} deg, "
            f"Vrms(Vpp)={vrms:.2f} V, Icalc={icalc:.2f} A, Pcalc={pcalc:.2f} W"
        )

    def _update_cards(self, r: CableAnalysisResult, system: SystemParameters) -> None:
        vmax_idx = int(np.argmax(np.abs(r.v_rms)))
        imax_idx = int(np.argmax(np.abs(r.i_rms)))
        wmax_idx = int(np.argmax(r.p_total_loss_w_per_m))
        self.card_labels["match_output"].setText(f"MATCH OUTPUT POWER\n{r.match_output_power_w:.2f} W")
        self.card_labels["cable_loss"].setText(f"CABLE DISSIPATION\n{r.cable_loss_w:.2f} W ({r.cable_loss_percent:.2f}%)")
        self.card_labels["chamber"].setText(f"CABLE END DELIVERED\n{r.chamber_power_w:.2f} W")
        self.card_labels["vmax"].setText(f"MAX VOLTAGE\n{np.abs(r.v_rms[vmax_idx]):.2f} Vrms @ {r.z_m[vmax_idx]:.3f} m")
        self.card_labels["imax"].setText(f"MAX CURRENT\n{np.abs(r.i_rms[imax_idx]):.2f} Arms @ {r.z_m[imax_idx]:.3f} m")
        self.card_labels["wmax"].setText(f"MAX LOCAL HEATING\n{r.p_total_loss_w_per_m[wmax_idx]:.3f} W/m @ {r.z_m[wmax_idx]:.3f} m")
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
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
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
        self._draw_distribution()

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
        self._draw_distribution()

    def _update_probe_readout(self) -> None:
        if self.last_result is None:
            return
        z = float(self.probe_pos.text())
        idx = self._position_index(z)
        r = self.last_result
        vrms = np.abs(r.v_rms[idx])
        irms = np.abs(r.i_rms[idx])
        pz = r.p_real_w[idx]
        local = r.p_total_loss_w_per_m[idx]
        cum = r.p_cumulative_loss_w[idx]
        self.probe_values.setText(
            f"Probe {z:.3f} m | Vrms={vrms:.2f} V | Irms={irms:.2f} A | "
            f"P={pz:.2f} W | Local={local:.3f} W/m | Cum={cum:.3f} W"
        )

    def _refresh_probe_table(self) -> None:
        if self.last_result is None:
            return
        self._update_probe_readout()
        r = self.last_result
        values = []
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
                    "z_m",
                    "V_rms",
                    "I_rms",
                    "P_real_W",
                    "P_cond_W_per_m",
                    "P_diel_W_per_m",
                    "P_total_loss_W_per_m",
                    "P_cumulative_loss_W",
                ]
            )
            for i in range(len(r.z_m)):
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
                    ]
                )

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
        eta_match = (r.match_output_power_w / r.generator_power_w * 100.0) if r.generator_power_w > 0 else 0.0
        eta_cable = (r.chamber_power_w / r.match_output_power_w * 100.0) if r.match_output_power_w > 0 else 0.0
        eta_total = (r.chamber_power_w / r.generator_power_w * 100.0) if r.generator_power_w > 0 else 0.0
        self.summary_label.setText(
            f"Reference planes:\n"
            f"Generator -> Matcher -> Match Output/Cable Input (z=0) -> Chamber (z=L)\n\n"
            f"Generator Power = {r.generator_power_w:.3f} W\n"
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
            f"Electrical length = {r.electrical_length_deg:.2f} deg"
        )
