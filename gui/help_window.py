from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout


HELP_HTML = """
<h1>RF Cable Optimizer Help</h1>

<h2>1. RF Cable Optimizer Overview</h2>
<p>RF Cable Optimizer analyzes RF power flow and transmission-line behavior between a matching network and the chamber/load.</p>
<p>It can calculate and visualize: Match Input Power, Match Output Power, Matcher Loss, Cable Dissipation, Cable End Delivered Power, voltage/current distribution, standing-wave behavior, local and cumulative dissipation, cable impedance/electrical length, chamber-side impedance, and Z0 optimization tradeoffs.</p>

<h2>2. Reference Planes</h2>
<p>Match Input -> Matcher -> Match Output / Cable Input -> RF Cable -> Cable End / Chamber Side</p>
<p>Quantities measured at different reference planes must not be treated as though measured at the same location.</p>

<h2>3. Selecting an Operating Point</h2>
<p>The Operating Point selector loads measured operating-point data.</p>
<p>Example: selecting 1300 W means <b>Match Input Power = 1300 W</b>. Where available it also loads Load, Tune, Vpp, Iout, Match Output Resistance, and Match Output Reactance.</p>
<p><b>Measured Iout is historical reference data and is currently UNTRUSTED.</b></p>

<h2>4. Match Power</h2>
<p>Core terms: Match Input Power, Match Loss, Match Output Power, Matcher Efficiency.</p>
<p><code>P_match_output = P_match_input * eta_match</code></p>
<p>Match Output Power modes:</p>
<ul>
  <li>Calculated from Matcher Model</li>
  <li>Measured/User Input</li>
  <li>Ideal Matcher</li>
</ul>

<h2>5. Cable Setup</h2>
<p>Configure Cable Length, Z0, Velocity Factor, Attenuation, and attenuation units (Datasheet mode) or RLGC values (RLGC mode).</p>
<p><b>z = 0</b> is Match Output / Cable Input. <b>z = L</b> is Cable End / Chamber Side.</p>
<p>Preset selector supports DX7 Existing Cable, COMCRAFT AA-11459, and Custom. AA-11459 uses <b>Velocity Factor = 0.76</b> (manufacturer spec).</p>
<p>AA-11459 attenuation can be used in MANUFACTURER TABLE ONLY mode (no direct 60 MHz value) or INTERPOLATE BETWEEN PUBLISHED POINTS mode (explicitly labeled INTERPOLATED with source points).</p>

<h2>5A. Scrollable Cable and Connector Specification</h2>
<p>The left-side specification panel is vertically scrollable. Scrolling only changes the visible GUI area.</p>
<p>Scrolling does not change RF calculations and does not trigger analysis by itself.</p>

<h2>5B. Match-Side Connector / Transition</h2>
<p>The model supports three coordinate concepts: Connector Physical Length, Connector Electrical Length, and Flexible Cable Physical Coordinate.</p>
<p>Implemented connector modes:</p>
<ul>
  <li>Coordinate Offset Only: changes coordinate labels only; Smith trajectory and RF solution are unchanged.</li>
  <li>Transmission-Line Section: includes connector Z0, velocity factor, electrical length, and loss in the RF model.</li>
</ul>
<p>Nominal 50 ohm does not automatically guarantee zero real-hardware discontinuity.</p>
<p>Connector parameters must come from verified specification data or explicit user input.</p>

<h2>6. Cable Analysis</h2>
<p>Press <b>Run Analysis</b>. Key result cards include Match Output Power, Cable Dissipation, Cable End Delivered Power, Maximum Voltage, Maximum Current, Maximum Local Dissipation, Lowest-Impedance Point, and Maximum-Current Point.</p>
<p>Lowest-impedance point is defined by <code>argmin(|Z(z)|)</code> with <code>Z(z)=V(z)/I(z)</code>, not by reactance crossing alone.</p>
<p>Maximum-current point is defined by <code>argmax(|I(z)|)</code>. These two points can be close, but they are not forced to be identical.</p>

<h2>7. Voltage / Current Distribution</h2>
<p>Plots include Voltage RMS, Current RMS, Real RF Power, Local Dissipation, and Cumulative Dissipation versus cable position.</p>
<p>Local maxima can be important for heating or damage investigation. Current maximum alone does not prove a burn location.</p>

<h2>8. Probe Position</h2>
<p>Use <b>Probe Position [m]</b> to inspect Vrms, Vpp, Irms, Real Power, Local Dissipation, and Cumulative Loss at a specific location.</p>
<p>The slider and numeric field represent the same cable position.</p>

<h2>9. Smith Chart</h2>
<p>The Smith Chart shows complex impedance along the physical cable at the selected RF frequency (not a frequency sweep).</p>
<p><b>z = 0</b> is Match Output / Cable Input and <b>z = L</b> is Cable End / Chamber Side.</p>
<p>The colored trajectory is the impedance transformation from Match Output to Cable End; the yellow marker is the selected probe point.</p>
<p>Probe movement inspects a different cable location only; it does not change the cable model.</p>
<p>Changing Cable Length or cable/Z0 parameters recalculates the entire trajectory.</p>
<p>Local impedance is derived from distributed phasors: <code>Z(z) = V(z) / I(z)</code>.</p>
<p>The Smith normalization reference is the active cable Z0 (including complex Z0 in RLGC mode when applicable).</p>
<p>When shown, the red segment is the historical damage-location range and is an independent comparison overlay.</p>
<p>High current at one Smith point alone does not prove failure location; reflected power is not cable dissipation.</p>

<h2>10. Measurement Positions</h2>
<p>Enter comma-separated positions (example: <code>0.05,0.10,0.15,0.20,0.50,1.00</code>) to populate the measurement table.</p>

<h2>11. Historical Damage Region</h2>
<p>An optional shaded historical DX7 region may be displayed for reference only.</p>
<p><b>The software does not force the simulated maximum into this region.</b></p>

<h2>12. Z0 Optimization</h2>
<p>Define Min Z0, Max Z0, and Z0 Step. The application compares cable loss, max voltage/current, and delivered power.</p>
<p>The optimization score is a user-defined engineering tradeoff, not a physical law.</p>

<h2>13. Coax Geometry</h2>
<p><b>D</b>: inner diameter of outer conductor, <b>d</b>: outer diameter of inner conductor, <b>epsilon_r</b>: relative dielectric constant.</p>
<p><code>Z0 = (60 / sqrt(epsilon_r)) * ln(D/d)</code></p>
<p>This is an ideal TEM coax approximation.</p>
<p>Do not assume outer cable diameter or dielectric diameter is automatically equal to electrical <b>D</b>. If reliable D is unavailable, treat it as UNKNOWN / INPUT REQUIRED.</p>

<h2>14. Datasheet Mode vs RLGC Mode</h2>
<p>Datasheet mode uses manufacturer attenuation and does not automatically decompose conductor/dielectric loss.</p>
<p>RLGC mode uses R', L', G', C' and can estimate conductor and dielectric components.</p>
<p><b>Cond=0 and Diel=0 in Datasheet mode means decomposition is unavailable, not zero real loss.</b></p>

<h2>15. Important RF Physics Notes</h2>
<p>Reflected Power is not Cable Dissipation. High VSWR does not automatically mean high cable dissipation. A lossless cable can have a standing wave with zero real cable loss.</p>
<p>On a lossy cable, the Smith trajectory radius can change with position; do not force a constant-|Gamma| circle interpretation.</p>
<p>Cable heating depends on conductor/dielectric/contact losses and thermal conditions.</p>

<h2>16. Source Status</h2>
<p>Interpret status tags carefully: MEASURED, DATASHEET, DERIVED, USER INPUT, ASSUMED/IDEAL, UNTRUSTED.</p>
"""


class HelpWindow(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RF Cable Optimizer Help")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(980, 760)

        layout = QVBoxLayout(self)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(HELP_HTML)
        layout.addWidget(self.browser)

        button_row = QHBoxLayout()
        self.always_on_top = QCheckBox("Always on Top")
        self.always_on_top.setChecked(False)
        self.always_on_top.toggled.connect(self._on_always_on_top_toggled)
        button_row.addWidget(self.always_on_top)
        button_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _on_always_on_top_toggled(self, enabled: bool) -> None:
        self.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)
        self.show()
