# Changelog

## V2026.10.9 - 2026-10-31

- Updated matcher S-parameter workflow UX:
  - default S-parameter fields preloaded with the current regression dataset
  - S-parameter calculated mode no longer requires manual efficiency entry
  - Match Efficiency field now displays computed efficiency in USER S-PARAMETERS + CALCULATED mode
  - Load R/X display updates with active loaded ZL in non-manual load modes
- Clarified System Input labels:
  - `Historical Load R/X` renamed to `Load R/X`
- Added/updated tests for:
  - computed efficiency display behavior in S-parameter mode
  - active ZL-to-Load R/X UI synchronization behavior

## V2026.10.8 - 2026-10-31

- Added a major **RF STRESS COMPARISON** dashboard under Cable Analysis with new sub-tabs:
  - **Overview**
  - **RF Stress Comparison**
  - **Smith Chart**
- Added reusable stress-comparison analysis module:
  - `analysis/stress_comparison.py`
  - pure functions for matched-reference/current/voltage enhancement and I-squared stress index
  - `RFStressComparisonResult` model built from existing distributed RF outputs (no separate RF model)
- Added highly visual RF stress summaries:
  - three-column reference comparison (**MANUFACTURER REFERENCE**, **IDEAL MATCHED REFERENCE**, **ACTUAL DISTRIBUTED RESULT**)
  - shared-scale comparison charts for current, voltage, and local current-squared stress
  - physical cable stress map with markers and current/voltage overlay profiles
  - executive RF current-stress visualization and distributed power-flow panel
- Added synchronization and workflow improvements:
  - clickable stress elements route to probe/Smith updates
  - **OPEN THERMAL ANALYSIS AT HOT SPOT** action links RF hot-spot location to thermal source selection
- Extended exports:
  - stress-comparison summary fields added to profile CSV export
  - new **Export RF Stress Dashboard Image** action (PNG, Windows-safe naming)
- Updated Help content with a dedicated **RF Stress Comparison** section and interpretation/disclaimer guidance.
- Added/updated automated tests for stress calculations, GUI integration, CSV export coverage, and new cable-analysis sub-tab behavior.
- UI follow-up polish:
  - moved animation control to Cable Analysis sub-tab header (top-right)
  - made RF Stress Comparison panel scrollable so all charts remain visible on smaller windows
  - improved stress-chart text readability with stronger contrast and label backgrounds.

## V2026.10.7 - 2026-10-31

- Added a new **Thermal Analysis** tab with a standalone thermal module architecture:
  - `thermal/models.py` (thermal data models/enums)
  - `thermal/engine.py` (heating-source resolution + lumped RC solver)
  - `thermal/validation.py` (thermal-input validation)
  - `gui/thermal_plot.py` (thermal plotting)
- Implemented lumped thermal RC analysis driven by RF-derived dissipation, including dynamic heating-source selection and connector-loss availability rules.
- Added required thermal UX labeling/warnings (RF-derived heating source display, assumption labels for Rth/Cth, and estimate warning text).
- Added thermal tab relayout with independent vertical scroll areas for:
  - Thermal Heating Source panel
  - Lumped Thermal RC Parameters panel
- Added two new tabs:
  - **Top feed** (displays `Top_feed.png`)
  - **Bottom feed** (displays `Bottom_feed.png`)
- Added/updated automated tests for thermal engine behavior, thermal GUI behavior, and new feed-tab presence.

## V2026.10.6 - 2026-10-31

- Added scrollable Cable Analysis sub-panels with targeted layout refinements: dedicated scrolling for Power Flow, Critical-Point Cards, Smith Chart Summary, Probe Impedance Details, and cable-input area horizontal scrolling support.
- Added optional **Match-Side Connector / Transition** modeling with two implemented modes:
  - **Coordinate Offset Only** (coordinate relabeling only, RF solution unchanged)
  - **Transmission-Line Section** (connector included in cascaded RF path and power accounting)
- Extended distributed analysis outputs with connector-aware section coordinates, cable-start offsets, connector metrics (electrical length/delay), and connector-inclusive power-balance terms.
- Updated Smith Chart behavior and visualization for connector workflows, including segmented trajectories in transmission-line mode and restored probe interaction synchronization.
- Updated menus and UX:
  - Added **File** actions for profile CSV and Smith image export
  - Added **Analysis -> Run Analysis**
  - Added **Window** menu tab shortcuts
  - Added **Help -> About** dialog with version and project metadata
- Added/updated automated tests covering connector coordinates, connector transmission-line behavior, connector power balance, scrollable panel behavior, and related GUI regressions.

## V2026.10.5 - 2026-10-30

- Added new engineering markers and cards for **Lowest-Impedance Point** (`argmin |Z(z)|`) and **Maximum-Current Point** (`argmax |I(z)|`) using existing distributed RF analysis arrays.
- Added robust valid-point filtering (NaN/Inf and low-current exclusion via current epsilon) so invalid samples cannot be selected as engineering extrema.
- Added Smith Chart overlays and distribution-plot position markers for both new engineering points.
- Updated help content and expanded automated tests for the new point calculations and UI states (including `Not Available` handling).

## V2026.10.4 - 2026-10-30

- Added interactive Smith Chart integration on Cable Analysis using distributed V(z)/I(z)-derived impedance and Gamma arrays from the analysis engine.
- Added reusable Smith math and Smith chart modules, including probe synchronization, historical-region overlay handling, measurement markers, and Smith image export.
- Added Smith-related CSV profile columns, endpoint consistency diagnostics, and updated Help documentation.
- Added Smith math/distributed/GUI test coverage.

## V2026.10.3 - 2026-10-30

- Corrected power reference-plane terminology and calculations around Match Input / Match Output / Cable End Delivered power.
- Added File/Help menu bar and non-modal dedicated Help window.
- Added COMCRAFT AA-11459 preset support with source-status labeling, attenuation source modes, interpolation labeling, and power-rating reference display.

## V2026.10.1 - 2026-10-29

- Added live Cable Analysis readout for current VSWR and cable loss (percent and watts).
- Fixed Coax Geometry mode handling so MODE 2 and MODE 3 calculations trigger correctly from the UI.

## V2026.10.0 - 2026-10-29

- Added centralized application version file: `version.py`.
- Wired GUI title to use `APP_VERSION` from `version.py`.
- Added `CHANGELOG.md` for release tracking.
