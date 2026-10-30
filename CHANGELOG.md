# Changelog

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
