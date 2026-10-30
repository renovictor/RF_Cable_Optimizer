# RF Cable Optimizer

RF Cable Optimizer is a PySide6 desktop application for match-to-chamber RF cable engineering analysis, including cable loss, VSWR, power distribution, probe readouts, and Z0 sweep optimization.

## Features

- System input, cable analysis, voltage/current distribution, thermal analysis, and engineering summary tabs
- Cable attenuation and RLGC modeling
- Lumped thermal RC analysis with RF-derived heating power sources
- Z0 optimization sweeps with plots and CSV export
- Coax geometry helper calculations
- Startup splash screen with:
  - ASM logo, app name, version, department/company labels
  - Live module-loading status and animated progress bar
  - Borderless, centered, always-on-top startup window
  - Minimum 5-second visual startup feedback while modules preload

## Requirements

- Python 3.11+ (recommended)
- Windows (for provided `build_exe.ps1`)
- Runtime dependencies:
  - `PySide6`
  - `numpy`
  - `matplotlib`

Optional startup-preload modules (app continues if missing):

- `pyserial`, `pandas`, `openpyxl`, `Pillow`, `python-docx`, `guardian_tlog`, `jsonl_to_excel`

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

## Build single-file EXE (PyInstaller)

Use the included build script:

```powershell
.\build_exe.ps1
```

The script produces a one-file windowed executable named with the application version, and bundles:

- `ASM-logo-small.gif`
- `global_search_international.ico`

## Test

```powershell
python -m unittest discover -s tests -q
```
