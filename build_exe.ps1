$ErrorActionPreference = "Stop"

$python = "python"
$versionLine = Select-String -Path ".\version.py" -Pattern 'APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1

if (-not $versionLine) {
    throw "Unable to read APP_VERSION from version.py"
}

$version = $versionLine.Matches[0].Groups[1].Value
$version = $version.TrimStart("V", "v")

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "RF_Cable_Optimizer_v$version" `
    --icon "global_search_international.ico" `
    --add-data "ASM-logo-small.gif;." `
    "main.py"
