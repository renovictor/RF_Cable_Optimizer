from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from gui.splash_screen import StartupCoordinator


def main() -> int:
    app = QApplication(sys.argv)
    app_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    icon_path = app_base / "smithchart.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    startup = StartupCoordinator(app)
    app.startup_coordinator = startup  # Keep a strong reference for app lifetime.
    startup.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
